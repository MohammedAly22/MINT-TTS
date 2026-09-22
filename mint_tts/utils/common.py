"""Seeding, devices, checkpoints, EMA and small helpers."""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 1234, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    try:
        import platform

        return platform.processor() or "CPU"
    except Exception:
        return "CPU"


def move_to(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class EMA:
    """Exponential moving average of model weights (evaluated, not trained)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup: dict = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        # One fused multi-tensor op instead of a Python loop over ~200 tensors
        # (and a fresh state_dict() walk) on every optimiser step.
        if getattr(self, "_pairs_model", None) is not model:
            live = model.state_dict()
            keys = [k for k in self.shadow if k in live]
            self._shadow_list = [self.shadow[k] for k in keys]
            self._live_list = [live[k] for k in keys]
            self._same_dtype = all(a.dtype == b.dtype for a, b in
                                   zip(self._shadow_list, self._live_list))
            self._pairs_model = model
        if self._same_dtype:
            torch._foreach_lerp_(self._shadow_list, self._live_list, 1 - self.decay)
        else:
            for sh, v in zip(self._shadow_list, self._live_list):
                sh.lerp_(v.float(), 1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(),
                               **{k: v.to(dtype=model.state_dict()[k].dtype)
                                  for k, v in self.shadow.items()}}, strict=False)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self.backup:
            model.load_state_dict({**model.state_dict(), **self.backup}, strict=False)
            self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state.get("decay", self.decay)
        self.shadow = {k: v.float() for k, v in state.get("shadow", {}).items()}
        self._pairs_model = None      # the cached tensor lists point at the old dict


def save_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None,
                    ema=None, step: int = 0, epoch: int = 0, cfg=None, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": cfg.to_dict() if cfg is not None and hasattr(cfg, "to_dict") else cfg,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if extra:
        payload["extra"] = extra
    # Write-then-rename: a Colab disconnect mid-save (checkpoints often live
    # on Google Drive) must leave the previous file intact, not a truncated one.
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def warm_start_weights(path: str | Path, prefer_ema: bool = True) -> dict:
    """A model state_dict from a checkpoint, for starting a NEW run from it.

    The EMA weights are preferred: they are what validation and synthesis
    use, so they are the model whose quality was actually measured. Only
    weights are returned -- a new stage gets a fresh optimiser, schedule and
    step counter.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(ckpt.get("model", ckpt))
    shadow = (ckpt.get("ema") or {}).get("shadow", {}) if prefer_ema else {}
    for k, v in shadow.items():
        if k in state:
            state[k] = v.to(state[k].dtype)
    return state


def checkpoint_step(path: str | Path) -> int:
    """The training step stored in a checkpoint, read without loading it all."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return -1
    return int(ckpt.get("step", -1))


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    """The most advanced checkpoint in a run directory (last/step_*/final/best).

    Chosen by the step stored inside, not by name or mtime: `best.pt` can be
    far older than `last.pt`, and Drive syncing rewrites mtimes.
    """
    directory = Path(directory)
    if not directory.exists():
        return None
    best, best_step = None, -1
    for f in directory.glob("*.pt"):
        step = checkpoint_step(f)
        if step > best_step:
            best, best_step = f, step
    return best


def load_checkpoint(path: str | Path, model=None, optimizer=None, scheduler=None, scaler=None,
                    ema=None, map_location="cpu", strict: bool = True) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt


def prune_checkpoints(directory: str | Path, keep: int = 5, pattern: str = "step_*.pt") -> None:
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime)
    for old in files[:-keep] if keep > 0 else []:
        old.unlink(missing_ok=True)


@contextmanager
def timer():
    start = time.perf_counter()
    box = {"elapsed": 0.0}
    try:
        yield box
    finally:
        box["elapsed"] = time.perf_counter() - start


def gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "(empty)"
    columns = columns or list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(f"{r.get(c, '')}") for r in rows)) for c in columns}
    line = "| " + " | ".join(str(c).ljust(widths[c]) for c in columns) + " |"
    sep = "|-" + "-|-".join("-" * widths[c] for c in columns) + "-|"
    body = [
        "| " + " | ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in columns) + " |"
        for r in rows
    ]
    return "\n".join([line, sep] + body)
