"""Unified experiment logger: TensorBoard and/or Weights & Biases.

Both backends receive the same calls, so switching is a config flag and the
monitoring code never branches.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import plotting
from .plotting import close as close_fig
from .plotting import to_image_arrays

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "mint_tts", level: int = logging.INFO,
               log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
    logger.addHandler(handler)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class ExperimentLogger:
    def __init__(self, cfg, run_dir: str | Path, resume_id: str | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.figure_scale = float(cfg.log.get("figure_scale", 1.6))
        self.figure_backend = plotting.set_backend(cfg.log.get("figure_backend", "matplotlib"))
        self._warned_kaleido = False
        self._pending: list[tuple] = []
        backends = cfg.log.get("backends", ["tensorboard"])
        self.backends = list(backends)
        self.tb = None
        self.wandb = None
        self.log = get_logger("mint_tts", log_file=self.run_dir / "train.log")

        if "tensorboard" in self.backends:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception as exc:  # pragma: no cover
                self.log.warning(f"TensorBoard unavailable: {exc}")
        if "wandb" in self.backends:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project=cfg.log.get("wandb_project", "adaptive-speech"),
                    entity=cfg.log.get("wandb_entity", None),
                    name=cfg.log.get("run_name", self.run_dir.name),
                    group=cfg.log.get("wandb_group", None),
                    tags=list(cfg.log.get("wandb_tags", [])),
                    dir=str(self.run_dir),
                    config=cfg.to_dict(),
                    id=resume_id,
                    resume="allow" if resume_id else None,
                )
            except Exception as exc:  # pragma: no cover
                self.log.warning(f"wandb unavailable: {exc}")
                self.wandb = None

    # -- scalars ----------------------------------------------------------
    def log_scalars(self, values: dict, step: int) -> None:
        clean = {}
        for k, v in values.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                clean[k] = fv
        if self.tb is not None:
            for k, v in clean.items():
                self.tb.add_scalar(k, v, step)
        if self.wandb is not None and clean:
            self.wandb.log(clean, step=step)

    def log_figure(self, tag: str, fig, step: int) -> None:
        """Log a plotly figure.

        W&B keeps it interactive; TensorBoard has no plotly renderer, so the
        figure is rasterised with kaleido and logged as an image. If kaleido
        is missing we say so once rather than silently dropping the panel.
        """
        is_plotly = fig.__class__.__module__.startswith("plotly")
        if self.wandb is not None:
            self.wandb.log(
                {tag: (self.wandb.Plotly(fig) if is_plotly else self.wandb.Image(fig))},
                step=step,
            )
        if self.tb is None:
            if not is_plotly:
                close_fig(fig)
            return
        if is_plotly:
            self._pending.append((tag, fig, step))   # batched, kaleido is slow
        else:
            # TensorBoard rasterises matplotlib itself: no browser, no kaleido.
            self.tb.add_figure(tag, fig, step, close=True)

    def flush_figures(self) -> None:
        """Rasterise and write every buffered figure in one batch.

        kaleido costs ~2.7 s for a single figure but ~0.6 s each when batched,
        so monitors queue their figures and flush once per pass instead of
        paying the browser start-up cost per panel.
        """
        if self.tb is None or not self._pending:
            return
        tags, figs, steps = zip(*self._pending)
        self._pending = []
        arrays = to_image_arrays(list(figs), scale=self.figure_scale)
        for tag, step, arr in zip(tags, steps, arrays):
            if arr is not None:
                self.tb.add_image(tag, arr, step)
            elif not self._warned_kaleido:
                self._warned_kaleido = True
                self.log.warning(
                    "Static image export failed, so figures cannot reach "
                    "TensorBoard. Install it with `pip install kaleido`, or log "
                    "to W&B, which renders plotly figures natively."
                )

    def log_audio(self, tag: str, wav, step: int, sample_rate: int) -> None:
        arr = wav.detach().cpu().float().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        arr = arr.squeeze()
        if arr.ndim > 1:
            arr = arr[0]
        if self.tb is not None:
            self.tb.add_audio(tag, arr[None, :], step, sample_rate=sample_rate)
        if self.wandb is not None:
            self.wandb.log({tag: self.wandb.Audio(arr, sample_rate=sample_rate)}, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self.tb is not None:
            self.tb.add_text(tag, text, step)
        if self.wandb is not None:
            self.wandb.log({tag: self.wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def log_table(self, tag: str, columns: list[str], rows: list[list], step: int) -> None:
        if self.wandb is not None:
            self.wandb.log({tag: self.wandb.Table(columns=columns, data=rows)}, step=step)
        if self.tb is not None:
            head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
            body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
            self.tb.add_text(tag, head + body, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        arr = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
        if arr.size == 0:
            return
        if self.tb is not None:
            self.tb.add_histogram(tag, arr, step)
        if self.wandb is not None:
            self.wandb.log({tag: self.wandb.Histogram(arr)}, step=step)

    def save_json(self, name: str, obj) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return path

    def close(self) -> None:
        self.flush_figures()
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()
