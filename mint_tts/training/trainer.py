"""Training loop for AdaptiveTTS."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import build_dataloader, resolve_dataset_fields
from ..evaluation.asr import ASRScorer
from ..evaluation.metrics import aggregate, mel_cepstral_distortion
from ..evaluation.mos import MOSPredictor, quality_score
from ..losses.compute import ComputeLoss, sample_budget
from ..losses.tts_losses import TTSLoss
from ..models.tts import build_model
from ..models.vocoder import Vocoder
from ..text.tokenizer import build_text_processor
from ..utils.common import (
    EMA,
    device_name,
    gpu_memory_mb,
    load_checkpoint,
    move_to,
    prune_checkpoints,
    resolve_device,
    save_checkpoint,
    set_seed,
)
from ..utils.flops import human, parameter_table
from ..utils.logging_utils import ExperimentLogger
from .monitors import ComplexityProbe, log_training_examples


def build_scheduler(optimizer, cfg, total_steps: int):
    s = cfg.train.scheduler
    warmup = int(s.get("warmup_steps", 4000))
    kind = s.get("name", "noam")

    if kind == "noam":
        d = cfg.model.d_model
        def fn(step):
            step = max(step, 1)
            return (d ** -0.5) * min(step ** -0.5, step * warmup ** -1.5) / (d ** -0.5 * warmup ** -0.5)
    elif kind == "cosine":
        min_ratio = float(s.get("min_lr_ratio", 0.05))
        def fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            p = (step - warmup) / max(total_steps - warmup, 1)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    elif kind == "constant":
        def fn(step):
            return min(1.0, step / max(warmup, 1))
    else:
        raise ValueError(f"Unknown scheduler '{kind}'")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class Trainer:
    def __init__(self, cfg, resume: str | None = None):
        self.cfg = cfg
        set_seed(cfg.get("seed", 1234), cfg.train.get("deterministic", False))
        self.device = resolve_device(cfg.train.get("device", "auto"))
        self.run_dir = Path(cfg.train.output_dir) / cfg.log.get("run_name", "run")
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExperimentLogger(cfg, self.run_dir)
        self.log = self.logger.log


        # The vocabulary is fixed by preprocessing; training must not invent
        # new symbols, or the embedding table and the data would disagree.
        symbols_file = Path(cfg.data.preprocessed_dir) / "symbols.json"
        if not symbols_file.exists():
            raise FileNotFoundError(
                f"{symbols_file} not found. Run scripts/preprocess.py first -- it "
                "builds the symbol table that the model's embedding is sized from."
            )
        self.tp = build_text_processor(cfg, symbols=symbols_file)
        self.tp.freeze()
        self.tp.save_symbols(self.run_dir / "symbols.json")
        resolve_dataset_fields(cfg, Path(cfg.data.preprocessed_dir))
        cfg.dump(self.run_dir / "config.yaml")
        self.model = build_model(cfg, self.tp.vocab_size).to(self.device)
        self.loss_fn = TTSLoss(cfg)
        self.compute_loss = ComputeLoss(cfg)

        self.train_loader = build_dataloader(cfg, cfg.data.train_index, train=True)
        self.val_loader = build_dataloader(cfg, cfg.data.val_index, train=False)

        p = cfg.train
        decay, no_decay = [], []
        for n, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim <= 1 or "emb" in n else decay).append(param)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": p.get("weight_decay", 0.01)},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=p.lr, betas=tuple(p.get("betas", [0.9, 0.98])), eps=p.get("eps", 1e-9),
        )
        self.total_steps = int(p.max_steps)
        self.scheduler = build_scheduler(self.optimizer, cfg, self.total_steps)
        self.use_amp = bool(p.get("amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = EMA(self.model, p.get("ema_decay", 0.999)) if p.get("use_ema", True) else None
        self.grad_accum = int(p.get("grad_accum", 1))

        self.vocoder = None
        if cfg.log.get("log_audio", True):
            try:
                self.vocoder = Vocoder(cfg, device=self.device)
            except Exception as exc:
                self.log.warning(f"Vocoder unavailable ({exc}); audio logging disabled.")

        self.probe = ComplexityProbe(cfg, self.tp, self.device)
        self.asr = ASRScorer(cfg.eval.get("asr_backend", "none"), device="cpu")
        self.mos = MOSPredictor(cfg.eval.get("mos_backend", "proxy"), device="cpu")

        self.step = 0
        self.epoch = 0
        self.best_val = float("inf")
        if resume:
            ckpt = load_checkpoint(resume, self.model, self.optimizer, self.scheduler,
                                   self.scaler, self.ema, map_location=self.device)
            self.step = int(ckpt.get("step", 0))
            self.epoch = int(ckpt.get("epoch", 0))
            self.log.info(f"Resumed from {resume} at step {self.step}")

        self._log_model_summary()

    # -- setup logging ----------------------------------------------------
    def _log_model_summary(self) -> None:
        table = parameter_table(self.model)
        total = table["TOTAL"]
        self.log.info(f"Device: {self.device} ({device_name(self.device)})")
        self.log.info(f"Parameters: {human(total)} ({total:,})")
        for k, v in sorted(table.items(), key=lambda kv: -kv[1]):
            if k != "TOTAL":
                self.log.info(f"  {k:<20s} {human(v):>10s}")
        self.log.info(
            f"Encoder routing={self.model.encoder.routing} max_steps={self.model.encoder.max_steps} "
            f"shared={self.model.encoder.share_weights} | "
            f"Decoder routing={self.model.decoder.routing} max_steps={self.model.decoder.max_steps} "
            f"shared={self.model.decoder.share_weights}"
        )
        self.log.info(f"Train utterances: {len(self.train_loader.dataset)} "
                      f"(filtered {self.train_loader.dataset.filtered}) | "
                      f"Val: {len(self.val_loader.dataset)}")
        self.logger.save_json("model_summary.json",
                              {"parameters": table, "device": str(self.device),
                               "device_name": device_name(self.device)})
        self.logger.log_scalars({"model/parameters": float(total)}, 0)

    # -- one step ---------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        batch = move_to(batch, self.device)
        B = batch["tokens"].size(0)
        c = self.cfg.loss.compute
        budget = sample_budget(
            B, self.device,
            q_range=tuple(c.get("q_range", [0.0, 1.0])),
            h_range=tuple(c.get("h_range", [0.3, 1.0])),
            use_hardware=bool(c.get("use_hardware_cap", True)),
        )
        amp_dtype = torch.bfloat16 if self.cfg.train.get("bf16", False) else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.use_amp):
            out = self.model(
                batch["tokens"], batch["token_lens"], mels=batch["mel"], mel_lens=batch["mel_lens"],
                pitch=batch.get("pitch"), energy=batch.get("energy"),
                speakers=batch.get("speakers"), emotions=batch.get("emotions"),
                budget=budget, attn_prior=batch.get("attn_prior"),
            )
            recon, logs = self.loss_fn(out, batch, self.step)
            comp, clogs = self.compute_loss(out, budget, self.step, batch.get("c_star"))
            loss = recon + comp
        logs.update(clogs)
        logs["loss/total"] = loss.detach()

        self.scaler.scale(loss / self.grad_accum).backward()
        if (self.step + 1) % self.grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            gn = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.get("grad_clip", 1.0))
            logs["train/grad_norm"] = gn.detach()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)

        logs["train/lr"] = self.optimizer.param_groups[0]["lr"]
        logs["train/encoder_depth"] = out.encoder_router.mean_depth().detach()
        logs["train/decoder_depth"] = out.decoder_router.mean_depth().detach()
        logs["train/encoder_ponder"] = out.encoder_router.mean_ponder().detach()
        logs["train/decoder_ponder"] = out.decoder_router.mean_ponder().detach()
        flops = self.model.flops(out)
        logs["train/flops_per_batch"] = flops.total
        logs["train/flops_saving"] = flops.saving
        return {"logs": logs, "out": out, "batch": batch}

    # -- validation -------------------------------------------------------
    @torch.inference_mode()
    def validate(self, max_batches: int | None = None) -> dict:
        self.model.eval()
        if self.ema is not None and self.cfg.train.get("eval_with_ema", True):
            self.ema.apply_to(self.model)
        n = max_batches or self.cfg.train.get("val_batches", 10)
        agg: dict[str, list[float]] = {}
        q_eval = float(self.cfg.eval.get("quality_budget", 0.9))
        mcds, cers, wers, mos_scores, computes, flops_list = [], [], [], [], [], []

        for i, batch in enumerate(self.val_loader):
            if i >= n:
                break
            batch = move_to(batch, self.device)
            B = batch["tokens"].size(0)
            budget = sample_budget(B, self.device, fixed_q=q_eval, fixed_h=1.0)
            out = self.model(
                batch["tokens"], batch["token_lens"], mels=batch["mel"], mel_lens=batch["mel_lens"],
                pitch=batch.get("pitch"), energy=batch.get("energy"),
                speakers=batch.get("speakers"), emotions=batch.get("emotions"),
                budget=budget, attn_prior=batch.get("attn_prior"),
            )
            loss, logs = self.loss_fn(out, batch, self.step)
            comp, clogs = self.compute_loss(out, budget, self.step)
            for k, v in {**logs, **clogs}.items():
                agg.setdefault("val/" + k.split("/", 1)[-1], []).append(float(v))
            agg.setdefault("val/total", []).append(float(loss + comp))

            flops = self.model.flops(out)
            flops_list.append(flops.total / max(B, 1))
            computes.append(float(out.encoder_router.per_utterance_depth().mean()
                                  / self.model.encoder.max_steps))

            k = min(self.cfg.eval.get("n_audio_eval", 2), B)
            for j in range(k):
                L = int(batch["mel_lens"][j])
                pred = out.mel_post[j, :, :L].float().cpu()
                tgt = batch["mel"][j, :, :L].float().cpu()
                mcd = mel_cepstral_distortion(pred, tgt)
                mcds.append(mcd)
                if self.vocoder is not None and self.asr.available:
                    wav = self.vocoder.to_wav(pred.to(self.device))
                    sc = self.asr.score(wav, self.cfg.audio.sample_rate, batch["text"][j])
                    wers.append(sc["wer"])
                    cers.append(sc["cer"])
                    if self.mos.backend == "utmos":
                        mos_scores.append(self.mos.score_wav(wav, self.cfg.audio.sample_rate))

        metrics = {k: float(np.mean(v)) for k, v in agg.items() if v}
        metrics["val/mcd"] = aggregate(mcds)["mean"]
        if wers:
            metrics["val/wer"] = aggregate(wers)["mean"]
            metrics["val/cer"] = aggregate(cers)["mean"]
        if mos_scores:
            metrics["val/mos"] = aggregate(mos_scores)["mean"]
        else:
            metrics["val/mos_proxy"] = MOSPredictor.proxy_from_metrics(
                metrics["val/mcd"], metrics.get("val/cer"))
        metrics["val/quality_score"] = quality_score({
            "mcd": metrics.get("val/mcd", float("nan")),
            "cer": metrics.get("val/cer", float("nan")),
            "mos": metrics.get("val/mos", metrics.get("val/mos_proxy", float("nan"))),
        })
        metrics["val/compute_norm"] = float(np.mean(computes)) if computes else float("nan")
        metrics["val/flops_per_utterance"] = float(np.mean(flops_list)) if flops_list else float("nan")

        if self.ema is not None and self.cfg.train.get("eval_with_ema", True):
            self.ema.restore(self.model)
        self.model.train()
        return metrics

    # -- main loop --------------------------------------------------------
    def fit(self) -> None:
        cfg = self.cfg
        self.model.train()
        t0 = time.time()
        running: dict[str, float] = {}
        count = 0
        log_every = cfg.log.get("log_every", 50)
        self.log.info(f"Starting training for {self.total_steps} steps")

        while self.step < self.total_steps:
            self.epoch += 1
            if hasattr(self.train_loader, "batch_sampler") and hasattr(
                    self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(self.epoch)
            for batch in self.train_loader:
                if self.step >= self.total_steps:
                    break
                res = self.train_step(batch)
                for k, v in res["logs"].items():
                    running[k] = running.get(k, 0.0) + float(v)
                count += 1
                self.step += 1

                if self.step % log_every == 0:
                    means = {k: v / count for k, v in running.items()}
                    means["train/steps_per_sec"] = count / max(time.time() - t0, 1e-6)
                    means["train/epoch"] = self.epoch
                    if self.device.type == "cuda":
                        means["train/gpu_mem_mb"] = gpu_memory_mb(self.device)
                    self.logger.log_scalars(means, self.step)
                    self.log.info(
                        f"step {self.step:>7d} | loss {means.get('loss/total', 0):.4f} "
                        f"| mel {means.get('loss/mel_post', 0):.4f} "
                        f"| enc_depth {means.get('train/encoder_depth', 0):.2f} "
                        f"| dec_depth {means.get('train/decoder_depth', 0):.2f} "
                        f"| saving {means.get('train/flops_saving', 0) * 100:.1f}% "
                        f"| {means['train/steps_per_sec']:.2f} it/s"
                    )
                    running, count, t0 = {}, 0, time.time()

                if self.step % cfg.log.get("figure_every", 1000) == 0:
                    log_training_examples(self.model, res["batch"], res["out"], self.logger,
                                          self.step, self.vocoder, cfg.audio.sample_rate,
                                          n=cfg.log.get("n_figure_examples", 2))
                if self.step % cfg.log.get("probe_every", 1000) == 0:
                    self.probe.run(self.model, self.logger, self.step, self.vocoder,
                                   hard=cfg.log.get("probe_hard_routing", True))
                if self.step % cfg.train.get("val_every", 2000) == 0:
                    metrics = self.validate()
                    self.logger.log_scalars(metrics, self.step)
                    self.log.info(
                        f"[val] step {self.step} | total {metrics.get('val/total', float('nan')):.4f} "
                        f"| mcd {metrics.get('val/mcd', float('nan')):.3f} "
                        f"| Q {metrics.get('val/quality_score', float('nan')):.3f} "
                        f"| compute {metrics.get('val/compute_norm', float('nan')):.3f}"
                    )
                    if metrics.get("val/total", float("inf")) < self.best_val:
                        self.best_val = metrics["val/total"]
                        save_checkpoint(self.ckpt_dir / "best.pt", self.model, self.optimizer,
                                        self.scheduler, self.scaler, self.ema, self.step,
                                        self.epoch, self.cfg, {"metrics": metrics})
                if self.step % cfg.train.get("save_every", 5000) == 0:
                    save_checkpoint(self.ckpt_dir / f"step_{self.step}.pt", self.model,
                                    self.optimizer, self.scheduler, self.scaler, self.ema,
                                    self.step, self.epoch, self.cfg)
                    prune_checkpoints(self.ckpt_dir, cfg.train.get("keep_checkpoints", 3))

        save_checkpoint(self.ckpt_dir / "final.pt", self.model, self.optimizer, self.scheduler,
                        self.scaler, self.ema, self.step, self.epoch, self.cfg)
        self.log.info("Training finished.")
        self.logger.close()
