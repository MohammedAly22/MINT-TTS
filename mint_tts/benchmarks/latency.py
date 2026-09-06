"""Wall-clock / FLOP / memory benchmarking.

FLOPs alone do not sell a system: a 1B-parameter model with 100M active
parameters is not automatically faster than a 300M dense model. So the
benchmark reports latency, real-time factor, peak memory *and* FLOPs, on
whichever device you point it at -- CPU and a low-end GPU are the targets
that matter for this project.
"""

from __future__ import annotations

import gc
import platform
import statistics
import time
from dataclasses import asdict, dataclass

import torch


@dataclass
class BenchResult:
    device: str
    device_name: str
    routing: str
    budget_q: float
    budget_h: float
    batch_size: int
    n_tokens: int
    n_frames: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    audio_seconds: float
    rtf: float
    flops: float
    flops_dense: float
    flops_saving: float
    encoder_compute: float
    decoder_compute: float
    peak_mem_mb: float
    threads: int

    def as_row(self) -> dict:
        return asdict(self)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or "cpu"


@torch.inference_mode()
def benchmark_model(
    model,
    tokens: torch.Tensor,
    token_lens: torch.Tensor,
    device: torch.device,
    quality: float = 0.9,
    hardware: float = 1.0,
    hard: bool = True,
    warmup: int = 3,
    iters: int = 20,
    hop_length: int = 256,
    sample_rate: int = 22050,
    threads: int | None = None,
) -> BenchResult:
    if threads and device.type == "cpu":
        torch.set_num_threads(int(threads))
    model = model.to(device).eval()
    tokens, token_lens = tokens.to(device), token_lens.to(device)
    B = tokens.size(0)
    budget = torch.tensor([[quality, hardware]] * B, device=device)

    for _ in range(warmup):
        model(tokens, token_lens, budget=budget, hard=hard)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    times = []
    out = None
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model(tokens, token_lens, budget=budget, hard=hard)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    flops = model.flops(out)
    n_frames = int(out.mel_mask.sum().item())
    audio_seconds = n_frames * hop_length / sample_rate
    mean_ms = statistics.fmean(times)
    peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == "cuda" else float("nan")

    return BenchResult(
        device=str(device),
        device_name=device_label(device),
        routing=f"{model.encoder.routing}/{model.decoder.routing}",
        budget_q=quality,
        budget_h=hardware,
        batch_size=B,
        n_tokens=int(token_lens.sum().item()),
        n_frames=n_frames,
        latency_ms_mean=mean_ms,
        latency_ms_p50=statistics.median(times),
        latency_ms_p90=sorted(times)[max(0, int(0.9 * len(times)) - 1)],
        audio_seconds=audio_seconds,
        rtf=(mean_ms / 1000.0) / max(audio_seconds, 1e-9),
        flops=flops.total,
        flops_dense=flops.dense_equivalent,
        flops_saving=flops.saving,
        encoder_compute=float(out.encoder_router.per_utterance_depth().mean()
                              / model.encoder.max_steps),
        decoder_compute=float(out.decoder_router.per_utterance_depth().mean()
                              / model.decoder.max_steps),
        peak_mem_mb=peak,
        threads=torch.get_num_threads() if device.type == "cpu" else 0,
    )


@torch.inference_mode()
def benchmark_vocoder(vocoder, mel: torch.Tensor, device: torch.device,
                      warmup: int = 2, iters: int = 10, hop_length: int = 256,
                      sample_rate: int = 22050) -> dict:
    for _ in range(warmup):
        vocoder.to_wav(mel)
    _sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        vocoder.to_wav(mel)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    audio_seconds = mel.shape[-1] * hop_length / sample_rate
    mean_ms = statistics.fmean(times)
    return {
        "vocoder": vocoder.name,
        "latency_ms_mean": mean_ms,
        "rtf": (mean_ms / 1000.0) / max(audio_seconds, 1e-9),
        "audio_seconds": audio_seconds,
    }
