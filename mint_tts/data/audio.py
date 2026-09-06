"""Mel-spectrogram front/back end.

The STFT settings default to the HiFi-GAN LJSpeech recipe (22.05 kHz, n_fft
1024, hop 256, 80 mels, fmin 0, fmax 8000) so that off-the-shelf HiFi-GAN
checkpoints can be used as a *fixed* vocoder -- every quality delta we then
measure is attributable to the adaptive acoustic model, not to the vocoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float | None = 8000.0
    clip_val: float = 1e-5
    center: bool = False        # HiFi-GAN pads manually; center=False matches it
    max_wav_value: float = 32768.0
    trim_silence: bool = True
    trim_top_db: float = 60.0

    @classmethod
    def from_cfg(cls, cfg) -> "AudioConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(cfg).items() if k in known})


@lru_cache(maxsize=8)
def _mel_basis(n_fft: int, n_mels: int, sr: int, fmin: float, fmax: float, device: str, dtype: str):
    fb = AF.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=fmin,
        f_max=fmax,
        n_mels=n_mels,
        sample_rate=sr,
        norm="slaney",
        mel_scale="slaney",
    )
    return fb.to(device=device, dtype=getattr(torch, dtype))


@lru_cache(maxsize=8)
def _window(win_length: int, device: str, dtype: str):
    return torch.hann_window(win_length, device=device, dtype=getattr(torch, dtype))


def dynamic_range_compression(x: torch.Tensor, clip_val: float = 1e-5, C: float = 1.0) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    return torch.exp(x) / C


def mel_spectrogram(wav: torch.Tensor, ac: AudioConfig) -> torch.Tensor:
    """(B, N) or (N,) waveform in [-1, 1] -> (B, n_mels, T) log-mel."""
    squeeze = wav.dim() == 1
    if squeeze:
        wav = wav.unsqueeze(0)
    if wav.abs().max() > 1.01:
        raise ValueError("Waveform must be normalised to [-1, 1] before mel extraction")

    dtype = str(wav.dtype).split(".")[-1]
    device = str(wav.device)
    pad = int((ac.n_fft - ac.hop_length) / 2)
    if not ac.center:
        wav = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.stft(
        wav,
        n_fft=ac.n_fft,
        hop_length=ac.hop_length,
        win_length=ac.win_length,
        window=_window(ac.win_length, device, dtype),
        center=ac.center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    fb = _mel_basis(ac.n_fft, ac.n_mels, ac.sample_rate, ac.fmin, ac.fmax or ac.sample_rate / 2, device, dtype)
    mel = torch.matmul(spec.transpose(1, 2), fb).transpose(1, 2)
    mel = dynamic_range_compression(mel, ac.clip_val)
    return mel.squeeze(0) if squeeze else mel


def read_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """(channels, samples) float32 in [-1, 1] plus the native sample rate.

    Uses soundfile rather than torchaudio.load: recent torchaudio delegates
    I/O to torchcodec, which is an extra native dependency we do not want to
    require on Colab or on a low-end machine.
    """
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), int(sr)
    except Exception:
        wav, sr = torchaudio.load(str(path))
        return wav, int(sr)


def load_wav(path: str | Path, ac: AudioConfig, normalize: bool = True) -> torch.Tensor:
    wav, sr = read_audio(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != ac.sample_rate:
        wav = AF.resample(wav, sr, ac.sample_rate)
    wav = wav.squeeze(0)
    if normalize:
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * 0.95
    return wav


def trim_silence(wav: torch.Tensor, ac: AudioConfig) -> torch.Tensor:
    """Energy-based trim (librosa.effects.trim equivalent, no librosa dep)."""
    frame = ac.win_length
    hop = ac.hop_length
    if wav.numel() < frame:
        return wav
    frames = wav.unfold(0, frame, hop)
    rms = frames.pow(2).mean(-1).clamp_min(1e-12).sqrt()
    db = 20.0 * torch.log10(rms / rms.max().clamp_min(1e-12))
    keep = (db > -ac.trim_top_db).nonzero().flatten()
    if keep.numel() == 0:
        return wav
    start = int(keep[0]) * hop
    end = min(int(keep[-1]) * hop + frame, wav.numel())
    return wav[start:end]


def compute_pitch(wav: torch.Tensor, ac: AudioConfig, n_frames: int) -> torch.Tensor:
    """Frame-level F0 via torchaudio's YIN-like detector, resampled to mel frames."""
    try:
        f0 = AF.detect_pitch_frequency(
            wav.unsqueeze(0), sample_rate=ac.sample_rate,
            frame_time=ac.hop_length / ac.sample_rate, freq_low=60, freq_high=600,
        ).squeeze(0)
    except Exception:
        return torch.zeros(n_frames)
    f0 = torch.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    return _resize_1d(f0, n_frames)


def compute_energy(mel: torch.Tensor) -> torch.Tensor:
    """L2 norm of the linear-domain mel per frame -> (T,)."""
    return dynamic_range_decompression(mel).norm(dim=0)


def _resize_1d(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.numel() == n:
        return x
    if x.numel() == 0:
        return torch.zeros(n)
    return torch.nn.functional.interpolate(
        x.view(1, 1, -1), size=n, mode="linear", align_corners=False
    ).view(-1)


def save_wav(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().float().clamp(-1, 1)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    try:
        import soundfile as sf

        sf.write(str(path), wav.transpose(0, 1).numpy(), int(sample_rate), subtype="PCM_16")
    except Exception:
        torchaudio.save(str(path), wav, sample_rate)


def np_save(path: str | Path, array: torch.Tensor | np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    np.save(str(path), array.astype(np.float32))
