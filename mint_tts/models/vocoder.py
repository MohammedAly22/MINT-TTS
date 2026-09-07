"""Vocoders. The vocoder is deliberately **fixed** for every experiment.

If the vocoder were trained or adapted alongside the acoustic model, any
quality/compute delta could come from either side and the experiment would not
isolate adaptive computation. So: one pretrained HiFi-GAN, frozen, shared by
every baseline and every adaptive variant. A Griffin-Lim fallback keeps the
whole pipeline runnable with zero downloads (useful for CI and quick checks,
not for quality claims).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                  padding=get_padding(kernel_size, d))) for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))) for _ in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            xt = c2(F.leaky_relu(xt, LRELU_SLOPE))
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in list(self.convs1) + list(self.convs2):
            remove_weight_norm(layer)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                  padding=get_padding(kernel_size, d))) for d in dilation
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class HiFiGANGenerator(nn.Module):
    """Reference HiFi-GAN generator (jik876/hifi-gan compatible state dicts)."""

    def __init__(self, h: dict):
        super().__init__()
        self.h = h
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.conv_pre = weight_norm(
            nn.Conv1d(h.get("num_mels", 80), h["upsample_initial_channel"], 7, 1, padding=3)
        )
        resblock = ResBlock1 if str(h.get("resblock", "1")) == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                h["upsample_initial_channel"] // (2 ** i),
                h["upsample_initial_channel"] // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2,
            )))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            for k, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"]):
                self.resblocks.append(resblock(ch, k, tuple(d)))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            xs = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = torch.tanh(self.conv_post(F.leaky_relu(x)))
        return x

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def normalise_generator_state(state: dict) -> dict:
    """Map third-party HiFi-GAN key layouts onto this generator.

    Two conventions are common in released checkpoints:

        jik876       conv_pre.weight_g
        SpeechBrain  conv_pre.conv.weight_g   (every conv wrapped in a module)

    Accepting both means a checkpoint can be pointed at directly, without a
    conversion step that silently produces a randomly-initialised vocoder.
    """
    for key in ("generator", "model", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    out = {}
    for k, v in state.items():
        for prefix in ("module.", "generator.", "model.g.", "hifi_gan."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        k = k.replace(".conv.", ".").replace(".conv_transpose.", ".")
        out[k] = v
    return out


DEFAULT_HIFIGAN_V1 = {
    "resblock": "1",
    "num_mels": 80,
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
}
DEFAULT_HIFIGAN_V3 = {
    "resblock": "2",
    "num_mels": 80,
    "upsample_rates": [8, 8, 4],
    "upsample_kernel_sizes": [16, 16, 8],
    "upsample_initial_channel": 256,
    "resblock_kernel_sizes": [3, 5, 7],
    "resblock_dilation_sizes": [[1, 2], [2, 6], [3, 12]],
}


class GriffinLimVocoder(nn.Module):
    """Zero-download fallback. Intelligible, not high fidelity."""

    def __init__(self, audio_cfg, n_iter: int = 32):
        super().__init__()
        from ..data.audio import AudioConfig, _mel_basis

        self.ac = audio_cfg if isinstance(audio_cfg, AudioConfig) else AudioConfig.from_cfg(audio_cfg)
        fb = _mel_basis(self.ac.n_fft, self.ac.n_mels, self.ac.sample_rate,
                        self.ac.fmin, self.ac.fmax or self.ac.sample_rate / 2, "cpu", "float32")
        self.register_buffer("inv_basis", torch.linalg.pinv(fb.T), persistent=False)
        self.n_iter = n_iter

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        import torchaudio

        squeeze = mel.dim() == 2
        if squeeze:
            mel = mel.unsqueeze(0)
        lin = torch.exp(mel)                                   # undo log compression
        spec = torch.matmul(self.inv_basis.to(mel.device), lin).clamp_min(1e-6)
        gl = torchaudio.transforms.GriffinLim(
            n_fft=self.ac.n_fft, hop_length=self.ac.hop_length, win_length=self.ac.win_length,
            power=1.0, n_iter=self.n_iter,
        ).to(mel.device)
        wav = gl(spec)
        return wav.unsqueeze(1) if wav.dim() == 2 else wav


class Vocoder(nn.Module):
    """Uniform wrapper: `Vocoder(cfg)(mel) -> waveform (B, 1, N)`."""

    def __init__(self, cfg, device: str | torch.device = "cpu"):
        super().__init__()
        v = cfg.vocoder
        self.name = v.get("name", "griffin_lim")
        self.sample_rate = cfg.audio.sample_rate
        self.device = torch.device(device)
        self.model: nn.Module

        if self.name == "hifigan":
            ckpt = v.get("checkpoint", "")
            cfg_path = v.get("config", "")
            h = dict(DEFAULT_HIFIGAN_V1)
            if cfg_path and Path(cfg_path).exists():
                h.update(json.loads(Path(cfg_path).read_text(encoding="utf-8")))
            elif v.get("variant", "v1") == "v3":
                h = dict(DEFAULT_HIFIGAN_V3)
            gen = HiFiGANGenerator(h)
            if ckpt and Path(ckpt).exists():
                raw = torch.load(ckpt, map_location="cpu", weights_only=False)
                gen.load_state_dict(normalise_generator_state(raw))
                gen.remove_weight_norm()
                self.loaded = True
            else:
                self.loaded = False
                if v.get("require_checkpoint", True):
                    raise FileNotFoundError(
                        f"HiFi-GAN checkpoint not found at '{ckpt}'. Run "
                        "`python scripts/download_vocoder.py` or set "
                        "vocoder.name=griffin_lim for a download-free fallback."
                    )
            self.model = gen.eval()
        elif self.name == "griffin_lim":
            self.model = GriffinLimVocoder(cfg.audio, v.get("griffin_lim_iters", 32)).eval()
            self.loaded = True
        else:
            raise ValueError(f"Unknown vocoder '{self.name}'")

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

    @torch.inference_mode()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        return self.model(mel.to(self.device)).clamp(-1, 1)

    @torch.inference_mode()
    def to_wav(self, mel: torch.Tensor) -> torch.Tensor:
        """(n_mels, T) or (B, n_mels, T) -> (N,) or (B, N)."""
        out = self(mel)
        out = out.squeeze(1)
        return out.squeeze(0) if out.size(0) == 1 and mel.dim() == 2 else out
