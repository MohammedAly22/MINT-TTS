"""Dataset / collate / samplers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..modules.aligner import beta_binomial_prior


def resolve_dataset_fields(cfg, preprocessed_dir: Path) -> dict:
    """Fill in config values that only the preprocessed corpus knows.

    `model.n_speakers: auto` and `model.n_emotions: auto` become the real
    counts from `stats.json`, so a multi-speaker run does not need the number
    hard-coded in two places (and cannot silently disagree with the data).
    """
    stats_path = Path(preprocessed_dir) / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    for key, stat_key in (("n_speakers", "n_speakers"), ("n_emotions", "n_emotions")):
        if str(cfg.model.get(key, 1)).lower() == "auto":
            cfg.model[key] = int(stats.get(stat_key, 1))
    return stats


class TTSDataset(Dataset):
    def __init__(self, index_path: str | Path, cfg, stats: dict | None = None, train: bool = True):
        self.cfg = cfg
        self.train = train
        self.rows = [
            json.loads(line)
            for line in Path(index_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"No rows in {index_path}")
        stats_path = Path(index_path).parent / "stats.json"
        self.stats = stats or (
            json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        )
        self.speaker_map = self.stats.get("speaker_map", {"default": 0})
        self.emotion_map = self.stats.get("emotion_map", {"neutral": 0})
        self.pitch_mean = self.stats.get("pitch_mean", 0.0)
        self.pitch_std = self.stats.get("pitch_std", 1.0)
        self.energy_mean = self.stats.get("energy_mean", 0.0)
        self.energy_std = self.stats.get("energy_std", 1.0)

        d = cfg.data
        self.max_frames = d.get("max_frames", 1200)
        self.min_frames = d.get("min_frames", 32)
        self.max_tokens = d.get("max_tokens", 400)
        self.use_prior = d.get("use_attn_prior", True)
        self.prior_scale = d.get("attn_prior_scaling", 1.0)
        if train:
            before = len(self.rows)
            self.rows = [
                r for r in self.rows
                if self.min_frames <= r["n_frames"] <= self.max_frames
                and r["n_tokens"] <= self.max_tokens
            ]
            self.filtered = before - len(self.rows)
        else:
            self.filtered = 0
        self.lengths = [r["n_frames"] for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        mel = torch.from_numpy(np.load(row["mel"])).float()          # (n_mels, T)
        pitch = torch.from_numpy(np.load(row["pitch"])).float()
        energy = torch.from_numpy(np.load(row["energy"])).float()
        T = mel.shape[-1]
        pitch = _fit(pitch, T)
        energy = _fit(energy, T)
        pitch = (pitch - self.pitch_mean) / self.pitch_std
        energy = (energy - self.energy_mean) / self.energy_std

        tokens = torch.tensor(row["tokens"], dtype=torch.long)
        item = {
            "uid": row["uid"],
            "tokens": tokens,
            "token_strings": row.get("token_strings", []),
            "word_ids": row.get("word_ids", []),
            "words": row.get("words", []),
            "text": row.get("clean_text", row["text"]),
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "speaker": self.speaker_map.get(row.get("speaker", "default"), 0),
            "emotion": self.emotion_map.get(row.get("emotion", "neutral"), 0),
            "c_star": float(row.get("c_star", float("nan"))),
        }
        if self.use_prior:
            item["attn_prior"] = beta_binomial_prior(len(tokens), T, self.prior_scale)
        return item


def _fit(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.numel() == n:
        return x
    if x.numel() == 0:
        return torch.zeros(n)
    return torch.nn.functional.interpolate(
        x.view(1, 1, -1), size=n, mode="linear", align_corners=False
    ).view(-1)


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    max_tok = max(b["tokens"].numel() for b in batch)
    max_mel = max(b["mel"].shape[-1] for b in batch)
    n_mels = batch[0]["mel"].shape[0]

    tokens = torch.zeros(B, max_tok, dtype=torch.long)
    mel = torch.zeros(B, n_mels, max_mel)
    pitch = torch.zeros(B, max_mel)
    energy = torch.zeros(B, max_mel)
    token_lens = torch.zeros(B, dtype=torch.long)
    mel_lens = torch.zeros(B, dtype=torch.long)
    has_prior = "attn_prior" in batch[0]
    attn_prior = torch.zeros(B, max_mel, max_tok) if has_prior else None

    for i, b in enumerate(batch):
        t, f = b["tokens"].numel(), b["mel"].shape[-1]
        tokens[i, :t] = b["tokens"]
        mel[i, :, :f] = b["mel"]
        pitch[i, :f] = b["pitch"]
        energy[i, :f] = b["energy"]
        token_lens[i], mel_lens[i] = t, f
        if has_prior:
            attn_prior[i, :f, :t] = b["attn_prior"]

    out = {
        "uid": [b["uid"] for b in batch],
        "text": [b["text"] for b in batch],
        "token_strings": [b["token_strings"] for b in batch],
        "word_ids": [b["word_ids"] for b in batch],
        "words": [b["words"] for b in batch],
        "tokens": tokens,
        "token_lens": token_lens,
        "mel": mel,
        "mel_lens": mel_lens,
        "pitch": pitch,
        "energy": energy,
        "speakers": torch.tensor([b["speaker"] for b in batch], dtype=torch.long),
        "emotions": torch.tensor([b["emotion"] for b in batch], dtype=torch.long),
        "c_star": torch.tensor([b.get("c_star", float("nan")) for b in batch],
                               dtype=torch.float32),
    }
    if has_prior:
        out["attn_prior"] = attn_prior
    return out


class LengthBucketSampler(Sampler):
    """Groups similar-length utterances to cut padding waste (and therefore
    make the FLOP measurements less padding-dominated)."""

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True,
                 bucket_multiplier: int = 20, drop_last: bool = False, seed: int = 0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = batch_size * bucket_multiplier
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for i in range(0, len(idx), self.bucket_size):
            chunk = sorted(idx[i: i + self.bucket_size], key=lambda j: self.lengths[j])
            for k in range(0, len(chunk), self.batch_size):
                b = chunk[k: k + self.batch_size]
                if len(b) == self.batch_size or not self.drop_last:
                    batches.append(b)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        n = len(self.lengths)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


def build_dataloader(cfg, index_path: str, train: bool = True, batch_size: int | None = None,
                     num_workers: int | None = None) -> DataLoader:
    ds = TTSDataset(index_path, cfg, train=train)
    bs = batch_size or (cfg.train.batch_size if train else cfg.train.get("eval_batch_size", 8))
    nw = cfg.train.get("num_workers", 2) if num_workers is None else num_workers
    if train and cfg.data.get("bucket_by_length", True):
        sampler = LengthBucketSampler(ds.lengths, bs, shuffle=True, seed=cfg.get("seed", 1234))
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=nw,
                          pin_memory=cfg.train.get("pin_memory", False),
                          persistent_workers=nw > 0)
    return DataLoader(ds, batch_size=bs, shuffle=train, collate_fn=collate, num_workers=nw,
                      pin_memory=cfg.train.get("pin_memory", False),
                      persistent_workers=nw > 0)
