"""Dataset / collate / samplers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..modules.aligner import beta_binomial_prior
from ..text.homographs_ar import difficulty_profile


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

        sem = cfg.model.get("semantic", {}) or {}
        self.use_semantic = bool(sem.get("enabled", False))
        self.semantic_dim = int(
            self.stats.get("semantic_hidden_size", sem.get("hidden_size", 768))
        )
        # Per-token difficulty prior. Computed here (cheaply, from the cached
        # word list) rather than in preprocessing, so the lexicon can be
        # edited without re-running feature extraction over 100 hours.
        self.use_difficulty = float(
            cfg.loss.get("compute", {}).get("difficulty_relief", 0.0)
        ) > 0
        ref = cfg.model.get("reference_encoder", {}) or {}
        self.use_reference = bool(ref.get("enabled", False))
        self.reference_frames = int(ref.get("reference_frames", 256))
        self.reference_same_speaker = bool(ref.get("same_speaker", True))
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

        # Index rows by speaker so a reference clip can be drawn from the same
        # voice. On a single-speaker corpus this is one bucket holding
        # everything, which is exactly what is wanted: the reference is always
        # a *different* utterance by that same speaker.
        self.by_speaker: dict[str, list[int]] = {}
        if self.use_reference:
            for i, r in enumerate(self.rows):
                self.by_speaker.setdefault(r.get("speaker", "default"), []).append(i)

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
            "audio_path": row.get("audio", ""),
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

        if self.use_semantic:
            item["semantic"] = self._load_semantic(row)
            # Every token maps to the row of the word it belongs to; tokens
            # with no word (padding) are clamped to 0 by the collate.
            item["word_index"] = torch.tensor(
                row.get("word_ids", []) or [0] * len(tokens), dtype=torch.long
            )
        if self.use_reference:
            item["reference_mel"] = self._sample_reference(i, row)
        if self.use_difficulty:
            item["difficulty"] = self._difficulty(row, len(tokens))
        return item

    def _difficulty(self, row: dict, n_tokens: int) -> torch.Tensor:
        """Per-token difficulty, broadcast from the per-word score."""
        words = row.get("words", [])
        word_ids = row.get("word_ids", [])
        if not words or not word_ids:
            return torch.zeros(n_tokens)
        per_word = difficulty_profile(words)
        out = torch.zeros(n_tokens)
        for t, w in enumerate(word_ids[:n_tokens]):
            if 0 <= w < len(per_word):
                out[t] = per_word[w]
        return out

    def _load_semantic(self, row: dict) -> torch.Tensor:
        """Per-word LM vectors for this utterance, (n_words, H)."""
        path = row.get("semantic")
        n_words = max(len(row.get("words", [])), 1)
        if not path or not Path(path).exists():
            # A missing cache file must not be silently treated as "neutral
            # semantics": it would train the adapter on zeros for that
            # utterance. Raising here is deliberate -- the preprocessing step
            # that writes these is not optional once semantics are enabled.
            raise FileNotFoundError(
                f"Semantic features missing for {row['uid']} ({path}). "
                "Re-run scripts/preprocess.py with model.semantic.enabled=true."
            )
        arr = np.load(path)
        if arr.shape[0] < n_words:   # truncated by the LM's max_length
            pad = np.zeros((n_words - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], 0)
        return torch.from_numpy(arr[:n_words]).float()

    def _sample_reference(self, i: int, row: dict) -> torch.Tensor:
        """A reference mel from the same speaker but a DIFFERENT utterance.

        Using the target utterance itself would hand the model the mel it is
        being asked to predict, so the reconstruction loss could be driven down
        without learning anything about voice.
        """
        pool = self.by_speaker.get(row.get("speaker", "default"), [i])
        j = i
        if len(pool) > 1:
            for _ in range(4):
                j = random.choice(pool)
                if j != i:
                    break
        mel = torch.from_numpy(np.load(self.rows[j]["mel"])).float()
        n = self.reference_frames
        if mel.shape[-1] > n:
            start = random.randint(0, mel.shape[-1] - n) if self.train else 0
            mel = mel[:, start:start + n]
        return mel


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

    has_semantic = "semantic" in batch[0]
    if has_semantic:
        max_words = max(b["semantic"].shape[0] for b in batch)
        sem_dim = batch[0]["semantic"].shape[1]
        semantic = torch.zeros(B, max_words, sem_dim)
        word_index = torch.zeros(B, max_tok, dtype=torch.long)
    has_difficulty = "difficulty" in batch[0]
    if has_difficulty:
        difficulty = torch.zeros(B, max_tok)
    has_reference = "reference_mel" in batch[0]
    if has_reference:
        max_ref = max(b["reference_mel"].shape[-1] for b in batch)
        reference_mel = torch.zeros(B, n_mels, max_ref)
        reference_lens = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        t, f = b["tokens"].numel(), b["mel"].shape[-1]
        tokens[i, :t] = b["tokens"]
        mel[i, :, :f] = b["mel"]
        pitch[i, :f] = b["pitch"]
        energy[i, :f] = b["energy"]
        token_lens[i], mel_lens[i] = t, f
        if has_prior:
            attn_prior[i, :f, :t] = b["attn_prior"]
        if has_semantic:
            w = b["semantic"].shape[0]
            semantic[i, :w] = b["semantic"]
            wi = b["word_index"][:t]
            # Clamp into this row's own word range: a stale word_id pointing
            # past the end would silently gather a different word's vector.
            word_index[i, :wi.numel()] = wi.clamp(0, max(w - 1, 0))
        if has_difficulty:
            dv = b["difficulty"][:t]
            difficulty[i, :dv.numel()] = dv
        if has_reference:
            rf = b["reference_mel"].shape[-1]
            reference_mel[i, :, :rf] = b["reference_mel"]
            reference_lens[i] = rf

    out = {
        "uid": [b["uid"] for b in batch],
        "audio_path": [b.get("audio_path", "") for b in batch],
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
    if has_semantic:
        out["semantic"] = semantic
        out["word_index"] = word_index
    if has_difficulty:
        out["difficulty"] = difficulty
    if has_reference:
        out["reference_mel"] = reference_mel
        out["reference_lens"] = reference_lens
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
