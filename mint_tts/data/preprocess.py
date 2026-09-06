"""Offline feature extraction.

Produces, per utterance: a log-mel `.npy`, frame-level pitch and energy
`.npy`, and a JSONL index containing the cached token sequence (so the G2P
runs once, not once per epoch) together with word provenance used by the
complexity heatmaps.

No human annotation is required anywhere: durations are learned by the
internal aligner at training time, and the *compute* labels come later from
`scripts/compute_curve.py`, which is generated supervision, not hand labels.
"""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..text.normalizer import TextNormalizer
from ..text.tokenizer import SymbolTable, TextProcessor
from .audio import (
    AudioConfig,
    compute_energy,
    compute_pitch,
    load_wav,
    mel_spectrogram,
    trim_silence,
)
from .manifest import Record, build_label_maps, read_manifest, write_jsonl


@dataclass
class PreprocessPaths:
    out_dir: Path

    @property
    def mel(self) -> Path:
        return self.out_dir / "mel"

    @property
    def pitch(self) -> Path:
        return self.out_dir / "pitch"

    @property
    def energy(self) -> Path:
        return self.out_dir / "energy"

    def mkdirs(self) -> None:
        for p in (self.mel, self.pitch, self.energy):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=4)
def _text_processor(spec: tuple) -> TextProcessor:
    """Built lazily *inside* each worker.

    Neither an espeak backend nor a G2P object pickles reliably across
    processes, so each worker constructs its own from a plain tuple spec.
    Workers return token *strings*; the parent assigns the ids, which is what
    keeps the vocabulary identical no matter how many workers ran.
    """
    (input_type, lowercase, keep_punct, bos_eos, word_boundary, punctuation,
     skip_steps, phonemizer_items) = spec
    return TextProcessor(
        input_type=input_type,
        phonemizer_kwargs=dict(phonemizer_items),
        normalizer=TextNormalizer(lowercase=lowercase, keep_punctuation=keep_punct,
                                  skip=tuple(skip_steps)),
        add_bos_eos=bos_eos,
        add_word_boundary=word_boundary,
        add_punctuation=punctuation,
        allow_growth=True,
    )


def _spec_from_cfg(cfg) -> tuple:
    t = cfg.text
    return (
        t.get("input_type", "ipa"),
        bool(t.get("lowercase", True)),
        t.get("keep_punctuation", "!'(),-.:;?\""),
        bool(t.get("add_bos_eos", True)),
        bool(t.get("add_word_boundary", True)),
        bool(t.get("add_punctuation", True)),
        tuple(t.get("skip_normalisation_steps", [])),
        tuple(sorted(dict(t.get("phonemizer", {}) or {}).items())),
    )


def process_one(
    rec: Record,
    root: Path,
    paths: PreprocessPaths,
    ac: AudioConfig,
    tp_spec: tuple,
    compute_pitch_feature: bool = True,
) -> dict | None:
    tp = _text_processor(tp_spec)
    wav_path = (root / rec.audio) if not Path(rec.audio).is_absolute() else Path(rec.audio)
    if not wav_path.exists():
        return {"uid": rec.uid, "error": f"missing audio {wav_path}"}
    try:
        wav = load_wav(wav_path, ac)
        if ac.trim_silence:
            wav = trim_silence(wav, ac)
        if wav.numel() < ac.win_length * 2:
            return {"uid": rec.uid, "error": "audio too short"}
        mel = mel_spectrogram(wav, ac)                     # (n_mels, T)
        n_frames = mel.shape[-1]
        energy = compute_energy(mel)
        pitch = (
            compute_pitch(wav, ac, n_frames)
            if compute_pitch_feature
            else torch.zeros(n_frames)
        )
        enc = tp.encode(rec.text)
        if len(enc.ids) < 2:
            return {"uid": rec.uid, "error": "empty token sequence"}

        np.save(paths.mel / f"{rec.uid}.npy", mel.numpy().astype(np.float32))
        np.save(paths.pitch / f"{rec.uid}.npy", pitch.numpy().astype(np.float32))
        np.save(paths.energy / f"{rec.uid}.npy", energy.numpy().astype(np.float32))

        return {
            "uid": rec.uid,
            "audio": str(wav_path),
            "text": rec.text,
            "clean_text": enc.text,
            "tokens": enc.ids,          # provisional; the parent re-assigns them
            "token_strings": enc.tokens,
            "word_ids": enc.word_ids,
            "words": enc.words,
            "speaker": rec.speaker,
            "emotion": rec.emotion,
            "lang": rec.lang,
            "n_frames": int(n_frames),
            "n_tokens": len(enc.ids),
            "duration_sec": float(wav.numel() / ac.sample_rate),
            "mel": str((paths.mel / f"{rec.uid}.npy").as_posix()),
            "pitch": str((paths.pitch / f"{rec.uid}.npy").as_posix()),
            "energy": str((paths.energy / f"{rec.uid}.npy").as_posix()),
        }
    except Exception as exc:  # pragma: no cover - per-file robustness
        return {"uid": rec.uid, "error": f"{exc}\n{traceback.format_exc(limit=2)}"}


def normalise_stats(rows: list[dict], key: str) -> tuple[float, float]:
    vals = []
    for row in rows:
        arr = np.load(row[key])
        arr = arr[arr > 0] if key == "pitch" else arr
        if arr.size:
            vals.append(arr)
    if not vals:
        return 0.0, 1.0
    cat = np.concatenate(vals)
    return float(cat.mean()), float(cat.std() + 1e-8)


def run_preprocess(cfg, manifest_path: str, out_dir: str, split_name: str = "train",
                   n_workers: int = 1, limit: int | None = None) -> dict:
    root = Path(cfg.data.root)
    paths = PreprocessPaths(Path(out_dir))
    paths.mkdirs()
    ac = AudioConfig.from_cfg(cfg.audio)
    tp_spec = _spec_from_cfg(cfg)
    records = read_manifest(manifest_path, list(cfg.data.get("columns", ["audio", "text"])),
                            cfg.data.get("delimiter", "|"))
    if limit:
        records = records[:limit]

    rows, errors = [], []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(process_one, r, root, paths, ac, tp_spec,
                            cfg.audio.get("extract_pitch", True))
                for r in records
            ]
            for fut in tqdm(futures, desc=f"preprocess[{split_name}]"):
                res = fut.result()
                (errors if res is None or "error" in res else rows).append(res)
    else:
        for r in tqdm(records, desc=f"preprocess[{split_name}]"):
            res = process_one(r, root, paths, ac, tp_spec, cfg.audio.get("extract_pitch", True))
            (errors if res is None or "error" in res else rows).append(res)

    # -- vocabulary ------------------------------------------------------
    # Workers only produced token *strings*. The parent owns the symbol table
    # so the vocabulary is identical no matter how many workers ran, and so
    # val/test can never introduce symbols the model was not trained on.
    symbols_path = Path(out_dir) / "symbols.json"
    if split_name == "train":
        table = SymbolTable.build()
        for row in rows:
            table.add(row["token_strings"])
        table.save(symbols_path)
    elif symbols_path.exists():
        table = SymbolTable.load(symbols_path)
    else:
        raise FileNotFoundError(
            f"{symbols_path} not found. Preprocess the train split first: it "
            "defines the vocabulary that every other split must reuse."
        )
    unknown = 0
    for row in rows:
        row["tokens"] = table.encode(row["token_strings"])
        unknown += sum(1 for t in row["token_strings"] if t not in table.symbols)

    speaker_map, emotion_map = build_label_maps(records)
    out_index = Path(out_dir) / f"{split_name}.jsonl"
    write_jsonl(out_index, rows)

    stats = {}
    if split_name == "train" and rows:
        pm, ps = normalise_stats(rows, "pitch")
        em, es = normalise_stats(rows, "energy")
        stats = {
            "pitch_mean": pm, "pitch_std": ps,
            "energy_mean": em, "energy_std": es,
            "n_utterances": len(rows),
            "total_hours": sum(r["duration_sec"] for r in rows) / 3600.0,
            "speaker_map": speaker_map,
            "emotion_map": emotion_map,
            "n_speakers": len(speaker_map),
            "n_emotions": len(emotion_map),
            "symbols": table.symbols,
            "vocab_size": len(table),
            "input_type": tp_spec[0],
            "symbols_file": str(symbols_path.as_posix()),
        }
        (Path(out_dir) / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if errors:
        (Path(out_dir) / f"{split_name}_errors.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8"
        )
    return {"n_ok": len(rows), "n_error": len(errors), "index": str(out_index),
            "vocab_size": len(table), "unknown_symbols": unknown, "stats": stats}
