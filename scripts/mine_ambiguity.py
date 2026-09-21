"""Discover which words are ambiguous, from the corpus itself.

    # text-only, needs nothing but the preprocessed corpus (bootstrap)
    python scripts/mine_ambiguity.py --config configs/egyptian_homograph.yaml

    # acoustic + context, once a checkpoint can align (much stronger)
    python scripts/mine_ambiguity.py --config configs/egyptian_homograph.yaml \
        --checkpoint runs/egyptian_homograph/checkpoints/best.pt

Writes `<preprocessed_dir>/ambiguity.json`: one entry per word type with its
measured ambiguity score, plus the evidence behind it.

Why this exists
---------------
There is no list of Egyptian homographs anywhere in this repository, and there
should not be. A hand-written list covers a few dozen words out of 62k, encodes
the author's guesses rather than the corpus, and has to be rewritten for every
new dialect or language. The same information is *in the data*: a homograph is
a spelling whose pronunciation varies with context, and both halves of that are
measurable.

Two modes
---------
**Text-only** (no `--checkpoint`). Uses the cached language-model vectors
alone: a word whose contextual embeddings spread out across the corpus is one
whose meaning shifts with context. Weaker evidence -- a spread of *meaning*
does not always imply a spread of *pronunciation* -- but it needs no alignment,
so it can run before the first training step.

**Acoustic** (with `--checkpoint`). Uses the aligner to find each word's mel
frames, clusters the pronunciations, and asks whether context predicts the
cluster. This is the real measurement, and it requires a checkpoint whose
alignment is already healthy (`align/entropy_ratio` well below 0.3).

The intended workflow is therefore: bootstrap text-only, train until alignment
is solid, re-mine acoustically, continue training with the better scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from mint_tts.config import load_config
from mint_tts.text.ambiguity import (
    AmbiguityMiner,
    AmbiguityTable,
    ContextualAmbiguity,
)
from mint_tts.utils.logging_utils import get_logger


def load_rows(index_path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in index_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return rows[:limit] if limit else rows


def word_frame_spans(word_ids: list[int], durations: np.ndarray, n_words: int):
    """Mel-frame span of each word, from per-token durations."""
    starts = np.cumsum(durations) - durations
    spans: list[tuple[int, int]] = []
    for w in range(n_words):
        idx = [i for i, ww in enumerate(word_ids) if ww == w and i < len(durations)]
        if not idx:
            spans.append((0, 0))
            continue
        spans.append((int(starts[idx[0]]), int(starts[idx[-1]] + durations[idx[-1]])))
    return spans


def acoustic_embedding(mel: np.ndarray, start: int, end: int, n_slices: int = 4) -> np.ndarray | None:
    """A fixed-size descriptor of how one word was pronounced.

    The span is split into `n_slices` equal parts and each is averaged, so the
    descriptor keeps the word's *shape over time* -- which is what separates
    `3alam` from `3elm`, since they differ in vowel quality at specific
    positions, not in their overall average. A single mean over the whole word
    would wash exactly that out.
    """
    if end - start < n_slices:
        return None
    seg = mel[:, start:end]
    parts = np.array_split(seg, n_slices, axis=1)
    return np.concatenate([p.mean(1) for p in parts]).astype(np.float32)


def mine_text_only(rows: list[dict], log, min_count: int) -> dict:
    """Bootstrap: LM vectors only, no audio."""
    miner = ContextualAmbiguity(min_count=min_count)
    missing = 0
    for row in rows:
        path = row.get("semantic")
        if not path or not Path(path).exists():
            missing += 1
            continue
        vecs = np.load(path)
        for w, word in enumerate(row.get("words", [])):
            if w < len(vecs):
                miner.add(word, vecs[w])
    if missing:
        log.warning(f"{missing} utterances had no cached semantic features; "
                    "re-run preprocessing with model.semantic.enabled=true")
    return miner.finalise()


class _MiningRows(torch.utils.data.Dataset):
    """Loads what one utterance needs for acoustic mining (in worker processes)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        sem_path = row.get("semantic")
        if not sem_path or not Path(sem_path).exists():
            return None
        try:
            return {"row": row, "mel": np.load(row["mel"]), "context": np.load(sem_path)}
        except Exception:
            return None


def _collate_mining(items):
    items = [it for it in items if it is not None]
    if not items:
        return None
    B = len(items)
    max_tok = max(len(it["row"]["tokens"]) for it in items)
    max_mel = max(it["mel"].shape[-1] for it in items)
    n_mels = items[0]["mel"].shape[0]
    tokens = torch.zeros(B, max_tok, dtype=torch.long)
    mels = torch.zeros(B, n_mels, max_mel)
    tlen = torch.zeros(B, dtype=torch.long)
    mlen = torch.zeros(B, dtype=torch.long)
    for i, it in enumerate(items):
        t, f = len(it["row"]["tokens"]), it["mel"].shape[-1]
        tokens[i, :t] = torch.tensor(it["row"]["tokens"], dtype=torch.long)
        mels[i, :, :f] = torch.from_numpy(it["mel"]).float()
        tlen[i], mlen[i] = t, f
    return {"items": items, "tokens": tokens, "mels": mels, "tlen": tlen, "mlen": mlen}


def mine_acoustic(rows: list[dict], cfg, checkpoint: str, log,
                  min_count: int, device: str, batch_size: int = 32,
                  num_workers: int = 4) -> tuple[dict, float]:
    """The real measurement: acoustic clusters predicted by context.

    Returns (stats, mean alignment entropy ratio). The entropy ratio is the
    same `align/entropy_ratio` the trainer logs, and says whether the
    alignment behind these word spans can be trusted at all.

    Batched: utterances are length-sorted and aligned `batch_size` at a time
    on the GPU, with the same beta-binomial prior the aligner was trained with
    (the aligner learned a *posterior* on top of it; dropping the prior here
    would mine with an alignment the model never produced).
    """
    import math

    from mint_tts.models.tts import build_model
    from mint_tts.modules.aligner import (
        beta_binomial_prior_batch,
        monotonic_alignment_search,
        path_to_durations,
    )
    from mint_tts.modules.transformer import lengths_to_mask
    from mint_tts.text.tokenizer import build_text_processor

    dev = torch.device(device if device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    symbols = Path(checkpoint).parent.parent / "symbols.json"
    if not symbols.exists():
        symbols = Path(cfg.data.preprocessed_dir) / "symbols.json"
    tp = build_text_processor(cfg, symbols=symbols)
    tp.freeze()
    model = build_model(cfg, tp.vocab_size)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(dev).eval()
    log.info(f"  checkpoint step {ckpt.get('step', '?')}, device {dev}, batch {batch_size}")

    use_prior = bool(cfg.data.get("use_attn_prior", True))
    prior_scale = float(cfg.data.get("attn_prior_scaling", 1.0))

    # Length-sorted batches: near-zero padding, so a batch costs what its
    # utterances cost rather than what its longest one does.
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("n_frames", 0))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    loader = torch.utils.data.DataLoader(
        _MiningRows(rows), batch_sampler=batches, collate_fn=_collate_mining,
        num_workers=num_workers, pin_memory=dev.type == "cuda",
        prefetch_factor=4 if num_workers > 0 else None,
    )

    miner = AmbiguityMiner(min_count=min_count)
    done, entropies = 0, []
    report_every = max(len(rows) // 10, 1)
    next_report = report_every

    for batch in loader:
        if batch is None:
            continue
        items = batch["items"]
        tokens = batch["tokens"].to(dev, non_blocking=True)
        mels = batch["mels"].to(dev, non_blocking=True)
        tlen = batch["tlen"].to(dev)
        mlen = batch["mlen"].to(dev)
        with torch.inference_mode():
            # Keys come from the token EMBEDDING, exactly as in training, so
            # these durations are the alignment the model actually learned.
            emb = model.embedding(tokens) * model.emb_scale
            text_mask = lengths_to_mask(tlen, tokens.size(1))
            prior = (beta_binomial_prior_batch(tlen, mlen, tokens.size(1), mels.size(-1),
                                               prior_scale) if use_prior else None)
            logp, _ = model.aligner(emb, mels, text_mask, prior)
            logp = logp.squeeze(1).float()
            hard = monotonic_alignment_search(logp, tlen, mlen)
            dur = path_to_durations(hard).cpu().numpy().astype(int)

            # entropy ratio per utterance, matching the trainer's diagnostic
            p = logp.exp()
            ent = -(p * logp.clamp_min(-1e4)).sum(-1)                  # (B, T_mel)
            fmask = lengths_to_mask(mlen, mels.size(-1)).float()
            ent = (ent * fmask).sum(1) / fmask.sum(1).clamp_min(1)
            norm = torch.log(tlen.float().clamp_min(2))
            entropies.extend((ent / norm).cpu().tolist())

        for i, it in enumerate(items):
            row, mel, context = it["row"], it["mel"], it["context"]
            words = row.get("words", [])
            d = dur[i, :len(row["tokens"])]
            spans = word_frame_spans(row.get("word_ids", []), d, len(words))
            text = row.get("clean_text", "")
            for w, word in enumerate(words):
                if w >= len(context):
                    continue
                start, end = spans[w]
                ac = acoustic_embedding(mel, start, end)
                if ac is None:
                    continue
                miner.add(word, ac, context[w], text)
        done += len(items)
        if done >= next_report:
            log.info(f"  {done}/{len(rows)}")
            next_report += report_every

    skipped = len(rows) - done
    if skipped:
        log.warning(f"{skipped} utterances skipped (missing semantic features)")
    mean_entropy = float(np.mean(entropies)) if entropies else float("nan")
    if not math.isfinite(mean_entropy):
        mean_entropy = float("nan")
    return miner.finalise(), mean_entropy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--checkpoint", default=None,
                    help="enables acoustic mining; without it the run is text-only")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-count", type=int, default=6,
                    help="word types rarer than this are not scored")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None,
                    help="defaults to <preprocessed_dir>/ambiguity.json")
    ap.add_argument("--top", type=int, default=40, help="how many to print")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-entropy", type=float, default=0.3,
                    help="acoustic mode: only promote the result to the active "
                         "ambiguity.json when the mean alignment entropy ratio is "
                         "at or below this (the aligner is locating words)")
    ap.add_argument("--force", action="store_true",
                    help="promote the acoustic table even if alignment looks unhealthy")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    log = get_logger("mine_ambiguity")
    pre = Path(cfg.data.preprocessed_dir)
    index = pre / f"{args.split}.jsonl"
    if not index.exists():
        log.error(f"{index} not found. Run scripts/preprocess.py first.")
        return 1

    rows = load_rows(index, args.limit)
    log.info(f"{len(rows)} utterances from {index}")

    entropy = float("nan")
    if args.checkpoint:
        log.info(f"acoustic mining with {args.checkpoint}")
        stats, entropy = mine_acoustic(rows, cfg, args.checkpoint, log, args.min_count,
                                       args.device, args.batch_size, args.workers)
        mode = "acoustic"
        log.info(f"alignment entropy ratio over the corpus: {entropy:.3f} "
                 f"(trust threshold {args.max_entropy})")
    else:
        log.info("text-only mining (no --checkpoint): language-model vectors only. "
                 "Re-run with a checkpoint once alignment is healthy.")
        stats = mine_text_only(rows, log, args.min_count)
        mode = "text_only"

    if not stats:
        log.error("No word types had enough occurrences to score. "
                  "Lower --min-count, or check that preprocessing wrote "
                  "semantic features.")
        return 1

    table = AmbiguityTable.from_stats(stats)
    # Each mode keeps its own file, so an acoustic pass can never destroy the
    # text-only bootstrap. `ambiguity.json` is the ACTIVE table the trainer
    # reads; an acoustic result only replaces it once alignment is healthy,
    # because word spans from a half-trained aligner are noise.
    mode_file = pre / f"ambiguity.{mode}.json"
    table.save(mode_file, stats)
    healthy = mode != "acoustic" or (entropy == entropy and entropy <= args.max_entropy)
    if args.out:
        out = Path(args.out)
        table.save(out, stats)
    elif healthy or args.force:
        out = pre / "ambiguity.json"
        table.save(out, stats)
    else:
        out = mode_file
        log.warning(
            "NOT promoting the acoustic table: the alignment entropy ratio is "
            f"{entropy:.3f} > {args.max_entropy}, so the aligner is not yet "
            "locating words and these pronunciation clusters are mostly noise. "
            "The active ambiguity.json is unchanged. Re-run once "
            "align/entropy_ratio has fallen (or pass --force)."
        )

    scores = np.array([s.ambiguity for s in stats.values()])
    log.info(f"scored {len(stats)} word types ({mode})")
    log.info(f"  ambiguity > 0.5 : {int((scores > 0.5).sum())}")
    log.info(f"  ambiguity > 0.2 : {int((scores > 0.2).sum())}")
    log.info(f"  median          : {float(np.median(scores)):.4f}")
    log.info(f"written to {out}" + ("" if out == mode_file else f" (and {mode_file.name})"))

    print(f"\nTop {args.top} most ambiguous words discovered:")
    print(f"{'word':<18}{'count':>7}{'sep':>8}{'predict':>9}{'ambig':>8}")
    for word, _ in table.top(args.top):
        s = stats[word]
        print(f"{word:<18}{s.count:>7}{s.separation:>8.3f}"
              f"{s.context_predictivity:>9.3f}{s.ambiguity:>8.3f}")

    print("\nThese were discovered from the corpus, not from any word list.")
    print("Sanity-check a few by eye: they should be spellings whose reading")
    print("genuinely depends on context. If they look like noise, alignment")
    print("is probably not healthy enough yet for acoustic mining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
