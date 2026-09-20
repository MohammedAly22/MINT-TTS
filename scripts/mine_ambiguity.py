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


def mine_acoustic(rows: list[dict], cfg, checkpoint: str, log,
                  min_count: int, device: str) -> dict:
    """The real measurement: acoustic clusters predicted by context."""
    from mint_tts.models.tts import build_model
    from mint_tts.modules.aligner import monotonic_alignment_search, path_to_durations
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

    miner = AmbiguityMiner(min_count=min_count)
    skipped = 0

    for n, row in enumerate(rows):
        if n % 500 == 0:
            log.info(f"  {n}/{len(rows)}")
        sem_path = row.get("semantic")
        if not sem_path or not Path(sem_path).exists():
            skipped += 1
            continue
        try:
            mel = np.load(row["mel"])
            context = np.load(sem_path)
            tokens = torch.tensor(row["tokens"], dtype=torch.long, device=dev).unsqueeze(0)
            tlen = torch.tensor([len(row["tokens"])], device=dev)
            mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(dev)
            mlen = torch.tensor([mel.shape[-1]], device=dev)

            with torch.inference_mode():
                # The aligner keys off the token EMBEDDING, exactly as in
                # training, so the durations here are the ones the model
                # actually learned rather than a different alignment.
                emb = model.embedding(tokens) * model.emb_scale
                text_mask = torch.ones(1, tokens.size(1), dtype=torch.bool, device=dev)
                logp, _ = model.aligner(emb, mel_t, text_mask, None)
                hard = monotonic_alignment_search(logp.squeeze(1).float(), tlen, mlen)
                dur = path_to_durations(hard)[0].cpu().numpy().astype(int)
        except Exception as exc:
            skipped += 1
            if skipped < 5:
                log.warning(f"  {row['uid']}: {exc}")
            continue

        words = row.get("words", [])
        spans = word_frame_spans(row.get("word_ids", []), dur, len(words))
        text = row.get("clean_text", "")
        for w, word in enumerate(words):
            if w >= len(context):
                continue
            start, end = spans[w]
            ac = acoustic_embedding(mel, start, end)
            if ac is None:
                continue
            miner.add(word, ac, context[w], text)

    if skipped:
        log.warning(f"{skipped} utterances skipped (missing features or alignment failure)")
    return miner.finalise()


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

    if args.checkpoint:
        log.info(f"acoustic mining with {args.checkpoint}")
        stats = mine_acoustic(rows, cfg, args.checkpoint, log, args.min_count, args.device)
        mode = "acoustic"
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
    out = Path(args.out) if args.out else pre / "ambiguity.json"
    table.save(out, stats)

    scores = np.array([s.ambiguity for s in stats.values()])
    log.info(f"scored {len(stats)} word types ({mode})")
    log.info(f"  ambiguity > 0.5 : {int((scores > 0.5).sum())}")
    log.info(f"  ambiguity > 0.2 : {int((scores > 0.2).sum())}")
    log.info(f"  median          : {float(np.median(scores)):.4f}")
    log.info(f"written to {out}")

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
