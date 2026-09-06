"""Dataset manifest parsing.

Three interchangeable on-disk formats are supported so you can point the repo
at an existing corpus without rewriting it:

1. **Pipe filelist** (`.txt`) -- classic TTS style, column names come from
   `data.columns`::

       wavs/LJ001-0001.wav|printing, in the only sense with which we are here concerned

2. **CSV** (`.csv`) -- with a header row; column names are used directly::

       audio,text,speaker,emotion
       wavs/0001.wav,"Hello there.",p225,neutral

3. **JSONL** (`.jsonl`) -- one JSON object per line; this is also what the
   preprocessing stage writes out.

The canonical record fields are:

    audio     (required) path to the waveform, relative to `data.root`
    text      (required) raw transcript
    speaker   optional speaker id/name        -> multi-speaker (VCTK/LibriTTS)
    emotion   optional emotion/style label    -> expressive TTS
    lang      optional language tag
    style     optional free-form style prompt
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CANONICAL_FIELDS = ["audio", "text", "speaker", "emotion", "lang", "style"]


@dataclass
class Record:
    audio: str
    text: str
    speaker: str = "default"
    emotion: str = "neutral"
    lang: str = "en"
    style: str = ""
    uid: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(d.pop("extra") or {})
        return d


def _uid_from_audio(audio: str) -> str:
    return Path(audio).stem


def read_manifest(path: str | Path, columns: list[str] | None = None,
                  delimiter: str = "|") -> list[Record]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        columns = columns or ["audio", "text"]
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split(delimiter)
            if len(parts) < len(columns):
                parts = parts + [""] * (len(columns) - len(parts))
            rows.append(dict(zip(columns, parts[: len(columns)])))

    records = []
    for row in rows:
        row = {k: (v if v is not None else "") for k, v in row.items()}
        known = {k: row.pop(k) for k in list(row) if k in CANONICAL_FIELDS}
        uid = row.pop("uid", "") or _uid_from_audio(known.get("audio", ""))
        if not known.get("audio") or not known.get("text"):
            raise ValueError(f"Manifest row missing audio/text: {known} (file {path})")
        rec = Record(
            audio=known["audio"].strip(),
            text=known["text"].strip(),
            speaker=(known.get("speaker") or "default").strip(),
            emotion=(known.get("emotion") or "neutral").strip(),
            lang=(known.get("lang") or "en").strip(),
            style=(known.get("style") or "").strip(),
            uid=uid,
            extra={k: v for k, v in row.items() if k not in {"audio", "text"}},
        )
        records.append(rec)
    return records


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_label_maps(records: list[Record]) -> tuple[dict[str, int], dict[str, int]]:
    speakers = sorted({r.speaker for r in records})
    emotions = sorted({r.emotion for r in records})
    return {s: i for i, s in enumerate(speakers)}, {e: i for i, e in enumerate(emotions)}
