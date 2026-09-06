"""Text -> token ids, with word provenance.

The tokenizer returns not just ids but also the *token strings* and a
*word map*. Those extra fields are what make the complexity heatmaps readable:
we can ask "how much compute went into the second `record`?" instead of "how
much went into index 27".

Symbol tables are built from the corpus during preprocessing and saved to
`symbols.json`, so IPA (whose inventory depends on the language and on
espeak's version) works without anyone hand-maintaining a phone list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .normalizer import TextNormalizer
from .phonemes import WORD_RE, build_phonemizer
from .symbols import BOS, EOS, PAD, PUNCTUATION, SPACE, SPECIALS, UNK

_TOKEN_RE = WORD_RE


@dataclass
class Encoded:
    ids: list[int]
    tokens: list[str]        # symbol strings, aligned with `ids`
    word_ids: list[int]      # index into `words` for every token
    words: list[str]         # normalised words, in order
    text: str                # normalised text
    raw_text: str

    def __len__(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SymbolTable:
    symbols: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._index = {s: i for i, s in enumerate(self.symbols)}

    @classmethod
    def build(cls, extra: list[str] | None = None) -> "SymbolTable":
        base = list(SPECIALS) + list(PUNCTUATION)
        for sym in extra or []:
            if sym not in base:
                base.append(sym)
        return cls(base)

    @classmethod
    def load(cls, path: str | Path) -> "SymbolTable":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.symbols, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def add(self, symbols) -> int:
        """Add unseen symbols; returns how many were new."""
        added = 0
        for sym in symbols:
            if sym not in self._index:
                self._index[sym] = len(self.symbols)
                self.symbols.append(sym)
                added += 1
        return added

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self._index[UNK]
        return [self._index.get(t, unk) for t in tokens]

    def decode(self, ids) -> list[str]:
        return [self.symbols[int(i)] if 0 <= int(i) < len(self.symbols) else UNK for i in ids]

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def pad_id(self) -> int:
        return self._index[PAD]


class TextProcessor:
    """Normalise -> phonemise -> tokenise, keeping the word map throughout."""

    def __init__(
        self,
        input_type: str = "ipa",
        phonemizer_kwargs: dict | None = None,
        normalizer: TextNormalizer | None = None,
        add_bos_eos: bool = True,
        add_word_boundary: bool = True,
        add_punctuation: bool = True,
        symbols: SymbolTable | list[str] | str | Path | None = None,
        allow_growth: bool = True,
    ):
        self.input_type = input_type
        self.normalizer = normalizer or TextNormalizer()
        self.phonemizer = build_phonemizer(input_type, **(phonemizer_kwargs or {}))
        # build_phonemizer may have degraded to characters; keep them in sync
        self.input_type = self.phonemizer.name
        self.add_bos_eos = add_bos_eos
        self.add_word_boundary = add_word_boundary
        self.add_punctuation = add_punctuation
        self.allow_growth = allow_growth

        if isinstance(symbols, SymbolTable):
            self.symbol_table = symbols
        elif isinstance(symbols, (str, Path)):
            self.symbol_table = SymbolTable.load(symbols)
        elif isinstance(symbols, list):
            self.symbol_table = SymbolTable(symbols)
        else:
            self.symbol_table = SymbolTable.build(
                list("abcdefghijklmnopqrstuvwxyz'") if self.input_type == "char" else None
            )

    # -- properties -------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        return self.symbol_table.symbols

    @property
    def vocab_size(self) -> int:
        return len(self.symbol_table)

    @property
    def pad_id(self) -> int:
        return self.symbol_table.pad_id

    # -- main API ---------------------------------------------------------
    def encode(self, text: str) -> Encoded:
        normalised = self.normalizer(text)
        pieces = self._split(normalised)
        words = [p for kind, p in pieces if kind == "word"]
        phones_per_word = self.phonemizer(words, normalised) if words else []

        tokens: list[str] = []
        word_ids: list[int] = []

        def push(sym: str, widx: int) -> None:
            tokens.append(sym)
            word_ids.append(widx)

        if self.add_bos_eos:
            push(BOS, 0)

        wi = 0
        seen_word = False
        for kind, piece in pieces:
            if kind == "word":
                if seen_word and self.add_word_boundary:
                    push(SPACE, max(wi - 1, 0))
                for phone in (phones_per_word[wi] or list(piece)):
                    push(phone, wi)
                seen_word = True
                wi += 1
            elif self.add_punctuation:
                push(piece, max(wi - 1, 0))

        if self.add_bos_eos:
            push(EOS, max(len(words) - 1, 0))

        if self.allow_growth:
            self.symbol_table.add(tokens)
        return Encoded(
            ids=self.symbol_table.encode(tokens),
            tokens=tokens,
            word_ids=word_ids,
            words=words,
            text=normalised,
            raw_text=text,
        )

    def decode(self, ids) -> list[str]:
        return self.symbol_table.decode(ids)

    def freeze(self) -> None:
        """Stop growing the vocabulary (call once training starts)."""
        self.allow_growth = False

    def save_symbols(self, path: str | Path) -> Path:
        return self.symbol_table.save(path)

    # -- internals --------------------------------------------------------
    @staticmethod
    def _split(text: str) -> list[tuple[str, str]]:
        """Split normalised text into ('word', w) / ('punct', p) pieces."""
        out, pos = [], 0
        for m in _TOKEN_RE.finditer(text):
            for ch in text[pos:m.start()]:
                if ch in PUNCTUATION:
                    out.append(("punct", ch))
            out.append(("word", m.group(0)))
            pos = m.end()
        for ch in text[pos:]:
            if ch in PUNCTUATION:
                out.append(("punct", ch))
        return out


def build_text_processor(cfg, symbols=None) -> TextProcessor:
    """Construct a TextProcessor from a config, preferring a saved symbol table."""
    t = cfg.text
    if symbols is None:
        candidate = t.get("symbols_file", None)
        if candidate and Path(candidate).exists():
            symbols = candidate
    kwargs = dict(t.get("phonemizer", {}) or {})
    normalizer = TextNormalizer(
        lowercase=t.get("lowercase", True),
        keep_punctuation=t.get("keep_punctuation", "!'(),-.:;?\""),
        skip=tuple(t.get("skip_normalisation_steps", [])),
    )
    return TextProcessor(
        input_type=t.get("input_type", "ipa"),
        phonemizer_kwargs=kwargs,
        normalizer=normalizer,
        add_bos_eos=t.get("add_bos_eos", True),
        add_word_boundary=t.get("add_word_boundary", True),
        add_punctuation=t.get("add_punctuation", True),
        symbols=symbols,
        allow_growth=t.get("allow_vocab_growth", True),
    )
