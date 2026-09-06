"""Lightweight config system: YAML -> nested attribute-access dicts, with
inheritance (`_base_`), CLI dotted overrides and provenance tracking.

We deliberately avoid heavy dependencies (hydra/omegaconf) so that a Colab
`pip install -r requirements.txt` stays fast and reproducible.
"""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class Config(dict):
    """A dict with attribute access that recursively wraps nested dicts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = _wrap(value)

    # -- attribute access -------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - error path
            raise AttributeError(
                f"Unknown config key '{item}'. Available: {sorted(self.keys())}"
            ) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def __delattr__(self, key: str) -> None:
        del self[key]

    # -- helpers ----------------------------------------------------------
    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Mapping):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = _wrap(value)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=_plain))

    def dump(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(self.to_dict()))


def _plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialise {type(obj)}")


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def merge(base: Mapping, override: Mapping) -> Config:
    """Deep-merge `override` into `base` (override wins)."""
    out = Config(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = _wrap(value)
    return out


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | os.PathLike, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config, resolving `_base_` chains and applying overrides.

    `_base_` may be a string or list of strings, relative to the child file.
    Overrides are `dotted.key=value` strings; values are parsed as Python
    literals when possible (so `model.d_model=512`, `train.fp16=true`,
    `log.test_sentences=['a','b']` all work).
    """
    path = Path(path)
    cfg = _resolve(path, seen=set())
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not of the form key=value")
        key, raw = item.split("=", 1)
        cfg.set_path(key.strip(), parse_value(raw.strip()))
    cfg.setdefault("_config_path", str(path))
    return cfg


def _resolve(path: Path, seen: set) -> Config:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular _base_ reference at {path}")
    seen.add(path)
    raw = _load_yaml(path)
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged = Config()
    for base in bases:
        merged = merge(merged, _resolve((path.parent / base), seen))
    return merge(merged, raw)


def parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
