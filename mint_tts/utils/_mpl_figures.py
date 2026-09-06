"""Matplotlib renderer -- the default, because it needs no browser.

The plotly figures are nicer to explore (hover a heatmap cell and read
``token='ɹ' word='record' depth=6.0/8``), but rasterising them for TensorBoard
goes through kaleido, and kaleido 1.x drives a real Chrome install that is
simply absent on a stock Colab runtime. That failed silently and cost a whole
training run: no IMAGES tab, so no alignment plot and no complexity heatmaps.

Matplotlib draws straight into TensorBoard via `add_figure`, with no external
binary, no version handshake, and about 10x less time per figure. Monitoring
the model should never depend on a headless browser.

Every function here mirrors its counterpart in `_plotly_figures.py`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

MEL_CMAP = "magma"
COMPLEXITY_CMAP = "inferno"
ALIGN_CMAP = "viridis"
HALT_CMAP = "cividis"
ACCENT = "#3b82f6"
ACCENT_WARM = "#ef4444"
ACCENT_GREEN = "#10b981"

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "text.color": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "grid.color": "#e2e8f0",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _auto_width(n: int, per_item: float = 0.16, lo: float = 6.0, hi: float = 22.0) -> float:
    return float(min(max(lo, per_item * n + 1.6), hi))


def _tick_labels(ax, labels: list[str], max_ticks: int = 90) -> None:
    n = len(labels)
    if n == 0:
        return
    step = 1 if n <= max_ticks else int(np.ceil(n / max_ticks))
    idx = list(range(0, n, step))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=90,
                       fontsize=7 if n <= 60 else 5.5)


# --------------------------------------------------------------------------
# spectrograms and alignment
# --------------------------------------------------------------------------
def plot_mel(mel: np.ndarray, title: str = "mel spectrogram",
             target: np.ndarray | None = None):
    if target is None:
        fig, ax = plt.subplots(figsize=(10, 3))
        im = ax.imshow(mel, aspect="auto", origin="lower", interpolation="nearest", cmap=MEL_CMAP)
        ax.set_title(title)
        ax.set_xlabel("frame")
        ax.set_ylabel("mel bin")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        fig.tight_layout()
        return fig

    vmin = float(min(mel.min(), target.min()))
    vmax = float(max(mel.max(), target.max()))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    for ax, z, name in ((axes[0], mel, "predicted"), (axes[1], target, "target")):
        im = ax.imshow(z, aspect="auto", origin="lower", interpolation="nearest",
                       cmap=MEL_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(name, loc="left", fontsize=9)
        ax.set_ylabel("mel bin")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    axes[1].set_xlabel("frame")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_alignment(attn: np.ndarray, title: str = "alignment",
                   tokens: list[str] | None = None):
    z = attn.T                                     # (T_text, T_mel)
    fig, ax = plt.subplots(figsize=(10, max(3.0, min(8.0, 0.11 * z.shape[0] + 1.6))))
    im = ax.imshow(z, aspect="auto", origin="lower", interpolation="nearest", cmap=ALIGN_CMAP)
    ax.set_xlabel("mel frame")
    ax.set_title(title)
    if tokens is not None and len(tokens) >= z.shape[0] and z.shape[0] <= 80:
        ax.set_yticks(range(z.shape[0]))
        ax.set_yticklabels(tokens[: z.shape[0]], fontsize=6)
    else:
        ax.set_ylabel("token")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# complexity
# --------------------------------------------------------------------------
def plot_token_complexity(tokens: list[str], complexity: np.ndarray, title: str = "",
                          max_steps: int | None = None,
                          words: list[str] | None = None,
                          word_ids: list[int] | None = None):
    n = min(len(tokens), len(complexity))
    tokens = list(tokens[:n])
    c = np.asarray(complexity[:n], dtype=float)
    fig, ax = plt.subplots(figsize=(_auto_width(n), 1.9))
    im = ax.imshow(c[None, :], aspect="auto", cmap=COMPLEXITY_CMAP, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_yticks([])
    _tick_labels(ax, tokens)
    subtitle = title
    if max_steps:
        subtitle += f"   ·   mean depth {c.mean() * max_steps:.2f} / {max_steps}"
    ax.set_title(subtitle, loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.005, label="compute")
    fig.tight_layout()
    return fig


def plot_word_complexity(words: list[str], values: np.ndarray,
                         title: str = "compute per word",
                         highlight: list[str] | None = None):
    values = np.asarray(values, dtype=float)
    n = min(len(words), len(values))
    words, values = list(words[:n]), values[:n]
    hi = {w.lower() for w in (highlight or [])}
    cmap = plt.get_cmap(COMPLEXITY_CMAP)
    colors = cmap(np.clip(values, 0, 1))

    fig, ax = plt.subplots(figsize=(max(6.0, min(20.0, 0.42 * n + 2.0)), 3.0))
    bars = ax.bar(range(n), values, color=colors)
    for w, bar in zip(words, bars):
        if w.lower() in hi:
            bar.set_edgecolor(ACCENT)
            bar.set_linewidth(2.2)
    ax.set_xticks(range(n))
    ax.set_xticklabels(words, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("normalised compute")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_halting_matrix(halting: np.ndarray, tokens: list[str] | None = None,
                        title: str = "halting probability per step"):
    n_steps, T = halting.shape
    fig, ax = plt.subplots(figsize=(_auto_width(T), 0.34 * n_steps + 1.9))
    im = ax.imshow(halting, aspect="auto", origin="lower", vmin=0, vmax=1,
                   cmap=HALT_CMAP, interpolation="nearest")
    ax.set_ylabel("step")
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels([str(i + 1) for i in range(n_steps)])
    if tokens is not None and T <= 90:
        _tick_labels(ax, list(tokens[:T]))
    else:
        ax.set_xlabel("position")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.005, label="p(halt)")
    fig.tight_layout()
    return fig


def plot_frame_complexity(values: np.ndarray, title: str = "acoustic compute per frame"):
    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.plot(values, lw=1.2, color=ACCENT)
    ax.fill_between(range(len(values)), 0, values, alpha=0.22, color=ACCENT)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("mel frame")
    ax.set_ylabel("normalised compute")
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_depth_histogram(depths: np.ndarray, max_steps: int,
                         title: str = "executed depth per position"):
    depths = np.asarray(depths, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(depths, bins=np.arange(0.5, max_steps + 1.5, 1.0), color=ACCENT,
            edgecolor="white")
    ax.set_xlabel("executed steps")
    ax.set_ylabel("positions")
    ax.set_title(f"{title}   ·   mean {depths.mean():.2f} / {max_steps}", loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_compute_curve(compute, quality, threshold: float | None = None,
                       c_star: float | None = None,
                       title: str = "compute-quality curve"):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(list(compute), list(quality), "o-", color=ACCENT_WARM, lw=2, ms=6)
    if threshold is not None:
        ax.axhline(threshold, ls="--", c="#94a3b8", lw=1, label=f"target Q = {threshold:.2f}")
    if c_star is not None:
        ax.axvline(c_star, ls=":", c=ACCENT, lw=1.6, label=f"C* = {c_star:.2f}")
    ax.set_xlabel("normalised compute")
    ax.set_ylabel("quality Q")
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    if threshold is not None or c_star is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_scatter(x, y, xlabel: str, ylabel: str, title: str = "",
                 labels: list[str] | None = None):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.scatter(list(x), list(y), s=42, color=ACCENT_GREEN, alpha=0.8,
               edgecolors="white", linewidths=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_group_comparison(groups: dict, title: str = "compute by sentence group",
                          ylabel: str = "normalised compute"):
    names = list(groups.keys())
    vals = [float(groups[k]) for k in names]
    cmap = plt.get_cmap(COMPLEXITY_CMAP)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(names) + 2.0), 3.0))
    ax.bar(names, vals, color=cmap(np.clip(vals, 0, 1)))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def to_image_array(fig, scale: float = 1.0) -> np.ndarray | None:
    """(3, H, W) uint8 -- pure matplotlib, no browser involved."""
    try:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        return np.transpose(buf, (2, 0, 1)).copy()
    except Exception:
        return None


def save_figure(fig, path, png: bool = True, html: bool = False) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.name
    for ext in (".png", ".html"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    target = path.parent / (name + ".png")
    fig.savefig(target, bbox_inches="tight")
    return [target]


def close(fig) -> None:
    try:
        plt.close(fig)
    except Exception:
        pass
