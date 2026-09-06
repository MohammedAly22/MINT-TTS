"""Plotly figures for TensorBoard, Weights & Biases and standalone files.

The complexity heatmaps are not decoration -- they are the primary evidence
for the hypothesis. If MINT-TTS is doing what it claims, "record" in
*"the record is broken by the record broker"* should light up while "the"
stays dark, and that has to be legible at a glance.

Plotly is used because the interesting figures are dense and token-labelled:
hovering a cell to read `token='ɹ'  depth=6.0/8  word='record'` is far more
useful than squinting at a 60-tick axis. Every figure is saved as an
interactive `.html` and, when `kaleido` is installed, as a `.png` too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# shared styling
# --------------------------------------------------------------------------
FONT = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
MEL_SCALE = "Magma"
COMPLEXITY_SCALE = "Inferno"
ALIGN_SCALE = "Viridis"
HALT_SCALE = "Cividis"
ACCENT = "#3b82f6"
ACCENT_WARM = "#ef4444"
ACCENT_GREEN = "#10b981"
GRID = "rgba(148,163,184,0.25)"


def _layout(fig: go.Figure, title: str = "", height: int = 320, width: int | None = None,
            **kwargs) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, family=FONT), x=0.01, xanchor="left"),
        template="plotly_white",
        font=dict(family=FONT, size=11, color="#334155"),
        margin=dict(l=60, r=20, t=44 if title else 18, b=52),
        height=height,
        width=width,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel=dict(font_family=FONT, font_size=11),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _colorbar(title: str, **kwargs) -> dict:
    return dict(title=dict(text=title, side="right", font=dict(size=10)),
                thickness=12, len=0.9, outlinewidth=0, tickfont=dict(size=9), **kwargs)


def _auto_width(n: int, per_item: float = 22.0, lo: int = 620, hi: int = 1800) -> int:
    return int(min(max(lo, per_item * n + 160), hi))


# --------------------------------------------------------------------------
# spectrograms and alignment
# --------------------------------------------------------------------------
def plot_mel(mel: np.ndarray, title: str = "mel spectrogram",
             target: np.ndarray | None = None) -> go.Figure:
    """mel: (n_mels, T). With `target`, stacks prediction above ground truth."""
    from plotly.subplots import make_subplots

    if target is None:
        fig = go.Figure(go.Heatmap(z=mel, colorscale=MEL_SCALE, colorbar=_colorbar("log-mel"),
                                   hovertemplate="frame %{x}<br>mel bin %{y}<br>%{z:.2f}<extra></extra>"))
        return _layout(fig, title, height=300).update_xaxes(title="frame").update_yaxes(title="mel bin")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("predicted", "target"))
    zmin = float(min(mel.min(), target.min()))
    zmax = float(max(mel.max(), target.max()))
    for row, z in ((1, mel), (2, target)):
        fig.add_trace(go.Heatmap(z=z, colorscale=MEL_SCALE, zmin=zmin, zmax=zmax,
                                 showscale=(row == 1), colorbar=_colorbar("log-mel"),
                                 hovertemplate="frame %{x}<br>bin %{y}<br>%{z:.2f}<extra></extra>"),
                      row=row, col=1)
    fig.update_annotations(font=dict(size=11, family=FONT))
    _layout(fig, title, height=520)
    fig.update_xaxes(title="frame", row=2, col=1)
    return fig


def plot_alignment(attn: np.ndarray, title: str = "alignment",
                   tokens: list[str] | None = None) -> go.Figure:
    """attn: (T_mel, T_text) -- plotted as text on y, frames on x."""
    z = attn.T
    hover = "frame %{x}<br>token %{y}<br>weight %{z:.3f}<extra></extra>"
    ticktext = None
    if tokens is not None and len(tokens) == z.shape[0] and len(tokens) <= 90:
        ticktext = tokens
        hover = "frame %{x}<br>token %{customdata}<br>weight %{z:.3f}<extra></extra>"
    fig = go.Figure(go.Heatmap(
        z=z, colorscale=ALIGN_SCALE, colorbar=_colorbar("weight"),
        customdata=np.array(tokens)[:, None].repeat(z.shape[1], 1) if ticktext else None,
        hovertemplate=hover,
    ))
    _layout(fig, title, height=max(320, min(760, 12 * z.shape[0] + 120)))
    fig.update_xaxes(title="mel frame")
    if ticktext:
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(ticktext))),
                         ticktext=ticktext, tickfont=dict(size=8), title="")
    else:
        fig.update_yaxes(title="token index")
    return fig


# --------------------------------------------------------------------------
# the complexity figures
# --------------------------------------------------------------------------
def plot_token_complexity(tokens: list[str], complexity: np.ndarray, title: str = "",
                          max_steps: int | None = None,
                          words: list[str] | None = None,
                          word_ids: list[int] | None = None) -> go.Figure:
    """One-row heatmap of per-token compute, labelled with the real tokens."""
    n = min(len(tokens), len(complexity))
    tokens, c = list(tokens[:n]), np.asarray(complexity[:n], dtype=float)
    steps = c * max_steps if max_steps else c
    owner = ["" for _ in range(n)]
    if words is not None and word_ids is not None:
        owner = [words[w] if 0 <= w < len(words) else "" for w in word_ids[:n]]

    custom = np.stack([np.array(tokens, dtype=object), np.array(owner, dtype=object),
                       steps], axis=-1)[None, :, :]
    fig = go.Figure(go.Heatmap(
        z=c[None, :], zmin=0.0, zmax=1.0, colorscale=COMPLEXITY_SCALE,
        colorbar=_colorbar("compute"), customdata=custom, xgap=1,
        hovertemplate=("token <b>%{customdata[0]}</b><br>word <b>%{customdata[1]}</b>"
                       "<br>depth %{customdata[2]:.2f}"
                       + (f" / {max_steps}" if max_steps else "")
                       + "<br>compute %{z:.3f}<extra></extra>"),
    ))
    subtitle = title
    if max_steps:
        subtitle += f"   ·   mean depth {steps.mean():.2f} / {max_steps}"
    _layout(fig, subtitle, height=190, width=_auto_width(n))
    fig.update_xaxes(tickmode="array", tickvals=list(range(n)), ticktext=tokens,
                     tickfont=dict(size=9), tickangle=-90, title="")
    fig.update_yaxes(showticklabels=False, title="")
    return fig


def plot_word_complexity(words: list[str], values: np.ndarray,
                         title: str = "compute per word",
                         highlight: list[str] | None = None) -> go.Figure:
    """Bar chart of per-word compute; `highlight` words get an outline."""
    values = np.asarray(values, dtype=float)
    n = min(len(words), len(values))
    words, values = list(words[:n]), values[:n]
    hi = {w.lower() for w in (highlight or [])}
    line_w = [2.5 if w.lower() in hi else 0 for w in words]

    fig = go.Figure(go.Bar(
        x=list(range(n)), y=values, marker=dict(
            color=values, colorscale=COMPLEXITY_SCALE, cmin=0.0, cmax=1.0,
            colorbar=_colorbar("compute"),
            line=dict(color=ACCENT, width=line_w)),
        customdata=np.array(words, dtype=object)[:, None],
        hovertemplate="<b>%{customdata[0]}</b><br>compute %{y:.3f}<extra></extra>",
    ))
    _layout(fig, title, height=300, width=_auto_width(n, per_item=46, lo=560))
    fig.update_xaxes(tickmode="array", tickvals=list(range(n)), ticktext=words,
                     tickangle=-40, tickfont=dict(size=10), title="")
    fig.update_yaxes(title="normalised compute", range=[0, 1.05])
    return fig


def plot_halting_matrix(halting: np.ndarray, tokens: list[str] | None = None,
                        title: str = "halting probability per step") -> go.Figure:
    """halting: (n_steps, T) -- shows *when* each position stopped, not just how deep."""
    n_steps, T = halting.shape
    hover = "step %{y}<br>position %{x}<br>p(halt) %{z:.3f}<extra></extra>"
    custom = None
    if tokens is not None and len(tokens) >= T:
        custom = np.array(tokens[:T], dtype=object)[None, :].repeat(n_steps, 0)[:, :, None]
        hover = ("step %{y}<br>token <b>%{customdata[0]}</b>"
                 "<br>p(halt) %{z:.3f}<extra></extra>")
    fig = go.Figure(go.Heatmap(z=halting, zmin=0, zmax=1, colorscale=HALT_SCALE,
                               colorbar=_colorbar("p(halt)"), customdata=custom,
                               hovertemplate=hover, xgap=1, ygap=1))
    _layout(fig, title, height=max(200, 34 * n_steps + 110), width=_auto_width(T))
    fig.update_yaxes(title="step", tickmode="array", tickvals=list(range(n_steps)),
                     ticktext=[str(i + 1) for i in range(n_steps)])
    if tokens is not None and T <= 90:
        fig.update_xaxes(tickmode="array", tickvals=list(range(T)), ticktext=tokens[:T],
                         tickangle=-90, tickfont=dict(size=9), title="")
    else:
        fig.update_xaxes(title="position")
    return fig


def plot_frame_complexity(values: np.ndarray,
                          title: str = "acoustic compute per frame") -> go.Figure:
    values = np.asarray(values, dtype=float)
    fig = go.Figure(go.Scatter(
        x=list(range(len(values))), y=values, mode="lines", line=dict(color=ACCENT, width=1.4),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.18)",
        hovertemplate="frame %{x}<br>compute %{y:.3f}<extra></extra>"))
    _layout(fig, title, height=240)
    fig.update_xaxes(title="mel frame")
    fig.update_yaxes(title="normalised compute", range=[0, 1.05])
    return fig


def plot_depth_histogram(depths: np.ndarray, max_steps: int,
                         title: str = "executed depth per position") -> go.Figure:
    depths = np.asarray(depths, dtype=float)
    fig = go.Figure(go.Histogram(
        x=depths, xbins=dict(start=0.5, end=max_steps + 0.5, size=1.0),
        marker=dict(color=ACCENT, line=dict(color="white", width=1)),
        hovertemplate="%{x} steps<br>%{y} positions<extra></extra>"))
    _layout(fig, f"{title}   ·   mean {depths.mean():.2f} / {max_steps}", height=280)
    fig.update_xaxes(title="executed steps", dtick=1)
    fig.update_yaxes(title="positions")
    return fig


def plot_compute_curve(compute, quality, threshold: float | None = None,
                       c_star: float | None = None,
                       title: str = "compute-quality curve") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=list(compute), y=list(quality), mode="lines+markers",
        line=dict(color=ACCENT_WARM, width=2), marker=dict(size=8),
        name="Q(C)", hovertemplate="compute %{x:.3f}<br>Q %{y:.3f}<extra></extra>"))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color="#94a3b8", dash="dash", width=1),
                      annotation_text=f"target Q = {threshold:.2f}",
                      annotation_font=dict(size=10))
    if c_star is not None:
        fig.add_vline(x=c_star, line=dict(color=ACCENT, dash="dot", width=1.6),
                      annotation_text=f"C* = {c_star:.2f}", annotation_font=dict(size=10))
    _layout(fig, title, height=340, width=620)
    fig.update_xaxes(title="normalised compute")
    fig.update_yaxes(title="quality Q")
    return fig


def plot_scatter(x, y, xlabel: str, ylabel: str, title: str = "",
                 labels: list[str] | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=list(x), y=list(y), mode="markers",
        marker=dict(size=9, color=ACCENT_GREEN, opacity=0.75,
                    line=dict(color="white", width=1)),
        customdata=np.array(labels, dtype=object)[:, None] if labels else None,
        hovertemplate=(("<b>%{customdata[0]}</b><br>" if labels else "")
                       + f"{xlabel} %{{x}}<br>{ylabel} %{{y:.3f}}<extra></extra>")))
    _layout(fig, title, height=340, width=620)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return fig


def plot_group_comparison(groups: dict[str, float], title: str = "compute by sentence group",
                          ylabel: str = "normalised compute") -> go.Figure:
    names, vals = list(groups.keys()), [groups[k] for k in groups]
    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker=dict(color=vals, colorscale=COMPLEXITY_SCALE, cmin=0, cmax=1,
                    colorbar=_colorbar("compute")),
        hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>"))
    _layout(fig, title, height=300, width=max(520, 110 * len(names) + 160))
    fig.update_yaxes(title=ylabel, range=[0, 1.05])
    return fig


# --------------------------------------------------------------------------
# export helpers
# --------------------------------------------------------------------------
def to_png_bytes(fig: go.Figure, scale: float = 2.0) -> bytes | None:
    """Static PNG via kaleido; None when kaleido is unavailable."""
    try:
        return fig.to_image(format="png", scale=scale)
    except Exception:
        return None


def to_image_array(fig: go.Figure, scale: float = 2.0) -> np.ndarray | None:
    """(3, H, W) uint8 array for TensorBoard's add_image."""
    png = to_png_bytes(fig, scale)
    return _png_to_array(png) if png is not None else None


def to_image_arrays(figs: list[go.Figure], scale: float = 1.6) -> list[np.ndarray | None]:
    """Rasterise many figures in ONE browser session.

    kaleido pays a large fixed cost per call (~2.7 s here) but only ~0.6 s per
    figure when they are batched, so the monitors buffer their figures and
    flush them through this function rather than exporting one at a time.
    """
    if not figs:
        return []
    import tempfile

    try:
        import plotly.io as pio

        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"f{i}.png" for i in range(len(figs))]
            pio.write_images(figs, [str(p) for p in paths], scale=scale)
            return [_png_to_array(p.read_bytes()) if p.exists() else None for p in paths]
    except Exception:
        return [to_image_array(f, scale) for f in figs]


def _png_to_array(png: bytes) -> np.ndarray | None:
    try:
        import io

        from PIL import Image

        arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        return np.transpose(arr, (2, 0, 1)).copy()
    except Exception:
        return None


def save_figure(fig: go.Figure, path: str | Path, png: bool = True,
                html: bool = True) -> list[Path]:
    """Write a figure as interactive HTML and (if possible) a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NB: not `with_suffix("")` -- a name like "000_q0.9_token_complexity"
    # would lose everything after the first dot.
    name = path.name
    for ext in (".html", ".png"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    stem = path.parent / name
    written = []
    if html:
        target = Path(str(stem) + ".html")
        fig.write_html(str(target), include_plotlyjs="cdn", full_html=True)
        written.append(target)
    if png:
        data = to_png_bytes(fig)
        if data is not None:
            target = Path(str(stem) + ".png")
            target.write_bytes(data)
            written.append(target)
    return written


def close(fig) -> None:  # noqa: ARG001 - kept for call-site compatibility
    """No-op: plotly figures hold no OS resources that need releasing."""
    return None
