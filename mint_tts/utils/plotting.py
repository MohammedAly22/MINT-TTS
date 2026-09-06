"""Figure API with two interchangeable renderers.

``matplotlib`` (default)
    Draws straight into TensorBoard through `add_figure`. No external binary,
    no version handshake, ~10x faster per figure. This is the default because
    monitoring must not depend on anything that can go missing.

``plotly``
    Interactive: hover a heatmap cell and read
    ``token='ɹ' word='record' depth=6.0/8``. Great in W&B and in saved HTML.
    Rasterising it for TensorBoard needs kaleido, and kaleido 1.x drives a real
    Chrome install which a stock Colab runtime does not have -- so it is opt-in.

Select with ``log.figure_backend`` or `set_backend()`. Both renderers expose
identical function signatures, so call sites never branch.
"""

from __future__ import annotations

from pathlib import Path

from . import _mpl_figures as _mpl

_BACKENDS = {"matplotlib": _mpl}
_BACKEND_NAME = "matplotlib"
_BACKEND = _mpl

PLOT_FUNCTIONS = [
    "plot_mel", "plot_alignment", "plot_token_complexity", "plot_word_complexity",
    "plot_halting_matrix", "plot_frame_complexity", "plot_depth_histogram",
    "plot_compute_curve", "plot_scatter", "plot_group_comparison",
]


def _load_plotly():
    if "plotly" not in _BACKENDS:
        from . import _plotly_figures

        _BACKENDS["plotly"] = _plotly_figures
    return _BACKENDS["plotly"]


def set_backend(name: str) -> str:
    """Choose the renderer. Falls back to matplotlib if plotly is unusable."""
    global _BACKEND, _BACKEND_NAME
    name = (name or "matplotlib").lower()
    if name == "plotly":
        try:
            _BACKEND = _load_plotly()
            _BACKEND_NAME = "plotly"
            return _BACKEND_NAME
        except Exception:
            name = "matplotlib"
    if name not in _BACKENDS:
        raise ValueError(f"Unknown figure backend '{name}'. Options: matplotlib, plotly")
    _BACKEND = _BACKENDS[name]
    _BACKEND_NAME = name
    return _BACKEND_NAME


def backend() -> str:
    return _BACKEND_NAME


def _dispatch(fn_name: str):
    def wrapper(*args, **kwargs):
        return getattr(_BACKEND, fn_name)(*args, **kwargs)
    wrapper.__name__ = fn_name
    return wrapper


for _name in PLOT_FUNCTIONS:
    globals()[_name] = _dispatch(_name)


# -- export helpers ---------------------------------------------------------
def _is_plotly(fig) -> bool:
    return fig.__class__.__module__.startswith("plotly")


def to_image_array(fig, scale: float = 1.6):
    """(3, H, W) uint8 for TensorBoard, whichever renderer produced `fig`."""
    if _is_plotly(fig):
        return _load_plotly().to_image_array(fig, scale)
    return _mpl.to_image_array(fig, scale)


def to_image_arrays(figs: list, scale: float = 1.6) -> list:
    """Rasterise many figures.

    Plotly figures are batched into one kaleido session (a single figure costs
    ~2.7 s, but ~0.6 s each when batched). Matplotlib needs no batching.
    """
    if not figs:
        return []
    if all(_is_plotly(f) for f in figs):
        return _load_plotly().to_image_arrays(figs, scale)
    return [to_image_array(f, scale) for f in figs]


def save_figure(fig, path, png: bool = True, html: bool = True) -> list[Path]:
    """Write a figure to disk; plotly also writes interactive HTML."""
    if _is_plotly(fig):
        return _load_plotly().save_figure(fig, path, png=png, html=html)
    return _mpl.save_figure(fig, path, png=png, html=False)


def close(fig) -> None:
    if _is_plotly(fig):
        return None
    return _mpl.close(fig)


__all__ = PLOT_FUNCTIONS + [
    "set_backend", "backend", "save_figure", "to_image_array", "to_image_arrays", "close",
]
