"""Figure rendering must not depend on anything that can go missing.

These exist because monitoring broke twice in a row on Colab: first a
plotly/kaleido version mismatch, then kaleido 1.x needing a Chrome install that
Colab does not ship. Both failed silently -- no IMAGES tab, so no alignment
plot and no complexity heatmaps, for a whole training run.
"""

import numpy as np
import pytest

from mint_tts.utils import plotting

N = 24
TOKENS = [f"t{i}" for i in range(N)]
VALUES = np.linspace(0.1, 1.0, N)


def _all_figures(mod):
    return [
        mod.plot_token_complexity(TOKENS, VALUES, "title", max_steps=8,
                                  words=["a", "b"], word_ids=[0] * N),
        mod.plot_word_complexity(["the", "record"], np.array([0.2, 0.9]),
                                 highlight=["record"]),
        mod.plot_alignment(np.random.rand(40, N), tokens=TOKENS),
        mod.plot_mel(np.random.randn(80, 40) - 5),
        mod.plot_mel(np.random.randn(80, 40) - 5, target=np.random.randn(80, 40) - 5),
        mod.plot_halting_matrix(np.random.rand(6, N), TOKENS),
        mod.plot_frame_complexity(np.random.rand(50)),
        mod.plot_depth_histogram(np.random.randint(1, 9, 100).astype(float), 8),
        mod.plot_compute_curve([0.2, 0.6, 1.0], [0.3, 0.8, 0.9], 0.85, 0.6),
        mod.plot_scatter([1, 2, 3], [0.2, 0.5, 0.9], "x", "y"),
        mod.plot_group_comparison({"easy": 0.3, "homograph": 0.8}),
    ]


def test_default_backend_needs_no_browser():
    """matplotlib is the default precisely so this can never regress."""
    assert plotting.backend() == "matplotlib"


def test_every_figure_rasterises_without_a_browser():
    figs = _all_figures(plotting)
    try:
        arrays = plotting.to_image_arrays(figs)
        assert len(arrays) == len(figs)
        for fig, arr in zip(figs, arrays):
            assert arr is not None, "a figure failed to rasterise"
            assert arr.ndim == 3 and arr.shape[0] == 3, f"expected (3,H,W), got {arr.shape}"
            assert arr.dtype == np.uint8
    finally:
        for f in figs:
            plotting.close(f)


def test_figures_are_saved_to_disk(tmp_path):
    fig = plotting.plot_token_complexity(TOKENS, VALUES, "t", max_steps=8)
    written = plotting.save_figure(fig, tmp_path / "fig")
    plotting.close(fig)
    assert written and all(p.exists() and p.stat().st_size > 0 for p in written)
    assert any(p.suffix == ".png" for p in written)


def test_dotted_filenames_are_not_truncated(tmp_path):
    """'000_q0.9_token_complexity' must not lose everything after the dot."""
    fig = plotting.plot_frame_complexity(np.random.rand(20))
    written = plotting.save_figure(fig, tmp_path / "000_q0.9_token_complexity")
    plotting.close(fig)
    assert any("q0.9_token_complexity" in p.name for p in written), written


def test_backend_switching_round_trips():
    original = plotting.backend()
    try:
        assert plotting.set_backend("matplotlib") == "matplotlib"
        # plotly may be absent; set_backend must degrade rather than raise
        name = plotting.set_backend("plotly")
        assert name in {"plotly", "matplotlib"}
    finally:
        plotting.set_backend(original)


def test_unknown_backend_raises():
    original = plotting.backend()
    try:
        with pytest.raises(ValueError):
            plotting.set_backend("ascii-art")
    finally:
        plotting.set_backend(original)


def test_plotly_backend_still_builds_figures_if_installed():
    """The interactive path must stay usable even though it is not the default."""
    pytest.importorskip("plotly")
    original = plotting.backend()
    try:
        if plotting.set_backend("plotly") != "plotly":
            pytest.skip("plotly backend unavailable")
        figs = _all_figures(plotting)
        assert all(f.__class__.__module__.startswith("plotly") for f in figs)
    finally:
        plotting.set_backend(original)
