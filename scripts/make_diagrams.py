"""Generate the MINTS logo and the documentation diagrams as SVG.

    python scripts/make_diagrams.py

Writes to assets/:
    logo.svg            leaf mark + wordmark
    logo-mark.svg       leaf mark alone (favicon / avatar)
    architecture.svg    the full model, with the adaptive block expanded
    pipeline.svg        text -> audio in six steps
    experiments.svg     the experiment ladder

Everything is generated rather than hand-drawn so the diagrams stay consistent
with each other, and so a change to the model is a change to this file rather
than a fiddly edit of raw SVG.
"""

from __future__ import annotations

import math
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# -- palette ---------------------------------------------------------------
LEAF_LIGHT = "#8CC63F"
LEAF_DARK = "#2E9B57"
INK = "#1e293b"
MUTED = "#64748b"
FAINT = "#94a3b8"
LINE = "#cbd5e1"
WHITE = "#ffffff"

GREEN_BG, GREEN_BR = "#f0fdf4", "#86efac"
BLUE_BG, BLUE_BR = "#eff6ff", "#93c5fd"
AMBER_BG, AMBER_BR = "#fffbeb", "#fcd34d"
PURPLE_BG, PURPLE_BR = "#faf5ff", "#d8b4fe"
SLATE_BG, SLATE_BR = "#f8fafc", "#cbd5e1"

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, "
        "Helvetica, Arial, sans-serif")


# -- primitives ------------------------------------------------------------
def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=13, fill=INK, weight="400", anchor="middle",
         family=FONT, spacing=None, style=None):
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    extra += f' font-style="{style}"' if style else ""
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"'
            f'{extra}>{esc(s)}</text>')


def arrow(x1, y1, x2, y2, color=FAINT, sw=1.8, dash=None, marker="arrow"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def path(d, stroke=FAINT, sw=1.8, fill="none", dash=None, marker=None):
    a = f' stroke-dasharray="{dash}"' if dash else ""
    a += f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{a}/>'


def defs(extra: str = "") -> str:
    return f"""<defs>
  <linearGradient id="leafL" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#A8D65C"/><stop offset="100%" stop-color="{LEAF_LIGHT}"/>
  </linearGradient>
  <linearGradient id="leafR" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#3FAE63"/><stop offset="100%" stop-color="{LEAF_DARK}"/>
  </linearGradient>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 1 L 9 5 L 0 9 z" fill="{FAINT}"/>
  </marker>
  <marker id="arrowg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 1 L 9 5 L 0 9 z" fill="{LEAF_DARK}"/>
  </marker>
  <marker id="arrowb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 1 L 9 5 L 0 9 z" fill="#3b82f6"/>
  </marker>
{extra}</defs>"""


# -- the leaf --------------------------------------------------------------
def leaf_half(sign: int, n=300, W=44.0, H=132.0, k=1.45, q=0.78, teeth=14, amp=2.0):
    pts = []
    for i in range(n + 1):
        t = i / n
        w = W * (math.sin(math.pi * (t ** k)) ** q)
        env = math.sin(math.pi * t) ** 0.6
        saw = 2.0 * abs(((t * teeth) % 1.0) - 0.5)
        w += amp * env * (saw - 0.5) * 2.0
        pts.append((sign * max(w, 0.0), t * H))
    return pts


def leaf_svg(cx: float, cy: float, scale: float = 1.0) -> str:
    """Two-tone mint leaf, tip up, centred horizontally on cx, top at cy."""
    def fmt(pts):
        return " ".join(f"{cx + px * scale:.2f},{cy + py * scale:.2f}" for px, py in pts)

    left, right = leaf_half(-1), leaf_half(+1)
    spine_top, spine_bot = f"{cx:.2f},{cy:.2f}", f"{cx:.2f},{cy + 132 * scale:.2f}"
    return (
        f'<polygon points="{fmt(left)} {spine_bot} {spine_top}" fill="url(#leafL)"/>'
        f'<polygon points="{spine_top} {spine_bot} {fmt(right[::-1])}" fill="url(#leafR)"/>'
    )


def write_logo() -> None:
    w, h = 900, 260
    body = [f'<rect width="{w}" height="{h}" fill="none"/>',
            leaf_svg(268, 64, 1.0),
            text(560, 172, "MINTS", size=112, weight="600", spacing="18", fill="#111111")]
    (ASSETS / "logo.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" role="img" aria-label="MINTS">\n{defs()}\n' + "\n".join(body) + "\n</svg>\n",
        encoding="utf-8")

    m = 200
    (ASSETS / "logo-mark.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {m} {m}" width="{m}" '
        f'height="{m}" role="img" aria-label="MINTS mark">\n{defs()}\n'
        + leaf_svg(m / 2, 34, 1.0) + "\n</svg>\n", encoding="utf-8")


# -- shared chrome ---------------------------------------------------------
def header(w: float, title: str, subtitle: str) -> list[str]:
    return [
        f'<rect width="{w}" height="100%" fill="{WHITE}"/>',
        f'<g transform="translate(0,0) scale(0.30)">{leaf_svg(150, 46, 1.0)}</g>',
        text(78, 44, "MINTS", size=21, weight="600", spacing="3", anchor="start", fill="#111111"),
        text(78, 64, title, size=13, weight="500", anchor="start", fill=MUTED),
        text(w - 26, 44, subtitle, size=12, anchor="end", fill=FAINT),
    ]


def stage(x, y, w, h, title, lines, bg, br, badge=None, dash=None, title_size=13.5):
    out = [box(x, y, w, h, bg, br, dash=dash),
           text(x + w / 2, y + 25, title, size=title_size, weight="600")]
    for i, ln in enumerate(lines):
        out.append(text(x + w / 2, y + 46 + i * 16, ln, size=11, fill=MUTED))
    if badge:
        out.append(text(x + w / 2, y + h - 10, badge, size=9.5, weight="600", fill=LEAF_DARK))
    return out


# -- architecture ----------------------------------------------------------
def write_architecture() -> None:
    W, H = 1500, 1000
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" role="img" aria-label="MINTS architecture">', defs()]
    o += header(W, "Architecture", "adaptive depth per token and per frame")

    # ---- row A -----------------------------------------------------------
    ya, ha = 120, 96
    row_a = [
        (40, 178, "Text", ["“The record is broken.”"], SLATE_BG, SLATE_BR),
        (236, 196, "Normalise", ["numbers, dates, currency,", "emails, abbreviations"], SLATE_BG, SLATE_BR),
        (456, 186, "Tokenise", ["char / IPA / ARPAbet", "+ word map"], SLATE_BG, SLATE_BR),
        (666, 206, "Embedding + Prenet", ["conv local mixing", "+ positional encoding"], GREEN_BG, GREEN_BR),
    ]
    for x, w, t, ls, bg, br in row_a:
        o += stage(x, ya, w, ha, t, ls, bg, br)
    for a, b in [(212, 236), (432, 456), (642, 666)]:
        o.append(arrow(a, ya + ha / 2, b - 4, ya + ha / 2))

    lx, lw = 902, 300
    o.append(box(lx, ya - 8, lw, ha + 16, GREEN_BG, LEAF_DARK, rx=12, sw=2.2))
    o.append(text(lx + lw / 2, ya + 20, "LINGUISTIC STACK", size=14, weight="700", fill=LEAF_DARK))
    o.append(text(lx + lw / 2, ya + 41, "one shared block, re-applied", size=11, fill=MUTED))
    o.append(text(lx + lw / 2, ya + 60, "adaptive depth PER TOKEN", size=11.5, weight="600"))
    o.append(text(lx + lw / 2, ya + 80, "“record” 7 passes · “the” 2", size=10.5, fill=LEAF_DARK))
    o.append(arrow(872, ya + ha / 2, lx - 4, ya + ha / 2))

    # the budget is an input, so it gets a box of its own
    o.append(box(1252, ya + 6, 208, 76, WHITE, BLUE_BR, rx=10, dash="5 4"))
    o.append(text(1356, ya + 28, "budget (q, h)", size=12, weight="600", fill="#2563eb"))
    o.append(text(1356, ya + 46, "q = quality you ask for", size=10, fill=MUTED))
    o.append(text(1356, ya + 62, "h = device capability", size=10, fill=MUTED))
    o.append(arrow(1252, ya + 44, lx + lw + 4, ya + 44, color="#3b82f6", marker="arrowb", dash="4 3"))

    # ---- row B -----------------------------------------------------------
    yb, hb = 400, 96
    o += stage(40, yb, 244, hb, "Duration · Pitch · Energy",
               ["frames per token,", "how high, how loud"], AMBER_BG, AMBER_BR,
               title_size=12.5)
    o += stage(308, yb, 192, hb, "Length Regulator",
               ["token rate → frame rate", "+ positional encoding"], PURPLE_BG, PURPLE_BR)

    ax_, aw = 540, 300
    o.append(box(ax_, yb - 8, aw, hb + 16, BLUE_BG, "#2563eb", rx=12, sw=2.2))
    o.append(text(ax_ + aw / 2, yb + 20, "ACOUSTIC STACK", size=14, weight="700", fill="#2563eb"))
    o.append(text(ax_ + aw / 2, yb + 41, "one shared block, re-applied", size=11, fill=MUTED))
    o.append(text(ax_ + aw / 2, yb + 60, "adaptive depth PER FRAME", size=11.5, weight="600"))
    o.append(text(ax_ + aw / 2, yb + 80, "silence cheap · onsets costly", size=10.5, fill="#2563eb"))

    o += stage(880, yb, 180, hb, "Linear + Postnet",
               ["conv refinement,", "added as a residual"], SLATE_BG, SLATE_BR)
    o += stage(1100, yb, 148, hb, "Mel", ["80 bands", "× frames"], SLATE_BG, SLATE_BR)
    o.append(box(1288, yb, 172, hb, PURPLE_BG, PURPLE_BR))
    o.append(text(1374, yb + 25, "HiFi-GAN", size=13.5, weight="600"))
    o.append(text(1374, yb + 46, "neural vocoder", size=11, fill=MUTED))
    o.append(text(1374, yb + 68, "FROZEN", size=10, weight="700", fill=LEAF_DARK))
    o.append(text(1374, yb + 83, "shared across runs", size=9.5, fill=LEAF_DARK))
    for a, b in [(284, 308), (500, ax_), (840, 880), (1060, 1100), (1248, 1288)]:
        o.append(arrow(a, yb + hb / 2, b - 4, yb + hb / 2))

    o.append(path(f"M 1374 {yb + hb} L 1374 {yb + hb + 30}", marker="arrow"))
    o.append(text(1374, yb + hb + 52, "AUDIO", size=15, weight="700", fill=LEAF_DARK))

    # the aligner sits under the embedding, because that is what feeds it
    o.append(box(666, 250, 206, 62, WHITE, AMBER_BR, rx=10, dash="5 4"))
    o.append(text(769, 272, "Aligner · training only", size=11.5, weight="600", fill="#b45309"))
    o.append(text(769, 289, "forward-sum + MAS", size=10, fill=MUTED))
    o.append(text(769, 304, "reads the embedding, not the stack", size=8.5, fill=FAINT))
    o.append(arrow(769, ya + ha, 769, 246, color="#f59e0b", dash="5 4"))
    o.append(path(f"M 769 312 L 769 344 L 104 344 L 104 {yb - 4}",
                  stroke="#f59e0b", dash="5 4", marker="arrow"))
    o.append(text(430, 338, "token embedding — unrouted, so alignment and routing "
                  "cannot destabilise each other", size=9.5, fill="#b45309"))

    # linguistic stack -> variance predictors
    o.append(path(f"M {lx + lw / 2} {ya + ha + 8} L {lx + lw / 2} 370 L 210 370 L 210 {yb - 4}",
                  marker="arrow"))
    o.append(text(1010, 364, "token representations", size=10, fill=MUTED))

    # ---- detail panel ----------------------------------------------------
    py, ph = 560, 300
    o.append(box(40, py, 1420, ph, "#fcfdfe", LINE, rx=14))
    o.append(text(64, py + 28, "Inside a stack — Adaptive Computation Time",
                  size=14, weight="700", anchor="start"))
    o.append(text(64, py + 48, "The router reads the state ENTERING each step, then decides "
                  "whether that position takes another pass.", size=11.5, fill=MUTED, anchor="start"))

    bx, by, bh = 110, py + 78, 74
    o.append(box(bx, by, 150, bh, WHITE, LINE))
    o.append(text(bx + 75, by + 30, "state", size=12.5, weight="600"))
    o.append(text(bx + 75, by + 50, "one vector per position", size=10, fill=MUTED))

    rx_, rw = 330, 200
    o.append(box(rx_, by, rw, bh, BLUE_BG, BLUE_BR))
    o.append(text(rx_ + rw / 2, by + 27, "Router", size=12.5, weight="600"))
    o.append(text(rx_ + rw / 2, by + 45, "sees state, q, h", size=10.5, fill=MUTED))
    o.append(text(rx_ + rw / 2, by + 61, "→ p(halt)", size=10.5, fill="#2563eb"))

    dx, dy, dw, dh = 610, by - 4, 130, bh + 8
    o.append(f'<polygon points="{dx + dw / 2},{dy} {dx + dw},{dy + dh / 2} '
             f'{dx + dw / 2},{dy + dh} {dx},{dy + dh / 2}" fill="{AMBER_BG}" '
             f'stroke="{AMBER_BR}" stroke-width="1.6"/>')
    o.append(text(dx + dw / 2, dy + dh / 2 + 4, "halt?", size=12, weight="600"))

    sx, sw_ = 830, 300
    o.append(box(sx, by - 10, sw_, bh + 20, GREEN_BG, GREEN_BR))
    o.append(text(sx + sw_ / 2, by + 12, "Shared Block", size=12.5, weight="600"))
    o.append(box(sx + 22, by + 24, 120, 34, WHITE, GREEN_BR, rx=8, sw=1.2))
    o.append(text(sx + 82, by + 46, "Self-Attention", size=10.5))
    o.append(box(sx + 158, by + 24, 120, 34, WHITE, GREEN_BR, rx=8, sw=1.2))
    o.append(text(sx + 218, by + 46, "Feed-Forward", size=10.5))
    o.append(text(sx + sw_ / 2, by + 74, "residual, pre-norm · same weights every pass",
                  size=10, fill=MUTED))

    ox = 1210
    o.append(box(ox, by, 200, bh, PURPLE_BG, PURPLE_BR))
    o.append(text(ox + 100, by + 27, "weighted output", size=12.5, weight="600"))
    o.append(text(ox + 100, by + 45, "combination of the", size=10, fill=MUTED))
    o.append(text(ox + 100, by + 59, "intermediate states", size=10, fill=MUTED))

    o.append(arrow(260, by + bh / 2, rx_ - 4, by + bh / 2))
    o.append(arrow(rx_ + rw, by + bh / 2, dx - 4, by + bh / 2))
    o.append(arrow(dx + dw, by + bh / 2, sx - 4, by + bh / 2, color="#3b82f6", marker="arrowb"))
    o.append(text(785, by + bh / 2 - 8, "no", size=10.5, weight="600", fill="#2563eb"))
    o.append(path(f"M {dx + dw / 2} {dy} L {dx + dw / 2} {py + 62} L {ox + 100} {py + 62} "
                  f"L {ox + 100} {by - 4}", stroke=LEAF_DARK, marker="arrowg"))
    o.append(text(dx + dw / 2 + 46, py + 56, "yes", size=10.5, weight="600", fill=LEAF_DARK))
    o.append(path(f"M {sx + sw_ / 2} {by + bh + 10} L {sx + sw_ / 2} {by + bh + 46} "
                  f"L {bx + 75} {by + bh + 46} L {bx + 75} {by + bh}", marker="arrow"))
    o.append(text(660, by + bh + 62, "next pass — the block is applied again, "
                  "with the same weights", size=10.5, fill=MUTED))

    notes = [
        "Halted positions keep frozen keys/values, so attention still sees the whole "
        "sequence while queries and the feed-forward are computed only for live positions.",
        "Training uses a dense masked path; inference gathers only the live positions. "
        "The two are asserted numerically identical in tests/test_adaptive.py.",
    ]
    for i, n in enumerate(notes):
        o.append(text(64, py + ph - 34 + i * 17, "• " + n, size=10.5, fill=MUTED, anchor="start"))

    # ---- footer ----------------------------------------------------------
    fy = 900
    o.append(f'<line x1="40" y1="{fy - 14}" x2="{W - 40}" y2="{fy - 14}" stroke="{LINE}"/>')
    for i, (label, detail) in enumerate([
        ("Efficient", "compute spent where it is needed"),
        ("Adaptive", "depth chosen per token and per frame"),
        ("Controllable", "quality and hardware budgets at inference"),
    ]):
        cx = 300 + i * 420
        o.append(text(cx, fy + 12, label, size=13, weight="700", fill=LEAF_DARK))
        o.append(text(cx, fy + 32, detail, size=11, fill=MUTED))
    o.append("</svg>")
    (ASSETS / "architecture.svg").write_text("\n".join(o) + "\n", encoding="utf-8")


# -- pipeline --------------------------------------------------------------
def write_pipeline() -> None:
    W, H = 1120, 706
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" role="img" aria-label="text to audio pipeline">', defs()]
    o += header(W, "From text to audio", "what happens to one sentence")

    steps = [
        ("normalise", "“the record is broken.”", "numbers, dates and abbreviations become speakable words"),
        ("tokenise", "t  h  e  ␣  r  e  c  o  r  d  ␣ …", "characters here, so homographs stay ambiguous for the model to resolve"),
        ("encode", "one vector per character", "the linguistic stack, at a depth the router picks per token"),
        ("durations", "how many frames does each character last?", "learned by the aligner during training, predicted at inference"),
        ("decode", "a mel spectrogram", "80 frequency bands × one column per 11.6 ms frame"),
        ("vocoder", "a waveform", "HiFi-GAN, frozen and identical across every experiment"),
    ]
    y0, gap = 118, 82
    o.append(box(40, y0 - 34, W - 80, 44, SLATE_BG, SLATE_BR))
    o.append(text(W / 2, y0 - 6, "“The record is broken.”", size=14, weight="600"))
    for i, (name, mid, note) in enumerate(steps):
        y = y0 + 24 + i * gap
        o.append(arrow(W / 2, y - 22, W / 2, y - 4))
        o.append(box(40, y, W - 80, 58, WHITE, LINE))
        o.append(f'<circle cx="76" cy="{y + 29}" r="15" fill="{GREEN_BG}" stroke="{GREEN_BR}"/>')
        o.append(text(76, y + 34, str(i + 1), size=13, weight="700", fill=LEAF_DARK))
        o.append(text(104, y + 24, name, size=13, weight="600", anchor="start"))
        o.append(text(104, y + 44, note, size=10.5, fill=MUTED, anchor="start"))
        o.append(text(W - 60, y + 34, mid, size=11.5, fill=INK, anchor="end", style="italic"))
    y = y0 + 24 + len(steps) * gap
    o.append(arrow(W / 2, y - 22, W / 2, y - 4))
    o.append(box(40, y, W - 80, 44, GREEN_BG, GREEN_BR))
    o.append(text(W / 2, y + 28, "AUDIO", size=14, weight="700", fill=LEAF_DARK))
    o.append("</svg>")
    (ASSETS / "pipeline.svg").write_text("\n".join(o) + "\n", encoding="utf-8")


# -- experiments -----------------------------------------------------------
def write_experiments() -> None:
    W, H = 1180, 748
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" role="img" aria-label="experiment ladder">', defs()]
    o += header(W, "Experiment ladder", "each rung is a stop/go decision, not a checklist")

    rungs = [
        ("0", "Dense baselines", "deep and shallow", "the quality ceiling and the cost floor", SLATE_BG, SLATE_BR),
        ("1", "Sentence-level adaptive", "one decision per utterance", "is there any headroom at all?", BLUE_BG, BLUE_BR),
        ("2", "Token-level adaptive", "one decision per token", "the headline claim", GREEN_BG, LEAF_DARK),
        ("3", "Shared vs independent", "matched parameters", "is depth free in parameters?", BLUE_BG, BLUE_BR),
        ("4", "Acoustic routing", "one decision per frame", "where 75% of the compute actually is", BLUE_BG, BLUE_BR),
        ("5", "Unified + hardware budget", "linguistic and acoustic", "the full C*(x, q, h)", PURPLE_BG, PURPLE_BR),
    ]
    y0, gap, h = 116, 86, 62
    for i, (num, title, sub, question, bg, br) in enumerate(rungs):
        y = y0 + i * gap
        o.append(box(60, y, 700, h, bg, br, sw=2.2 if num == "2" else 1.6))
        o.append(f'<circle cx="98" cy="{y + h / 2}" r="18" fill="{WHITE}" stroke="{br}" stroke-width="1.8"/>')
        o.append(text(98, y + h / 2 + 5, num, size=15, weight="700", fill=INK))
        o.append(text(130, y + 26, title, size=13.5, weight="700" if num == "2" else "600", anchor="start"))
        o.append(text(130, y + 45, sub, size=11, fill=MUTED, anchor="start"))
        o.append(text(790, y + h / 2 + 4, question, size=12, fill=MUTED, anchor="start",
                      style="italic"))
        if i < len(rungs) - 1:
            o.append(arrow(98, y + h, 98, y + gap - 4))

    fy = y0 + len(rungs) * gap + 8
    o.append(box(60, fy, W - 120, 74, "#fffbeb", AMBER_BR, rx=12))
    o.append(text(84, fy + 26, "Stop at any rung that fails.", size=12.5, weight="700",
                  anchor="start", fill="#b45309"))
    o.append(text(84, fy + 48, "If sentence-level adaptivity cannot beat the dense baseline "
                  "on quality per FLOP, per-token routing will not rescue it.",
                  size=11.5, fill=MUTED, anchor="start"))
    o.append("</svg>")
    (ASSETS / "experiments.svg").write_text("\n".join(o) + "\n", encoding="utf-8")


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    write_logo()
    write_architecture()
    write_pipeline()
    write_experiments()
    for name in ("logo.svg", "logo-mark.svg", "architecture.svg", "pipeline.svg",
                 "experiments.svg"):
        p = ASSETS / name
        print(f"  {p.relative_to(ASSETS.parent)}  {p.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
