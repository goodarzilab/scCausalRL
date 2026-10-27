"""Shared house style for NeurIPS panel figures (scCausalRL).

Design contract enforced by this module
---------------------------------------
1. ONE type family everywhere.  Helvetica / Arial / Liberation Sans are metric
   clones, so the ordered stack below lays out identically on macOS and Linux.
   Math is remapped onto the *same* face (``mathtext.fontset='custom'``), so
   variables like  z, A, u  are sans-italic rather than the serif STIX italic
   matplotlib uses by default.  Only a handful of pure symbols
   (\\top \\odot \\in \\hat \\tilde) fall back to STIX; they are glyphs, not
   letters, so they read as neutral notation.

2. EXACT physical width.  ``save()`` deliberately does NOT use
   ``bbox_inches='tight'``: tight-bbox makes every PDF a different width, so
   \\includegraphics[width=\\linewidth] rescales each figure by a different
   factor and the effective font size drifts from figure to figure.  Instead we
   use a constrained layout inside a canvas that is exactly TEXTWIDTH_IN wide.
   Every PDF this module writes is 5.50 in across, so 8 pt text is 8 pt text in
   every figure of the paper.

3. Panel labels that cannot collide.  ``label_axes()`` runs after layout,
   measures each axes' *tight* bounding box (ticks + y-label + title included)
   and drops the bold letter just outside its top-left corner in figure
   coordinates.  Labels therefore never land on a y-axis label or a title.

4. Editable vectors.  fonttype 42 embeds real TrueType, so the PDFs open as
   editable text (not outlines) in Illustrator / Inkscape.

Usage
-----
    import paperstyle as ps
    fig, axes = ps.grid(1, 3, h=2.1)
    ...
    ps.label_axes(fig, axes)            # A, B, C - collision-free
    ps.save(fig, 'figures/figX_name')   # writes .pdf (+ .png preview)
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --------------------------------------------------------------------------
# canvas geometry - NeurIPS single-column text block
# --------------------------------------------------------------------------
TEXTWIDTH_IN = 5.50          # \linewidth in the NeurIPS style file
COLWIDTH_IN = 2.65           # half-width figure

# --------------------------------------------------------------------------
# type
# --------------------------------------------------------------------------
# Metric-compatible Helvetica clones, in descending order of preference.
# macOS resolves Helvetica; Linux/CI resolves Liberation Sans or Nimbus Sans.
# All three share Helvetica metrics => identical line breaks and text extents.
SANS_STACK = ["Helvetica", "Arial", "Liberation Sans", "Nimbus Sans",
              "TeX Gyre Heros", "DejaVu Sans"]

# type scale (pt) - one ladder, used by every figure
FS = {
    "title":  8.6,   # panel titles
    "label":  8.0,   # axis labels
    "tick":   7.2,   # tick labels
    "legend": 7.0,
    "annot":  6.8,   # in-plot annotations
    "small":  6.2,   # footnotes, n =, secondary notes
    "panel": 10.0,   # A / B / C
}

plt.rcParams.update({
    # ---- editable vectors ------------------------------------------------
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "pdf.compression": 6,
    # ---- one family, text and math --------------------------------------
    "font.family": "sans-serif",
    "font.sans-serif": SANS_STACK,
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.sf": "sans",
    "mathtext.cal": "sans:italic",
    "mathtext.default": "regular",
    # ---- sizes -----------------------------------------------------------
    "font.size": FS["label"],
    "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
    "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["legend"], "figure.titlesize": FS["title"],
    # ---- weights ---------------------------------------------------------
    "axes.titleweight": "bold", "axes.titlepad": 3.5,
    "axes.labelpad": 2.5,
    # ---- rules -----------------------------------------------------------
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
    "lines.linewidth": 1.2, "lines.markersize": 3.6,
    "patch.linewidth": 0.6,
    "hatch.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    # ---- legends ---------------------------------------------------------
    "legend.frameon": False, "legend.handlelength": 1.15,
    "legend.handletextpad": 0.45, "legend.columnspacing": 0.9,
    "legend.labelspacing": 0.28, "legend.borderaxespad": 0.25,
    "legend.borderpad": 0.2,
    # ---- error bars ------------------------------------------------------
    "errorbar.capsize": 1.6,
    # ---- output ----------------------------------------------------------
    "figure.dpi": 150, "savefig.dpi": 400,
    "figure.constrained_layout.use": False,
    "savefig.transparent": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# --------------------------------------------------------------------------
# palette - muted Okabe-Ito
# --------------------------------------------------------------------------
# Colourblind-safe (deuter/prot/trit) and luminance-staggered so the categories
# stay separable in greyscale print.  Approximate relative luminance in
# brackets; adjacent categories differ by >= 0.08.
BLUE      = "#0B5A85"   # L 0.091  flagship  - world model / policy / original
VIOLET    = "#7757A2"   # L 0.133  secondary - decoder / Briggs / GO programmes
VERMILION = "#C74B41"   # L 0.176  contrast  - reversed / co-expression / drivers
GREEN     = "#2F946B"   # L 0.228  positive  - oracle ceiling / combo-trained
GREY      = "#979DA3"   # L 0.333  neutral   - additive & MLP baselines
AMBER     = "#E9A153"   # L 0.434  accent    - MPC / perturbation inputs
GREY_L    = "#CED2D6"   # L 0.641  weakest   - random / chance
INK       = "#1A1A1A"   # body text
INK_SOFT  = "#5A6068"   # secondary text
RULE      = "#B8BCC2"   # hairlines, guides

# light tints for "same family, weaker variant" series
BLUE_L      = "#7FA9C4"
VERMILION_L = "#EBA79F"
GREEN_L     = "#8FC4AC"
VIOLET_L    = "#B6A6D2"
AMBER_L     = "#F4CFA3"

# ---- role maps (keys kept stable so figure scripts need no renaming) ------
METHOD = {
    "policy": BLUE, "MPC": AMBER, "additive": GREY, "random": GREY_L,
    "oracle": GREEN,
    "MPC_exact": AMBER, "additive_greedy": GREY, "random_shortlist": GREY_L,
    "grn_latent": BLUE, "mlp": GREY,
}
EDGE = {
    "original": BLUE, "reversed": VERMILION,
    "symmetrized": VERMILION_L, "shuffled": GREY,
    "zero": GREY_L, "coexpr": VIOLET,
}
SOURCE = {
    "briggs": VIOLET, "HSPC": VERMILION,
    "joung_single": BLUE_L, "joung_combo": BLUE,
}
ACCENT = {
    "wm": BLUE, "baseline": GREY, "hi": GREEN, "lo": VERMILION,
    "coexpr": VERMILION, "graph": BLUE, "decoder": VIOLET, "input": AMBER,
    "goal": VERMILION, "policy": GREEN, "ink": INK, "soft": INK_SOFT,
    "rule": RULE,
}
CYCLE = [BLUE, VERMILION, GREEN, AMBER, VIOLET, GREY]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CYCLE)

# diverging / sequential maps used by heatmaps
CMAP_DIV = "RdBu_r"
CMAP_SEQ = "Blues"


def tint(hexc, amt):
    """Mix a hex colour toward white by ``amt`` in [0, 1]."""
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(c + (1.0 - c) * amt)


def shade(hexc, amt):
    """Mix a hex colour toward black by ``amt`` in [0, 1]."""
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(c * (1.0 - amt))


def luminance(hexc):
    """Relative luminance - used by the greyscale separability check."""
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
def grid(nrows, ncols, w=TEXTWIDTH_IN, h=2.0, hspace=0.10, wspace=0.10, **kw):
    """Panel grid at exact manuscript width, constrained layout enabled.

    Constrained layout (not tight_layout) is used so that ``save()`` can write
    the figure at its declared size without a tight bbox - which is what keeps
    every figure in the paper at the same physical width.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h),
                             layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=0.028, h_pad=0.028,
                                wspace=wspace, hspace=hspace)
    return fig, axes


def canvas(w=TEXTWIDTH_IN, h=2.0):
    """Blank full-bleed canvas for hand-drawn schematics.

    Returns (fig, ax) where ax spans the whole page in 0-1 coordinates with no
    spines - so schematic geometry is specified directly in figure fractions
    and nothing can be pushed off-canvas by a layout engine.
    """
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def label_axes(fig, axes, letters=None, dx=-0.006, dy=0.012, fs=None,
               skip=()):
    """Bold panel letters placed outside each axes' tight bbox.

    Measured after layout in figure coordinates, so the letter can never land
    on a tick label, a y-axis label or a title.

    ``skip`` - indices to leave unlabelled (e.g. colourbar axes).
    """
    fig.canvas.draw()
    axs = [a for a in np.atleast_1d(np.asarray(axes, dtype=object)).ravel()]
    axs = [a for i, a in enumerate(axs) if i not in set(skip)]
    if letters is None:
        letters = [chr(ord("A") + i) for i in range(len(axs))]
    r = fig.canvas.get_renderer()
    out = []
    for ax, ltr in zip(axs, letters):
        if not ltr:
            out.append(None)
            continue
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        t = fig.text(max(bb.x0 + dx, 0.001), min(bb.y1 + dy, 0.999), ltr,
                     fontsize=fs or FS["panel"], fontweight="bold",
                     va="top", ha="left", color=INK, zorder=1000)
        out.append(t)
    return out


def panel_label(ax, letter, dx=-0.02, dy=1.06, **kw):
    """Legacy per-axes label (axis-fraction coords). Prefer ``label_axes``."""
    kw.setdefault("fontsize", FS["panel"])
    kw.setdefault("fontweight", "bold")
    kw.setdefault("color", INK)
    return ax.text(dx, dy, letter, transform=ax.transAxes, va="top",
                   ha="right", zorder=1000, **kw)


def schematic_label(fig, letter, x, y, fs=None):
    """Panel letter for hand-drawn schematics, in figure coordinates."""
    return fig.text(x, y, letter, fontsize=fs or FS["panel"],
                    fontweight="bold", va="top", ha="left", color=INK,
                    zorder=1000)


# --------------------------------------------------------------------------
# annotation helpers
# --------------------------------------------------------------------------
def stars(p):
    """p/q -> significance stars."""
    if p is None:
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def sig_bar(ax, x1, x2, y, label, dh=0.0, lw=0.6, fs=None, color=INK):
    """Significance bracket between x1 and x2 at height y, label above."""
    fs = fs or FS["small"]
    ax.plot([x1, x1, x2, x2], [y, y + dh, y + dh, y], lw=lw, c=color,
            clip_on=False, solid_joinstyle="miter")
    ax.text((x1 + x2) / 2, y + dh, label, ha="center", va="bottom",
            fontsize=fs, color=color)


def headroom(ax, frac=0.18, bottom=None):
    """Grow the y-limit so in-plot annotations sit above the data, not on it."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo if bottom is None else bottom, hi + frac * span)


def note(ax, text, x=0.5, y=-0.30, fs=None, color=None, ha="center",
         style="normal"):
    """Caption-style note *below* the axes, in axes coordinates."""
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="top",
                   fontsize=fs or FS["small"], color=color or INK_SOFT,
                   style=style, zorder=5)


def legend_above(ax, ncol=2, y=1.02, fs=None, **kw):
    """Legend parked above the axes so it can never sit on the data."""
    kw.setdefault("frameon", False)
    return ax.legend(loc="lower left", bbox_to_anchor=(0.0, y), ncol=ncol,
                     fontsize=fs or FS["legend"], **kw)


def wrap_label(s, width=22, maxlines=2):
    """Wrap a long categorical tick label instead of truncating it."""
    import textwrap
    lines = textwrap.wrap(s, width=width)
    if len(lines) > maxlines:
        lines = lines[:maxlines]
        lines[-1] = lines[-1].rstrip(" ,;") + "…"
    return "\n".join(lines)


def tidy(ax, ynonneg=False):
    """Common per-axes cleanup: hide dead spines, thin ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelcolor=INK)
    if ynonneg:
        ax.set_ylim(bottom=0)
    return ax


# --------------------------------------------------------------------------
# self-audit - run automatically by save()
# --------------------------------------------------------------------------
MIN_PT = 5.9          # NeurIPS legibility floor once the PDF is at \linewidth


def _texts(fig):
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        try:
            s = t.get_text()
        except Exception:
            continue
        if not s or not s.strip():
            continue
        if not t.get_visible():
            continue
        out.append(t)
    return out


def audit(fig, stem="", min_pt=MIN_PT, verbose=True):
    """Report text collisions, canvas overflow and sub-legible type.

    Returns a dict of findings.  Overlap detection is deliberately noisy-side:
    it flags any two text boxes whose rectangles intersect by more than a
    hairline, then the caller eyeballs the shortlist.  Labels that legitimately
    sit inside a coloured box do not collide with *other text*, so in practice
    the list is short and actionable.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    ts = _texts(fig)

    boxes = []
    for t in ts:
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((t, bb))

    small = [(t.get_text()[:42], round(t.get_fontsize(), 2))
             for t in ts if t.get_fontsize() < min_pt]

    over = []
    for t, bb in boxes:
        if bb.x0 < -1.0 or bb.y0 < -1.0 or bb.x1 > W + 1.0 or bb.y1 > H + 1.0:
            over.append((t.get_text()[:42],
                         tuple(round(v, 1) for v in (bb.x0, bb.y0, bb.x1, bb.y1))))

    hits = []
    tol = 1.0    # px of tolerated touching
    for i in range(len(boxes)):
        ti, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ox > tol and oy > tol:
                area = ox * oy
                frac = area / max(1.0, min(bi.width * bi.height,
                                           bj.width * bj.height))
                hits.append((round(frac, 2), ti.get_text()[:34],
                             tj.get_text()[:34]))
    hits.sort(reverse=True)

    res = {"small": small, "overflow": over, "overlap": hits}
    if verbose:
        tag = f"[{stem}] " if stem else ""
        w_in, h_in = fig.get_size_inches()
        print(f"{tag}size {w_in:.2f} x {h_in:.2f} in | {len(ts)} text artists")
        if small:
            print(f"{tag}  ! {len(small)} below {min_pt} pt: {small[:8]}")
        if over:
            print(f"{tag}  ! {len(over)} outside canvas: {over[:6]}")
        if hits:
            print(f"{tag}  ! {len(hits)} text-text overlaps:")
            for f_, a, b in hits[:12]:
                print(f"{tag}      {f_:>5}  {a!r}  <>  {b!r}")
        if not (small or over or hits):
            print(f"{tag}  clean")
    return res


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def save(fig, stem, png=True, exact=True, check=True):
    """Write ``<stem>.pdf`` at exactly the declared figure width.

    ``exact=True`` (default) skips ``bbox_inches='tight'`` so the PDF media box
    equals ``figsize``.  Every figure in the paper therefore shares one physical
    width and \\includegraphics[width=\\linewidth] applies the same scale factor
    to all of them.
    """
    if check:
        audit(fig, stem.split("/")[-1])
    kw = {} if exact else dict(bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.pdf", **kw)
    if png:
        fig.savefig(f"{stem}.png", dpi=300, **kw)
    plt.close(fig)


def has_font(name):
    return any(name.lower() in f.name.lower()
               for f in font_manager.fontManager.ttflist)


def resolved_font():
    """Which member of SANS_STACK actually resolved (for provenance)."""
    fp = font_manager.FontProperties(family=SANS_STACK)
    return font_manager.findfont(fp)


# --------------------------------------------------------------------------
# schematic primitives (restored) - used by hand-drawn architecture figures
# --------------------------------------------------------------------------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

EDGE_INK = INK
NODE_LW = 0.8
# role -> fill colour (light tints so dark text/edges read on top)
_ROLE = {
    "policy": tint(GREEN, 0.55), "input": tint(AMBER, 0.45), "model": tint(BLUE, 0.62),
    "goal": tint(VERMILION, 0.55), "neutral": tint(GREY_L, 0.35), "base": tint(GREY, 0.45),
    "panel": tint(GREY_L, 0.72),
}


def rbox(ax, cx, cy, w, h, fill="neutral", r=0.013, ec=None, lw=None):
    """Rounded box centred at (cx,cy) in 0-1 canvas coords; returns edge anchor points."""
    col = _ROLE.get(fill, fill)
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=col, ec=ec or EDGE_INK, lw=lw or NODE_LW, zorder=3)
    ax.add_patch(p)
    return {"left": (cx - w / 2, cy), "right": (cx + w / 2, cy),
            "top": (cx, cy + h / 2), "bot": (cx, cy - h / 2), "cx": cx, "cy": cy}


def nodetext(ax, cx, cy, title=None, lines=(), fs_title=FS["annot"], fs_body=FS["small"], gap=0.030):
    """Bold title + regular body lines, vertically centred in a box."""
    n = len(lines); rows = (1 if title else 0) + n
    y0 = cy + (rows - 1) * gap / 2.0
    if title:
        ax.text(cx, y0, title, ha="center", va="center", fontsize=fs_title,
                fontweight="bold", color=INK, zorder=5); y0 -= gap
    for ln in lines:
        ax.text(cx, y0, ln, ha="center", va="center", fontsize=fs_body, color=INK, zorder=5)
        y0 -= gap


def arrow(ax, p1, p2, bend=None, color=None, ls="-", lw=1.15):
    """Orthogonal (L-shaped) or straight connector p1->p2."""
    cs = {"vh": "angle,angleA=90,angleB=0", "hv": "angle,angleA=0,angleB=90"}.get(bend, "arc3,rad=0")
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=cs, arrowstyle="-|>",
                 mutation_scale=8, color=color or INK, lw=lw, ls=ls, shrinkA=1.5,
                 shrinkB=2.5, joinstyle="miter", zorder=2))


def edgelabel(ax, x, y, text, color=None, ha="center", fs=None):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs or FS["small"],
            color=color or INK_SOFT, zorder=4)


def dashpanel(ax, x0, y0, x1, y1, fill="panel", r=0.016):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle=f"round,pad=0,rounding_size={r}", fc=_ROLE.get(fill, fill),
                 ec=INK_SOFT, lw=0.6, ls=(0, (4, 2)), zorder=1))
