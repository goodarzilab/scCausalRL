"""Fig 1 - Structural causal world-model ARCHITECTURE (vector PDF).

WM-only schematic (the planner loop lives in a separate figure). Every box /
arrow / matrix is a matplotlib patch, so the output is fully editable vector art.

Layout contract
---------------
The whole figure is drawn in ONE full-bleed axes with x, y in [0, 1], so no
layout engine can shift an element and nothing is clipped by an axes boundary.
Panel bands are declared once below; every element is positioned relative to
them.  Type never drops below ps.MIN_PT, and ps.save() runs a collision /
overflow audit before writing.

The A -> B -> C reading order: A builds the latent z (mechanism of one
intervention on a cell); B is the readout decoder z -> genes; C composes k
interventions into ONE exact predicted effect, reusing the solve (A) and the
decoder (B).

Equations transcribed from code (NOT guessed):
  A  SEM + shift-mutilate do-operator, fed by an encoder for the control cell
       eps0     = encoder q_phi(z|x) of a control cell         causal_vae.py:32,35
       S        = sum_g compute_shift(g)                       causal_vae.py:443-446
       S_int    = W_out( ((sum x)^2 - sum x^2)/2 )  (combos)   causal_vae.py:408-419
       M        = diag(1 - u) ,  A~ = A.M  (col i x (1-u_i))   dag.py:376-379
       z        = (I - diag(1-u) A^T)^-1 (eps0 + S + S_int)    dag.py:390-391 + cvae:446
       rest:  z0 = (I - A^T)^-1 eps0   (u=0)                   dag.py:245-268
       u_i=1 -> parents of program i cut (do); u_i=0 -> obs.   dag.py:354-359
  B  masked-GO ("SENA") decoder
       xhat     = z @ D_eff^T + b ,  D_eff = D (*) M_GO        decoder.py:49-61
       D in R^{4366 x 357}: cols = 357 GO:BP programs, rows = 4366 genes;
       mask frozen, off-annotation entries lambda-shrunk
  C  exact & compositional k-TF perturbation
       k-TF set -> sum_g S_g (+ S_int) -> ONE solve z_post -> Delta = D_eff(z_post - z_ctrl)

Run:  cd paper/figures && python fig1.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse
import paperstyle as ps


# ----------------------------------------------------------------- geometry
FIGW, FIGH = ps.TEXTWIDTH_IN, 4.05
ASP = FIGW / FIGH                      # x-compression for true circles

fig, ax = ps.canvas(FIGW, FIGH)

# ----------------------------------------------------------------- palette
C_SHIFT = ps.AMBER        # per-TF shift / TF inputs
C_SEM   = ps.BLUE         # latent-DAG SEM / world model / encoder
C_DEC   = ps.VIOLET       # masked-GO decoder
C_GOOD  = ps.GREEN        # property tags
C_CUT   = ps.VERMILION    # mutilation / severed edges
C_GREY  = ps.GREY
INK     = ps.INK
SOFT    = ps.INK_SOFT

FS_T   = 8.6              # panel titles
FS_BOX = 6.6              # box captions
FS_LBL = 6.4              # element labels
FS_EQ  = 8.2              # display equations
FS_SUB = 6.2              # footnotes  (== ps.MIN_PT floor + margin)


# ----------------------------------------------------------------- helpers
def tint(hexc, amt):
    return ps.tint(hexc, amt)


def box(cx, cy, w, h, text="", fc="white", ec=INK, lw=0.9, fs=FS_BOX,
        tc=INK, rounding=0.012, weight="normal", z=2, ls="-", va="center",
        ls_pad=1.15):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad=0.0,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, mutation_aspect=1.0)
    ax.add_patch(b)
    if text:
        ty = cy if va == "center" else (cy + h / 2 - 0.010)
        ax.text(cx, ty, text, ha="center", va=va, fontsize=fs, color=tc,
                weight=weight, zorder=z + 1, linespacing=ls_pad)
    return dict(cx=cx, cy=cy, w=w, h=h)


def side(b, where):
    cx, cy, w, h = b["cx"], b["cy"], b["w"], b["h"]
    return {"top": (cx, cy + h / 2), "bot": (cx, cy - h / 2),
            "left": (cx - w / 2, cy), "right": (cx + w / 2, cy),
            "c": (cx, cy)}[where]


def arrow(p0, p1, color=INK, lw=0.9, rad=0.0, ls="-", sa=2, sb=2,
          style="-|>", z=4, ms=7):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=sa, shrinkB=sb,
                                mutation_scale=ms,
                                connectionstyle=f"arc3,rad={rad}"))


def ecirc(c, r, **kw):
    e = Ellipse(c, width=2.0 * r / ASP, height=2.0 * r, **kw)
    ax.add_patch(e)
    return e


def vvec(cx, cy, n, cw, ch, base, vals=None, ec="white", lw=0.3, z=3):
    """Vertical vector = column of n cells centred at (cx, cy)."""
    if vals is None:
        vals = np.linspace(0.25, 0.95, n)
    vals = np.clip(np.asarray(vals, dtype=float), 0.10, 1.0)
    total = n * ch
    y0 = cy + total / 2 - ch
    for i in range(n):
        ax.add_patch(Rectangle((cx - cw / 2, y0 - i * ch), cw, ch,
                               fc=tint(base, 1.0 - float(vals[i])), ec=ec,
                               lw=lw, zorder=z))
    ax.add_patch(Rectangle((cx - cw / 2, cy - total / 2), cw, total,
                           fill=False, ec=tint(base, 0.10), lw=0.6, zorder=z + 1))
    return dict(top=(cx, cy + total / 2), bot=(cx, cy - total / 2),
                left=(cx - cw / 2, cy), right=(cx + cw / 2, cy))


def cutmark(x, y, s=0.011, color=C_CUT, lw=1.1, z=8):
    """Small severed-edge X (drawn, not a glyph - keeps one font in the PDF)."""
    ax.plot([x - s / ASP, x + s / ASP], [y - s, y + s], c=color, lw=lw,
            zorder=z, solid_capstyle="round")
    ax.plot([x - s / ASP, x + s / ASP], [y + s, y - s], c=color, lw=lw,
            zorder=z, solid_capstyle="round")


# =========================================================== PANEL A
ax.text(0.030, 0.985, "Structural causal SEM with a shift-fed mutilation "
        "do-operator", fontsize=FS_T, weight="bold", ha="left", va="top",
        color=INK)
ps.schematic_label(fig, "A", 0.002, 0.985)

Y_OBS = 0.910                                # observational sub-stream
Y_INT = 0.775                                # interventional sub-stream
Y_MRG = 0.832                                # merge / SEM centre line
rng = np.random.default_rng(4)

# ------------------------------------------------ observational sub-stream
# a real control cell enters the model through the encoder -> rest state eps0
b_cell = box(0.062, Y_OBS, 0.100, 0.054, "control cell " + r"$x$" +
             "\n(4366 genes)", fc=tint(C_GREY, 0.86), ec=C_GREY, lw=0.9,
             fs=FS_SUB, ls_pad=1.30)
b_enc = box(0.200, Y_OBS, 0.092, 0.054, "encoder\n" + r"$q_\phi(z\mid x)$",
            fc=tint(C_SEM, 0.92), ec=C_SEM, lw=1.0, fs=FS_SUB, ls_pad=1.30)
arrow(side(b_cell, "right"), side(b_enc, "left"), color=INK, lw=0.9, ms=6)
h_e0 = vvec(0.310, Y_OBS, 5, 0.019, 0.0125, C_SEM,
            vals=np.abs(rng.standard_normal(5)) * 0.5 + 0.25)
arrow(side(b_enc, "right"), h_e0["left"], color=C_SEM, lw=0.9, ms=6, sb=1)
ax.text(0.330, Y_OBS, "rest state\n" + r"$\varepsilon_0$", ha="left",
        va="center", fontsize=FS_LBL, color=C_SEM, linespacing=1.25)

# ------------------------------------------------ interventional sub-stream
ax.text(0.010, 0.867, "intervened TFs", ha="left", va="center",
        fontsize=FS_LBL, color=C_SHIFT, weight="bold")
tf_y = [Y_INT + 0.050, Y_INT, Y_INT - 0.050]     # 0.825 / 0.775 / 0.725
tf_boxes = [box(0.045, yy, 0.052, 0.042, lb, fc=tint(C_SHIFT, 0.80),
                ec=C_SHIFT, lw=0.9, fs=7.0)
            for lb, yy in zip([r"$g_1$", r"$g_2$", r"$g_3$"], tf_y)]

bx_emb = box(0.150, Y_INT, 0.096, 0.150, fc=tint(C_SHIFT, 0.90), ec=C_SHIFT,
             lw=1.0)
ax.text(0.150, Y_INT, "per-TF\nESM / PCA\nshift encoder\n" + r"$f\!:g\mapsto S_g$",
        ha="center", va="center", fontsize=FS_SUB, color=INK, linespacing=1.30)
for b in tf_boxes:
    arrow(side(b, "right"), (0.102, b["cy"]), color=C_SHIFT, lw=0.8, ms=6)

ax.text(0.243, 0.862, "shift  " + r"$S_g$", ha="center", va="center",
        fontsize=FS_SUB, color=C_SHIFT, weight="bold")
shift_anchor = []
for j, yy in enumerate(tf_y):
    v = np.abs(rng.standard_normal(4)) * (0.55 + 0.13 * j)
    h = vvec(0.243, yy, 4, 0.017, 0.0105, C_SHIFT, vals=np.clip(v, 0.15, 0.98))
    shift_anchor.append(h)
    arrow(side(bx_emb, "right"), h["left"], color=C_SHIFT, lw=0.7, ms=5.5, sb=1)

# summation node -> S (= sum_g S_g + S_int)
sumc = (0.300, Y_INT)
ecirc(sumc, 0.022, fc="white", ec=INK, lw=0.9, zorder=5)
ax.text(sumc[0], sumc[1], r"$\Sigma$", ha="center", va="center", fontsize=8.5,
        zorder=6, color=INK)
for h in shift_anchor:
    arrow(h["right"], (sumc[0] - 0.022 / ASP, sumc[1]), color=C_SHIFT, lw=0.7,
          ms=5.5, rad=(0.10 if h["right"][1] > sumc[1] else -0.10), sb=2)
h_S = vvec(0.352, Y_INT, 5, 0.019, 0.0135, C_SHIFT)
arrow((sumc[0] + 0.022 / ASP, sumc[1]), h_S["left"], color=C_SHIFT, lw=0.9,
      ms=6, sb=1)

# learned rank-32 interaction head S_int (combinations only) feeds the sum node
h_int = vvec(0.300, 0.700, 4, 0.017, 0.0104, C_SHIFT,
             vals=np.abs(rng.standard_normal(4)) * 0.4 + 0.20)
arrow(h_int["top"], (sumc[0], sumc[1] - 0.022), color=C_SHIFT, lw=0.8, ms=5.5,
      sb=1)
ax.text(0.322, 0.700, "interaction head\n" + r"$S_{\mathrm{int}}$ (combos)",
        ha="left", va="center", fontsize=FS_SUB, color=C_SHIFT,
        linespacing=1.25)

# merge  eps0 + S + S_int  ->  SEM
oc = (0.428, Y_MRG)
ecirc(oc, 0.016, fc="white", ec=INK, lw=0.9, zorder=6)
ax.text(oc[0], oc[1], "+", ha="center", va="center", fontsize=9, zorder=7,
        color=INK)
arrow(h_S["right"], (oc[0] - 0.016 / ASP, oc[1] - 0.006), color=C_SHIFT,
      lw=0.9, ms=6, rad=0.18, sb=1)
arrow(h_e0["bot"], (oc[0] - 0.016 / ASP, oc[1] + 0.008), color=C_SEM, lw=0.9,
      ms=6, rad=-0.22, sb=1)

# --- SEM box: latent DAG A + mutilation gate ------------------------------
bx_sem = box(0.680, Y_MRG, 0.420, 0.240, fc=tint(C_SEM, 0.955), ec=C_SEM,
             lw=1.2, rounding=0.014)
ax.text(0.680, 0.936, "linear structural equation model", ha="center",
        va="center", fontsize=FS_BOX, color=C_SEM, weight="bold")
arrow((oc[0] + 0.016 / ASP, oc[1]), (0.470, Y_MRG), color=INK, lw=1.0, ms=7)

nx = {0: (0.560, 0.878), 1: (0.632, 0.900), 2: (0.706, 0.874),
      3: (0.582, 0.806), 4: (0.668, 0.808), 5: (0.744, 0.822)}
edges = [(0, 3), (1, 0), (1, 2), (2, 5), (0, 4), (3, 4), (1, 4), (2, 4), (5, 4)]
intervened = {2, 4}
for a_, b_ in edges:
    into_int = b_ in intervened
    arrow(nx[a_], nx[b_],
          color=(C_CUT if into_int else tint(C_SEM, 0.35)),
          lw=(0.8 if into_int else 0.7),
          ls=((0, (2, 1.4)) if into_int else "-"),
          sa=4, sb=5, rad=0.12, ms=5.5, z=3)
for k, (xx, yy) in nx.items():
    isint = k in intervened
    ecirc((xx, yy), 0.0135, zorder=6,
          fc=(tint(C_SHIFT, 0.15) if isint else "white"),
          ec=(C_CUT if isint else C_SEM), lw=1.0)
for k in intervened:
    cutmark(nx[k][0] + 0.020, nx[k][1] + 0.007)

# labels live in the free right-hand third of the box - clear of every node
ax.text(0.792, 0.884, "latent\nDAG  " + r"$A$", ha="left", va="center",
        fontsize=FS_LBL, color=C_SEM, linespacing=1.25)
ax.text(0.792, 0.822, "intervened\nprogram", ha="left", va="center",
        fontsize=FS_SUB, color=C_CUT, linespacing=1.25)
ax.text(0.680, 0.752,
        r"mutilation gate  $u\in[0,1]$:   $\tilde{A}=A\,\mathrm{diag}(1-u)$",
        fontsize=FS_LBL, color=C_CUT, ha="center", va="center", weight="bold")
ax.text(0.680, 0.700, "dashed edges = parents cut by " + r"$u$",
        fontsize=FS_SUB, color=SOFT, ha="center", va="center")

# --- output latent state z  ->  decoder (panel B) --------------------------
arrow(side(bx_sem, "right"), (0.928, Y_MRG), color=INK, lw=1.0, ms=7)
h_z = vvec(0.952, Y_MRG, 6, 0.017, 0.0135, C_DEC)
ax.text(0.952, 0.894, "latent\nstate  " + r"$z$", ha="center", va="bottom",
        fontsize=FS_LBL, color=C_DEC, weight="bold", linespacing=1.25)
ax.text(0.952, 0.784, r"$\rightarrow$ decoder" + "\n(panel B)", ha="center",
        va="top", fontsize=FS_SUB, color=SOFT, linespacing=1.25)

# --- master closed-form solve + one gate gloss -----------------------------
ax.text(0.500, 0.642,
        r"$z=(I-\mathrm{diag}(1-u)\,A^{\top})^{-1}\,"
        r"(\varepsilon_0 + \sum_g S_g + S_{\mathrm{int}})$",
        ha="center", va="center", fontsize=FS_EQ, color=INK)
ax.text(0.500, 0.586,
        r"$u=1$: cut a program's parents (do)"
        "            "
        r"$u=0$: observational row",
        ha="center", va="center", fontsize=FS_SUB, color=SOFT)

# separator rule between the panel A band and the panel B / C band
ax.plot([0.02, 0.98], [0.560, 0.560], lw=0.5, color=ps.RULE, zorder=1)


# =========================================================== PANEL B
ax.text(0.030, 0.522, 'Masked-GO ("SENA") decoder', fontsize=FS_T,
        weight="bold", ha="left", va="top", color=INK)
ps.schematic_label(fig, "B", 0.002, 0.522)

# z as a horizontal program vector
zn, zx0, zcw = 10, 0.130, 0.0175
zc_y = 0.432
for i in range(zn):
    v = 0.30 + 0.62 * abs(np.sin(1.3 * i + 0.6))
    ax.add_patch(Rectangle((zx0 + i * zcw, zc_y - 0.013), zcw * 0.9, 0.026,
                           fc=tint(C_SEM, 1.0 - v), ec="white", lw=0.3, zorder=3))
ax.add_patch(Rectangle((zx0, zc_y - 0.013), zn * zcw * 0.9 + zcw * 0.1, 0.026,
                       fill=False, ec=tint(C_SEM, 0.10), lw=0.6, zorder=4))
ax.text(zx0 + zn * zcw / 2, zc_y + 0.020, r"$z$  —  357 GO:BP program latents",
        ha="center", va="bottom", fontsize=FS_LBL, color=C_SEM)

# decoder matrix D_eff
mx0, my0, mw, mh = 0.120, 0.148, 0.195, 0.230
ncol, nrow = 10, 16
rg = np.random.default_rng(7)
member = np.zeros((nrow, ncol))
for c in range(ncol):
    centre = (c + 0.5) / ncol * nrow
    for r_ in range(nrow):
        p = np.exp(-0.5 * ((r_ - centre) / 1.5) ** 2)
        member[r_, c] = 1.0 if rg.random() < (0.15 + 0.85 * p) else 0.0
cw, ch = mw / ncol, mh / nrow
for r_ in range(nrow):
    for c in range(ncol):
        fcc = (tint(C_DEC, 0.10 + 0.30 * (c % 2)) if member[r_, c]
               else tint(C_DEC, 0.90))
        ax.add_patch(Rectangle((mx0 + c * cw, my0 + (nrow - 1 - r_) * ch),
                               cw * 0.94, ch * 0.9, fc=fcc, ec="white",
                               lw=0.2, zorder=3))
ax.add_patch(Rectangle((mx0, my0), mw, mh, fill=False, ec=tint(C_DEC, 0.10),
                       lw=0.8, zorder=4))
ax.text(mx0 - 0.014, my0 + mh / 2, "4366 genes", rotation=90, ha="center",
        va="center", fontsize=FS_SUB, color=SOFT)
ax.text(mx0 + mw / 2, my0 - 0.014, "columns = 357 GO:BP programs",
        ha="center", va="top", fontsize=FS_SUB, color=SOFT)
# matrix shape label sits ABOVE the matrix, left of x-hat's column
ax.text(mx0 + mw + 0.014, my0 + mh + 0.010,
        r"$D_{\mathrm{eff}}\in\mathbf{R}^{4366\times 357}$",
        ha="left", va="bottom", fontsize=FS_LBL, color=C_DEC, weight="bold")

arrow((zx0 + zn * zcw / 2, zc_y - 0.015), (mx0 + mw / 2, my0 + mh + 0.006),
      color=INK, lw=0.9, ms=6)

# output gene vector x-hat, well clear of the shape label above
gx, gn = 0.400, 16
gcw, gch = 0.026, mh / gn
for r_ in range(gn):
    v = 0.25 + 0.7 * abs(np.sin(0.9 * r_ + 0.3))
    ax.add_patch(Rectangle((gx - gcw / 2, my0 + (gn - 1 - r_) * gch), gcw,
                           gch * 0.9, fc=tint(C_DEC, 1.0 - v), ec="white",
                           lw=0.2, zorder=3))
ax.add_patch(Rectangle((gx - gcw / 2, my0), gcw, mh, fill=False,
                       ec=tint(C_DEC, 0.10), lw=0.6, zorder=4))
ax.text(gx, my0 - 0.014, r"$\hat{x}$", ha="center", va="top", fontsize=7.6,
        color=C_DEC)
arrow((mx0 + mw + 0.004, my0 + mh / 2), (gx - gcw / 2 - 0.008, my0 + mh / 2),
      color=INK, lw=0.9, ms=6)

ax.text(0.235, 0.088,
        r"$\hat{x}=D_{\mathrm{eff}}\,z,\quad D_{\mathrm{eff}}"
        r"=D\odot M_{\mathrm{GO}}$",
        ha="center", va="center", fontsize=7.4, color=INK)
ax.text(0.235, 0.044, "GO:BP mask frozen; off-annotation entries "
        r"$\lambda$-shrunk", ha="center", va="center", fontsize=FS_SUB,
        color=SOFT)


# =========================================================== PANEL C
ax.text(0.545, 0.522, "Exact, compositional multi-perturbation",
        fontsize=FS_T, weight="bold", ha="left", va="top", color=INK)
ps.schematic_label(fig, "C", 0.512, 0.522)

CX = 0.770                                   # panel C centre line
b_tf = box(CX, 0.455, 0.430, 0.048,
           r"$k$-TF cocktail   $\{g_1,\dots,g_k\}$",
           fc=tint(C_SHIFT, 0.82), ec=C_SHIFT, lw=1.0, fs=FS_BOX)

# sum the k per-TF shifts (+ S_int)
sy = 0.378
for j, xx in enumerate((0.610, 0.664, 0.718)):
    for i in range(4):
        v = 0.35 + 0.5 * abs(np.sin(1.1 * (i + j)))
        ax.add_patch(Rectangle((xx - 0.008, sy - 0.020 + i * 0.0105), 0.016,
                               0.0092, fc=tint(C_SHIFT, 1.0 - v), ec="white",
                               lw=0.25, zorder=3))
    if j < 2:
        ax.text(xx + 0.027, sy, "+", ha="center", va="center", fontsize=8,
                color=SOFT)
ax.text(0.745, sy, "=", ha="center", va="center", fontsize=8, color=SOFT)
vvec(0.779, sy, 4, 0.018, 0.0105, C_SHIFT)
ax.text(0.808, sy, r"$\sum_g S_g + S_{\mathrm{int}}$", ha="left", va="center",
        fontsize=FS_BOX, color=C_SHIFT)
arrow(side(b_tf, "bot"), (0.664, sy + 0.031), color=INK, lw=0.9, ms=6)

# THE SAME one closed-form solve as panel A - no rollout, any k
b_solve = box(CX, 0.268, 0.450, 0.130, "", fc=tint(C_SEM, 0.93), ec=C_SEM,
              lw=1.2, rounding=0.012)
ax.text(CX, 0.310, "ONE closed-form solve — reuses the solve (A)",
        ha="center", va="center", fontsize=FS_BOX, weight="bold", color=C_SEM)
ax.text(CX, 0.267,
        r"$z_{\mathrm{post}}=(I-\mathrm{diag}(1-u)A^{\top})^{-1}"
        r"(\varepsilon_0+\sum_g S_g+S_{\mathrm{int}})$",
        ha="center", va="center", fontsize=6.6, color=INK)
ax.text(CX, 0.226, "one solve for any " + r"$k$" + ", no rollout", ha="center",
        va="center", fontsize=FS_SUB, color=SOFT, style="italic")
arrow((0.664, sy - 0.024), (0.664, 0.337), color=INK, lw=0.9, ms=6)

# decode through B -> predicted perturbation effect
b_out = box(CX, 0.150, 0.400, 0.056, "", fc=tint(C_DEC, 0.85), ec=C_DEC,
            lw=0.9, rounding=0.012)
ax.text(CX, 0.164,
        r"$\Delta_{\mathrm{pert}}=D_{\mathrm{eff}}\,"
        r"(z_{\mathrm{post}}-z_{\mathrm{ctrl}})$",
        ha="center", va="center", fontsize=FS_BOX, color=INK)
ax.text(CX, 0.136, "predicted effect — decode via " + r"$B$" + " (novel combo)",
        ha="center", va="center", fontsize=FS_SUB, color=SOFT)
arrow((CX, 0.203), side(b_out, "top"), color=INK, lw=0.9, ms=6)

# one concise contrast line (plain text, not pills)
ax.text(CX, 0.070, "one solve for any " + r"$k$" +
        "; exact & zero-shot for unseen combos,\nunlike amortized foundation "
        "models", ha="center", va="center", fontsize=FS_SUB, color=SOFT,
        linespacing=1.35)

ps.save(fig, "fig1")
print("wrote fig1.pdf / .png")
