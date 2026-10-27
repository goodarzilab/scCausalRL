"""Fig 5 - Planner ARCHITECTURE over the frozen world model (vector PDF).

A  goal-conditioned trajectory-balance GFlowNet loop:
     target fate x*  ->  planner pi_theta  ->  TF cocktail  ->  frozen WM do-op
     ->  reward R  ->  TB gradient back to the planner.
B  the GRN-latent slot head: GRN latents enter the action logits as a *gated
   residual* on the MLP base, so the graph can only add signal, never replace it.

     l_cand = W_head feat + beta [ (I + A^T + (A^T)^2) W_f feat ] W_dec^T
     beta learnable (init 0.5);  A^T frozen (row-normalised + self-loops), 357x357
     V = 280 TFs + STOP                                     set_policy.py:77-88

Drawing conventions (shared kit in paperstyle: rbox / arrow / dashpanel)
-----------------------------------------------------------------------
* Role is carried by the FILL; every node takes the same dark hairline edge.
* Connectors are orthogonal L-shapes with rounded corners, so panel A reads as
  a circuit rather than a bundle of free-hand arcs.  Panel A is laid out as a
  literal rectangular loop with the goal parked in its centre - the one place
  no connector passes through.
* Drawn in ONE full-bleed axes with x, y in [0, 1] (ps.canvas): no axes
  boundary can clip an element and no layout engine can move one.  Type never
  drops below ps.MIN_PT; ps.save() audits collisions and overflow before write.

Run:  cd paperv3/figures && python fig5.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import paperstyle as ps

FIGW, FIGH = ps.TEXTWIDTH_IN, 2.55
ASP = FIGW / FIGH
fig, ax = ps.canvas(FIGW, FIGH)

INK, SOFT = ps.INK, ps.INK_SOFT
FS_T, FS_N, FS_S, FS_EQ = 8.6, 6.9, 6.2, 7.4
GOAL = ps.VERMILION


def box(cx, cy, w, h, fill, title=None, lines=(), fs=FS_N, fsb=FS_S, gap=0.030):
    b = ps.rbox(ax, cx, cy, w, h, fill=fill, r=0.013)
    ps.nodetext(ax, cx, cy, title, lines, fs_title=fs, fs_body=fsb, gap=gap)
    return b


# =========================================================== PANEL A
ax.text(0.030, 0.988, "Planner over the frozen world model", fontsize=FS_T,
        weight="bold", ha="left", va="top", color=INK)

# --- the four stations of the loop, on a rectangle -------------------------
plan = box(0.158, 0.742, 0.252, 0.128, "policy",
           r"GFlowNet planner $\pi_\theta$", ("MLP policy head; cond. $\log Z(x^{\star})$",),
           gap=0.036)
cock = box(0.500, 0.916, 0.272, 0.054, "input",
           r"TF cocktail  $\{g_1,\dots,g_k\}$")
wm = box(0.856, 0.742, 0.252, 0.128, "model",
         "frozen world model",
         ("do-operator", r"$\to$ predicted $\Delta$ state"), gap=0.032)
rew = box(0.500, 0.578, 0.372, 0.054, "goal",
          r"reward $\Phi$ = ensemble-pessimistic fate match", fs=FS_S)

# --- the goal sits in the dead centre of the loop --------------------------
tgt = box(0.500, 0.742, 0.238, 0.054, "goal", r"target fate  $x^{\star}$")
ps.arrow(ax, tgt["left"], plan["right"], color=GOAL, ls=(0, (3, 2)), lw=1.15)
ps.edgelabel(ax, 0.298, 0.792, "condition on goal", color=GOAL, ha="left")

# --- clockwise circuit: every hop is a single rounded elbow ----------------
ps.arrow(ax, plan["top"], cock["left"], bend="vh")
ps.edgelabel(ax, 0.262, 0.930, "propose")

ps.arrow(ax, cock["right"], wm["top"], bend="hv")
ps.edgelabel(ax, 0.748, 0.930, "apply do-op")

ps.arrow(ax, wm["bot"], rew["right"], bend="vh")
ps.edgelabel(ax, 0.778, 0.604, "predict  " + r"$\hat{\Delta}\to \Phi$")

ps.arrow(ax, rew["left"], plan["bot"], bend="hv", color=ps.GREEN,
         ls=(0, (3, 2)))
ps.edgelabel(ax, 0.180, 0.640, "policy update (TB)", color=ps.GREEN, ha="left")

# --- the objective the loop is trained under -------------------------------
ax.text(0.500, 0.516,
        r"trajectory balance:   $\mathcal{L}=(\log Z(x^{\star})"
        r"+\sum_i\log\pi_\theta(a_i\,|\,s_i)-\beta\,\Phi)^{2}$,   "
        r"$R=e^{\beta\Phi},\ \beta=48$",
        ha="center", va="center", fontsize=FS_EQ, color=INK)

ax.set_ylim(0.452, 1.0)   # single panel: crop to the planner loop

ps.save(fig, "fig5")
print("wrote fig5.pdf / .png")
