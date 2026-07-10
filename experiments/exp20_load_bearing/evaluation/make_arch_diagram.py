"""Architecture schematic of the exp20 shift-encoder causal-VAE (base model), with equations:
control encoder -> exogenous noise; ESM shift encoder + gate (L0 sparse OR hard-self localized);
mutilated structural do-op; decoder + GRN readout; loss. Pure matplotlib (no data)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, AQUA, YEL, VIO, ORA, GRY = '#2a78d6', '#1baf7a', '#eda100', '#4a3aa7', '#eb6834', '#eef0f3'
INK, MUT = '#0b0b0b', '#52514e'
fig, ax = plt.subplots(figsize=(15, 8.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 8.6); ax.axis('off')


def box(x, y, w, h, fc, title, eqs, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=INK, lw=1.4, alpha=0.96))
    ax.text(x + w/2, y + h - 0.30, title, ha='center', va='top', fontsize=11, fontweight='bold', color=tc)
    ax.text(x + w/2, y + h - 0.78, eqs, ha='center', va='top', fontsize=10.5, color=INK)


def arrow(x1, y1, x2, y2, lab='', c=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16,
                                 lw=1.8, color=c, connectionstyle=f"arc3,rad={rad}"))
    if lab:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, lab, ha='center', va='bottom', fontsize=9.5, color=c)

ax.text(7.5, 8.4, "Shift-encoder causal-VAE (exp20 base): perturbation as a mutilated structural do-operator",
        ha='center', fontsize=13.5, fontweight='bold', color=INK)

# --- inputs ---
box(0.2, 5.7, 2.7, 1.5, GRY, "Control cells", r"$x_{ctrl}\in R^{G}$" + "\n(basal transcriptome)")
box(0.2, 2.6, 2.7, 1.5, GRY, "Perturbation ID", r"$e_g$ = ESM2 protein" + "\nembedding of TF $g$")

# --- encoder ---
box(3.5, 5.6, 3.0, 1.7, GRY, "Encoder  q(ε|x)",
    r"$\mu,\log\sigma^2=\mathrm{Enc}(x_{ctrl})$" + "\n" + r"$\varepsilon\sim\mathcal{N}(\mu,\sigma^2)$")
# --- shift encoder + gate ---
box(3.5, 2.2, 3.0, 2.5, GRY, "Shift encoder  (+ gate)",
    r"$S=c\,(g\odot \mathrm{MLP}(e_g))$" + "\n\n" +
    r"L0 gate: $g=\sigma(\frac{\ell+\gamma}{\tau})$" + "\n" +
    r"penalty $\lambda_0\,\overline{\sigma(\ell/\tau)}$" + "\n\n" +
    "OR localized (hard):\n" + r"$g=\mathbf{1}[\mathrm{top}\!-\!k\ |D_{g,\cdot}|]$")

# --- do-operator / SCM ---
box(7.4, 4.0, 3.7, 2.7, GRY, "Structural do-operator  (mutilated SCM)",
    r"$u=\mathrm{clip}(\beta|S|,\,0,\,u_{max})$" + "\n\n" +
    r"$z_{post}=(I-\mathrm{diag}(1-u)\,A^{\top})^{-1}(\varepsilon+S)$" + "\n\n" +
    r"$z_{ctrl}=(I-A^{\top})^{-1}\varepsilon$")
box(7.4, 2.1, 3.7, 1.4, GRY, "Latent DAG  A",
    r"$A=\mathrm{gate}\odot W\odot(1-I)$,  spectral-clamped" + "\n" + r"$\rho(A)<1\Rightarrow(I-A^{\top})$ invertible")

# --- decoder + readouts ---
box(11.7, 4.6, 3.1, 2.2, GRY, "Decoder  D",
    r"$\hat{x}=D\,z+b$" + "\n\n" + r"$\Delta\hat{x}=D(z_{post}-z_{ctrl})$" + "\n\n" +
    "D: learned / masked-\nGO(λ) / frozen resid-PCA")
box(11.7, 2.1, 3.1, 1.9, GRY, "Read-outs",
    "predicted pert profile\n" + r"$\hat{x}_{pert}$" + "\nGRN  " + r"$G=D\,A\,D^{\top}$")

# --- loss ---
box(2.2, 0.15, 10.6, 1.5, '#f4f4f2',
    "Training loss",
    r"$\mathcal{L}=\|\hat{x}-x\|^2+\beta\,\mathrm{KL}+\lambda_p\,(\|\Delta\hat{x}-\Delta_{obs}\|^2+\mathrm{InfoNCE}(\Delta\hat{x}-\overline{\Delta},\ \Delta_{obs}-\overline{\Delta}))+\lambda_0\,\mathcal{L}_{L0}$"
    + "\n" + "recon  +  KL  +  (raw-Δ MSE  +  drift-removed contrastive)  +  L0 sparsity")

# --- arrows ---
arrow(2.9, 6.45, 3.5, 6.45, r"$x_{ctrl}$")                         # ctrl -> encoder
arrow(2.9, 3.35, 3.5, 3.45, r"$e_g$")                              # TF -> shift
arrow(6.5, 6.2, 7.6, 6.0, r"$\varepsilon$", MUT)                   # encoder -> SCM
arrow(6.5, 3.6, 7.6, 4.6, r"$S,\,u$", MUT)                         # shift -> SCM
arrow(9.25, 3.5, 9.25, 3.98, r"$A$", MUT)                          # DAG -> SCM
arrow(11.1, 5.4, 11.7, 5.6, r"$z_{post},z_{ctrl}$", MUT)           # SCM -> decoder
arrow(13.2, 4.6, 13.2, 4.0, '', MUT)                               # decoder -> readouts
ax.text(0.2, 1.75, "Fixes:  (1) single mutilated solve, S into exogenous noise   "
        "(2) sparse L0 gate  /  hard-self localization   —  force downstream effects through A",
        fontsize=9.2, color=MUT, style='italic')

fig.tight_layout()
import os; os.makedirs('results/figures', exist_ok=True)
fig.savefig('results/figures/architecture_diagram.png', dpi=150, bbox_inches='tight')
print("wrote results/figures/architecture_diagram.png")
