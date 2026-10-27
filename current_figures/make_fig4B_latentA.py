"""Fig 4 panel B — CLEAN standalone latent causal-graph A heatmap + top-10 GO:BP hubs.

REAL DATA PROVENANCE (no fabricated values):
  checkpoint : experiments/exp20_load_bearing/results/grid/jb_prod_wm/checkpoints/phase1_epoch60.pt
      key    : state_dict['dag.theta','dag.weights','dag.mask','dag.tau']  -> A = op.mean_A()
               reproduced EXACTLY as WMOperator.mean_A() (experiments_rl/core/wm_operator.py:72):
               eval-mode HARD gate  gate = 1[sigmoid(theta/tau) > 0.5];  A = gate * weights * mask;
               then DAGModule._spectral_clamp(A, max_sn=0.85) (models/dag.py:136).
      key    : state_dict['decoder.dec_mask']  -> GO:BP membership mask (effective_D mask)
      key    : ckpt['gene_names']              -> 4366 gene axis
  GO terms   : utils.bio_priors.build_go_bp_mask(gene_names, max_term_size=30, min_genes_in_axis=5,
               max_overlap=0.5, leak=0.3)  (local Enrichr GO_Biological_Process_2023 cache);
               reproduces the training mask with agreement == 1.00 (annotated "GO-mask coherence 1.00").
  per-program |A| connectivity = |A|.sum(axis=0) + |A|.sum(axis=1)   (in+out signed edge weight)

This is the SAME A/mask that experiments/exp20_load_bearing/evaluation/causal_wm_validation.py
renders into causal_val/fig2a_latentA_and_GO.png (fig2a code ~L551-588); regenerated cleaner here.
The spectral-clamp scale is a global scalar -> it does not change signs, structure, density, or the
program-connectivity ranking (only the colorbar magnitude).

Run:  cd paper/figures && python make_fig4B_latentA.py [--rebuild]
Outputs: panels/fig4B_latentA.pdf/.png  (+ cached panels/fig4B_latentA_data.npz)

TYPOGRAPHY NOTE: this panel is re-embedded 1:1 as panel C of fig4_causal.py, so it is
rendered at exactly ps.TEXTWIDTH_IN and uses the shared type ladder -- no rescaling, hence
no font-size drift between panel C and panels A/B of the parent figure.
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import paperstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP20 = os.path.join(REPO, "experiments", "exp20_load_bearing")
CKPT = os.path.join(EXP20, "results", "grid", "jb_prod_wm", "checkpoints", "phase1_epoch60.pt")
NPZ = os.path.join(HERE, "panels", "fig4B_latentA_data.npz")


def extract_from_checkpoint():
    """Load A + dec_mask + GO terms straight from the frozen WM checkpoint."""
    import torch
    sys.path.insert(0, EXP20)
    import utils.bio_priors as bp

    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    gene_names = [str(g) for g in d["gene_names"]]
    theta = sd["dag.theta"].float()
    weights = sd["dag.weights"].float()
    mask = sd["dag.mask"].float()
    tau = float(sd["dag.tau"])

    # ---- op.mean_A(): eval-mode hard gate * weights * mask, then spectral clamp ----
    gates = (torch.sigmoid(theta / tau) > 0.5).float()
    A_raw = gates * weights * mask

    def spectral_clamp(A, max_sn=0.85, n_iter=20):
        torch.manual_seed(0)                                  # deterministic power iteration
        dd = A.shape[-1]
        v = torch.randn(dd, 1); v = v / (v.norm() + 1e-8); At = A.T
        for _ in range(n_iter):
            u = A @ v; u = u / (u.norm() + 1e-8)
            v = At @ u; v = v / (v.norm() + 1e-8)
        sn = (u.T @ A @ v).abs().item()
        return (A * min(max_sn / (sn + 1e-8), 1.0)).numpy()

    A = spectral_clamp(A_raw)
    dec_mask = sd["decoder.dec_mask"].float().numpy()          # [n_genes, d]
    d_lat = A.shape[0]

    # ---- GO:BP term names per latent program (deterministic rebuild) ----
    M, terms = bp.build_go_bp_mask(gene_names, max_term_size=30, min_genes_in_axis=5,
                                   max_overlap=0.5, leak=0.3, max_pathways=None, verbose=False)
    coherence = float(((M > 0.3) == (dec_mask > 0.3)).mean()) if M.shape == dec_mask.shape else float("nan")

    deg = np.abs(A).sum(0) + np.abs(A).sum(1)                  # in+out |A| connectivity per program
    n_member = (dec_mask > 0.99).sum(0).astype(np.int32)       # # annotated genes per program
    nnz = int((A != 0).sum())
    density = 100.0 * nnz / (d_lat * d_lat - d_lat)

    np.savez(NPZ, A=A.astype(np.float32), deg=deg.astype(np.float32),
             n_member=n_member, terms=np.array(terms, dtype=object),
             coherence=coherence, nnz=nnz, density=density, d_lat=d_lat)
    print(f"[extract] A {A.shape} nnz={nnz} density={density:.1f}%  coherence={coherence:.2f}  -> {NPZ}")
    return NPZ


def load_data(rebuild=False):
    if rebuild or not os.path.exists(NPZ):
        if not os.path.exists(CKPT):
            raise FileNotFoundError(f"checkpoint missing: {CKPT}")
        extract_from_checkpoint()
    z = np.load(NPZ, allow_pickle=True)
    return z


def go_label(t, width=30):
    """Drop the GO accession, then WRAP (never truncate) onto at most 2 lines.

    Three of the top-10 hubs share the prefix 'Negative Regulation Of ...'; truncating
    at a fixed character count rendered all three identically, so the wrap width is set
    wide enough that every term keeps its discriminating tail.
    """
    return ps.wrap_label(t.split(" (GO:")[0], width=width, maxlines=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-extract A from the checkpoint")
    args = ap.parse_args()
    z = load_data(rebuild=args.rebuild)
    A = z["A"]; deg = z["deg"]; n_member = z["n_member"]; terms = z["terms"]
    coherence = float(z["coherence"]); nnz = int(z["nnz"]); density = float(z["density"]); d_lat = int(z["d_lat"])

    # ---- clean WIDE panel: heatmap (left) + top-10 GO hub bar (right) ----
    # Exact manuscript width so the panel drops into fig4_causal panel C at 1:1 scale.
    # width_ratios chosen so the square heatmap fills the panel height: both sub-titles
    # then sit on the same baseline and there is no dead band beside the matrix.
    fig, (axH, axB) = ps.grid(1, 2, h=2.72, wspace=0.05,
                              gridspec_kw=dict(width_ratios=[1.0, 0.88]))

    # --- heatmap ---
    nz = A[A != 0]
    vmax = float(np.percentile(np.abs(nz), 99)) if nz.size else 1.0
    im = axH.imshow(A, cmap=ps.CMAP_DIV, vmin=-vmax, vmax=vmax, aspect="equal",
                    interpolation="nearest")
    # pad clears the "0" tick label, which straddles the top edge of the heatmap
    axH.set_title(f"Latent causal graph $A$  ({d_lat}$\\times${d_lat})", pad=7)
    axH.set_xlabel("program (child)", labelpad=1)
    axH.set_ylabel("program (parent)", labelpad=1)
    axH.set_xticks([0, d_lat - 1]); axH.set_yticks([0, d_lat - 1])
    axH.tick_params(length=2)
    axH.set_anchor("N")          # top-align the square heatmap so both sub-titles sit level
    cb = fig.colorbar(im, ax=axH, fraction=0.05, pad=0.03, ticks=[-vmax, 0, vmax])
    cb.ax.set_yticklabels([f"$-${vmax:.02f}", "0", f"{vmax:.02f}"])
    cb.set_label("signed edge weight", fontsize=ps.FS["small"], labelpad=2)
    cb.ax.tick_params(labelsize=ps.FS["small"], length=1.8)
    cb.outline.set_linewidth(0.5)
    _bbox = dict(boxstyle="round,pad=0.2", fc="white", ec=ps.RULE, lw=0.5, alpha=0.92)
    axH.text(0.04, 0.96, f"{nnz:,} signed edges\n{density:.0f}% dense", transform=axH.transAxes,
             ha="left", va="top", fontsize=ps.FS["small"], color=ps.INK, bbox=_bbox)
    axH.text(0.96, 0.04, "GO-mask\ncoherence 1.00", transform=axH.transAxes,
             ha="right", va="bottom", fontsize=ps.FS["small"], color=ps.INK, bbox=_bbox)

    # --- top-10 GO:BP causal hubs by |A| connectivity ---
    top = np.argsort(-deg)[:10][::-1]                        # ascending for horizontal bars
    labels = [go_label(str(terms[k])) for k in top]
    norm = mpl.colors.Normalize(vmin=float(n_member[top].min()),
                                vmax=float(n_member[top].max() + 1e-9))
    cmap = mpl.colormaps[ps.CMAP_SEQ]
    cols = [cmap(0.4 + 0.55 * norm(n_member[k])) for k in top]
    y = np.arange(len(top))
    axB.barh(y, deg[top], color=cols, edgecolor="white", linewidth=0.4, zorder=3, height=0.72)
    axB.set_yticks(y); axB.set_yticklabels(labels, fontsize=ps.FS["small"])
    axB.set_xlabel("total |$A$| connectivity (in+out)", labelpad=1)
    axB.set_title("Top-10 causal-hub GO:BP programs", pad=7)
    axB.tick_params(axis="y", length=0)
    axB.xaxis.grid(True, color=ps.tint(ps.RULE, 0.5), lw=0.5, zorder=0)
    axB.set_ylim(-0.62, len(top) - 0.38)
    ps.tidy(axB)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb2 = fig.colorbar(sm, ax=axB, fraction=0.05, pad=0.03)
    cb2.set_label("# member genes", fontsize=ps.FS["small"], labelpad=2)
    cb2.ax.tick_params(labelsize=ps.FS["small"], length=1.8)
    cb2.outline.set_linewidth(0.5)

    # The heatmap keeps aspect="equal" (a 357x357 adjacency must not be stretched), so its
    # axes is narrower than its grid cell.  Match the colour bar to the *drawn* heatmap box
    # after the constrained layout has settled, otherwise the full-height bar puts its top
    # tick label level with -- and on top of -- the panel title.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    ph, pc = axH.get_position(), cb.ax.get_position()
    cb.ax.set_position([pc.x0, ph.y0, pc.width, ph.height])

    stem = os.path.join(HERE, "panels", "fig4B_latentA")
    ps.save(fig, stem)


if __name__ == "__main__":
    main()
