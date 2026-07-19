"""Interpretability + load-bearing evals for the SENA-δ encoder runs.

These are the NEW evals requested alongside the SENA architecture. They read the
LATENT structure directly (pathway activities α and the latent DAG A) — NOT the
decoder-projected G=D·A·Dᵀ, which we showed memorizes the prior. CollecTRI stays
strictly eval-only (handled by grn_dag_sweep); nothing here uses it as a target.

Three evals (after de la Fuente et al., ICLR 2025, §4.3 + Appendix I):

  (1) Pathway differential activation — Hits@N + DAR.
      For each overexpressed TF g that belongs to ≥1 GO:BP pathway, does perturbing g
      differentially activate ITS OWN pathways more than the rest?  α = encoder
      pathway activity; DA_k = |mean_pert α_k − mean_ctrl α_k|.
        Hits@N = frac of g's pathways ranked in the top-N by DA.
        DAR    = (|W̄|·Σ_{k∈W} DA_k) / (|W|·Σ_{k∈W̄} DA_k)   (>1 ⇒ on-target).
      A model whose pathway layer is biologically meaningful scores Hits@N≫chance, DAR>1.

  (2) Program → pathway map (interpretation of the latent factors).
      μ_j = α·δ_j (mu_head row j over pathways) ⇒ program j's top pathways are the
      GO:BP terms with largest |δ_j|. This NAMES each latent SCM node — the
      interpretable read-out that replaces D·A·Dᵀ.

  (3) Latent-A read-out + intervention faithfulness.
      Read A directly (sparsity, spectral radius). For each TF, which programs does the
      shift do-operator drive (|Δz| via compute_shift→(I−Aᵀ)⁻¹)? Cross-reference the
      top driven program's pathway labels — a sanity read on whether interventions route
      through named, on-target programs.

Usage:
  python analysis/sena_interp_eval.py \
      --checkpoint <exp19>/results/grid/grnfix_sena/checkpoints/phase1_epoch60.pt \
      [--joung <h5ad>] [--top_n 100] [--n_cells 4000] [--out <dir>]
"""
import os, sys, argparse, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')

EXP19 = '/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose'
sys.path.insert(0, EXP19)
JOUNG = f'{EXP19}/../exp17_joung_tfatlas/data/joung_singles_proc.h5ad'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--joung', default=JOUNG)
    ap.add_argument('--top_n', type=int, default=100)
    ap.add_argument('--n_cells', type=int, default=4000, help='max cells sampled per group')
    ap.add_argument('--top_pathways', type=int, default=8, help='pathways per program to report')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), '..', 'sena_interp')
    os.makedirs(out, exist_ok=True)

    from evaluation.run_shift_eval_norman import load_shift_model
    from utils.bio_priors import build_go_bp_mask
    model, gene_names, ckpt = load_shift_model(args.checkpoint, 'cpu')
    if getattr(model, 'encoder_type', 'mlp') != 'sena':
        raise SystemExit("checkpoint is not a SENA encoder run")
    g2i = {str(g).upper(): i for i, g in enumerate(gene_names)}
    cfg_m = (ckpt.get('config', {}) or {}).get('model', {})

    # Rebuild GO:BP term NAMES deterministically with the SAME params used in training
    # (the checkpoint stores the mask M but not the term strings). Verify the membership
    # mask matches the restored buffer so the names line up with columns.
    M_named, terms = build_go_bp_mask(
        gene_names,
        max_term_size=int(cfg_m.get('sena_max_term_size', 30)),
        min_genes_in_axis=int(cfg_m.get('sena_min_genes', 5)),
        max_overlap=float(cfg_m.get('sena_max_overlap', 0.5)),
        leak=float(cfg_m.get('sena_leak', 0.1)), verbose=False)
    M = model.encoder.M.numpy()
    K = M.shape[1]
    assert len(terms) == K, f"rebuilt K {len(terms)} != model K {K}"
    member = M >= 0.999                                   # [n_genes, K] hard membership
    leak = float(cfg_m.get('sena_leak', 0.1))
    if not np.allclose(M_named, M):
        print("[warn] rebuilt mask != stored mask; term names may be misaligned")

    # ── load Joung, sample control + per-TF perturbed cells ──
    import anndata as ad, scipy.sparse as sp
    A = ad.read_h5ad(args.joung)
    Xall = A.X
    is_ctrl = A.obs['is_control'].astype(str).values == 'True'
    tfg = A.obs['TF_gene'].astype(str).str.upper().values
    rng = np.random.default_rng(0)

    def dense(rows):
        x = Xall[rows]
        return (x.toarray() if sp.issparse(x) else np.asarray(x)).astype(np.float32)

    def alpha_of(rows):
        if len(rows) > args.n_cells:
            rows = rng.choice(rows, args.n_cells, replace=False)
        xb = torch.from_numpy(dense(np.sort(rows)))
        return model.encoder.pathway_activity(xb).numpy().mean(0)     # [K]

    ctrl_rows = np.where(is_ctrl)[0]
    a_ctrl = alpha_of(ctrl_rows)

    # ── (1) Hits@N + DAR over TFs that are in ≥1 pathway and present as a perturbation ──
    rows1 = []
    perturbed_tfs = [t for t in pd.unique(tfg[~is_ctrl]) if t in g2i]
    for tf in perturbed_tfs:
        gi = g2i[tf]
        W = np.where(member[gi])[0]                       # pathways containing this TF
        if W.size == 0:
            continue
        rows = np.where((~is_ctrl) & (tfg == tf))[0]
        if rows.size < 20:
            continue
        DA = np.abs(alpha_of(rows) - a_ctrl)              # [K]
        rank = (-DA).argsort().argsort()                  # 0 = largest DA
        hits = float(np.mean(rank[W] < args.top_n))
        Wbar = np.setdiff1d(np.arange(K), W)
        denom = (DA[Wbar].sum() / max(Wbar.size, 1))
        dar = float((DA[W].sum() / W.size) / denom) if denom > 0 else np.nan
        rows1.append(dict(TF=tf, n_pathways=int(W.size), n_cells=int(rows.size),
                          hits_at_n=hits, DAR=dar))
    df1 = pd.DataFrame(rows1).sort_values('DAR', ascending=False)
    df1.to_csv(f'{out}/sena_pathway_DA.csv', index=False)

    # ── (2) program → top pathways (μ head over pathway activities) ──
    Wmu = model.encoder.mu_head.weight.numpy()            # [d, K]
    rows2 = []
    for j in range(Wmu.shape[0]):
        top = np.abs(Wmu[j]).argsort()[::-1][:args.top_pathways]
        rows2.append(dict(program=j,
                          top_pathways='; '.join(terms[k] for k in top),
                          top_weights='; '.join(f'{Wmu[j,k]:+.2f}' for k in top)))
    df2 = pd.DataFrame(rows2)
    df2.to_csv(f'{out}/sena_program_pathways.csv', index=False)

    # ── (3) latent A read-out + per-TF driven program (intervention faithfulness) ──
    Asamp = model.dag.sample_A(n_samples=1).squeeze(0).numpy()
    d = Asamp.shape[0]; off = ~np.eye(d, dtype=bool)
    a_dens = float((np.abs(Asamp[off]) > 1e-3).mean())
    a_spec = float(np.abs(np.linalg.eigvals(Asamp)).max())
    program_top = {j: terms[np.abs(Wmu[j]).argmax()] for j in range(d)}
    rows3 = []
    if model.use_shift_encoder and model._pert_label_set:
        I = torch.eye(d)
        At = torch.from_numpy(Asamp).float()
        for tf in perturbed_tfs:
            S = model.compute_shift(torch.tensor([g2i[tf]]))[0]       # [d]
            from models.dag import _robust_solve
            dz = _robust_solve(I - At.T, S).numpy()                   # propagated Δz
            jt = int(np.abs(dz).argmax())
            rows3.append(dict(TF=tf, top_program=jt,
                              top_program_pathway=program_top[jt],
                              driven_norm=float(np.linalg.norm(dz)),
                              in_own_pathway=bool(member[g2i[tf]].any()
                                                  and member[g2i[tf], jt])))
    df3 = pd.DataFrame(rows3)
    df3.to_csv(f'{out}/sena_intervention_targets.csv', index=False)

    # ── summary ──
    print("=" * 70)
    print(f"SENA INTERP EVAL — {os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))}")
    print(f"  K pathways={K}  latent d={d}  leak={leak}")
    if len(df1):
        print(f"  (1) pathway DA over {len(df1)} in-pathway TFs:")
        print(f"      Hits@{args.top_n}: mean={df1.hits_at_n.mean():.3f} "
              f"(chance≈{args.top_n/K:.3f})   DAR: median={df1.DAR.median():.2f} "
              f"frac>1={np.mean(df1.DAR>1):.2f}")
    print(f"  (2) program→pathway map: {len(df2)} programs named -> sena_program_pathways.csv")
    print(f"  (3) latent A: density={a_dens:.3f} spectral={a_spec:.3f}")
    if len(df3):
        print(f"      intervention faithfulness: {np.mean(df3.in_own_pathway):.2f} of TFs "
              f"drive a program whose top pathway contains the TF")
    print(f"  saved -> {out}/")


if __name__ == '__main__':
    main()
