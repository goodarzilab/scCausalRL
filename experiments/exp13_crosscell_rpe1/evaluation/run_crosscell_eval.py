"""Cross-cell-line external validation (in-modality CRISPRi).

Takes a K562-trained shift checkpoint and evaluates it ZERO-SHOT on a different
Replogle cell line (RPE1 / HepG2 / Jurkat) — same CRISPRi screen, different cell
type. SAME modality → no sign flip, so AUROC AND Pearson are both meaningful
(unlike the cross-modality Norman eval). Auto-detects mutilation-trained
checkpoints and evaluates with the coherent operator.

Usage:
  python evaluation/run_crosscell_eval.py --checkpoint <ckpt> --config <yaml> \
      --celline_path <h5ad> --cell_line rpe1 --output_dir <dir>
"""
import os, sys, argparse, warnings
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.run_perturbation_eval import compute_observed_de, compute_correlation_matrix, evaluate_baseline, _safe_mean, _extract
from evaluation.run_shift_eval import evaluate_single_perts_shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--config', required=True)
    ap.add_argument('--celline_path', required=True); ap.add_argument('--cell_line', default='rpe1')
    ap.add_argument('--output_dir', default='results/crosscell'); ap.add_argument('--device', default=None)
    ap.add_argument('--max_cells', type=int, default=200000); ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--max_perts', type=int, default=400, help="subsample this many held-out perts for speed (0=all)")
    args = ap.parse_args(); os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, gene_names, ckpt = load_shift_model(args.checkpoint, device)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    # mutilation-coherent eval if the ckpt was mutilation-trained
    p1 = (ckpt.get('config', {}) or {}).get('phase1', {})
    MUT = p1.get('pert_loss_mode', 'shift') == 'shift_mutilate'
    MUT_KW = dict(mutilate=MUT, mut_u_mode=p1.get('pert_mutilate_u_mode', 'loading'),
                  mut_u_max=float(p1.get('pert_mutilate_u_max', 0.5)), mut_beta=float(p1.get('pert_mutilate_beta', 1.0)))
    print(f"[crosscell {args.cell_line}] mutilation-trained={MUT}")

    from data.load_replogle_celline_aligned import load_replogle_celline_aligned
    dd = load_replogle_celline_aligned(args.celline_path, target_gene_names=gene_names,
                                       cell_line=args.cell_line, max_cells=args.max_cells)
    import scipy.sparse
    X = dd['adata'].X; X = (X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)).astype(np.float32)
    n_ext = len(dd['single_perts'])
    if n_ext == 0:
        raise RuntimeError("no evaluable single perts on the axis")
    print(f"External {args.cell_line} single perts: {n_ext}")

    de_single, _, ctrl_mean = compute_observed_de(X, dd, gene_names)
    corr = compute_correlation_matrix(X[dd['control_idx']], gene_names)

    # subsample perts for speed — 400 random perts gives a tight AUROC SEM and avoids
    # the per-pert distributional-metric cost blowing past the walltime (2164 perts → >1h).
    if args.max_perts and len(de_single) > args.max_perts:
        import random
        keep = random.Random(0).sample(sorted(de_single.keys()), args.max_perts)
        de_single = {k: de_single[k] for k in keep}
        print(f"subsampled to {len(de_single)} perts (of {n_ext}) for speed")
    n_eval = len(de_single)

    res = evaluate_single_perts_shift(model, X, dd, de_single, gene_names, device,
                                      n_ctrl_cells=args.n_ctrl_cells, c_mode='fixed', **MUT_KW)
    corr_res = evaluate_baseline(corr, gene_to_idx, de_single, gene_names)

    summary = {'cell_line': args.cell_line, 'n_external_perts': n_ext, 'n_evaluated': n_eval,
               'model_auroc': _safe_mean(_extract(res, 'auroc')),
               'model_pearson_deg': _safe_mean(_extract(res, 'pearson_deg')),
               'model_spearman': _safe_mean(_extract(res, 'spearman')),
               'corr_auroc': _safe_mean(_extract(corr_res, 'auroc')),
               'corr_spearman': _safe_mean(_extract(corr_res, 'spearman')),
               'mutilation_trained': MUT, 'checkpoint': args.checkpoint}
    pd.DataFrame(res).to_csv(os.path.join(args.output_dir, 'crosscell_model.csv'), index=False)
    pd.DataFrame(corr_res).to_csv(os.path.join(args.output_dir, 'crosscell_corr.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    print(f"\n{'='*64}\nCROSS-CELL ({args.cell_line}, in-modality CRISPRi) — n={n_ext}\n{'='*64}")
    print(f"  model  AUROC={summary['model_auroc']:.4f}  Pearson-DEG={summary['model_pearson_deg']:.4f}  Spearman={summary['model_spearman']:.4f}")
    print(f"  corr   AUROC={summary['corr_auroc']:.4f}  Spearman={summary['corr_spearman']:.4f}")
    print(f"saved → {args.output_dir}/")


if __name__ == '__main__':
    main()
