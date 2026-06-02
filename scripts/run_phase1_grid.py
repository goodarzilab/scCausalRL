"""
Phase 1 grid search on a Replogle K562 subset.

Goal: figure out which (lambda_sparse, lambda_prior, prior_margin) combination
lets the DAG escape its `theta ≈ -5` initialisation (current symptom: mean edge
prob ≈ 0.0071 ≈ sigmoid(-5)). Per the diagnostic, lambda_sparse=1.0 plus a weak
margin-ranking prior is the suspected bottleneck.

Sweep (8 runs total):
    lambda_sparse              ∈ {1.0, 0.1, 0.01, 0.0}
    (lambda_prior, margin)     ∈ {(0.1, 0.1), (1.0, 0.5)}

Each run trains for `--n_epochs` (default 40) on `replogle_max_cells_grid`
cells (default 5000) and writes a single row of metrics to
results/grid_search/summary.csv together with its own checkpoint and DAG
diagnostics. Pick the row with the highest `edges_gt_0.1` that does not
regress recon loss by more than 20% vs. the baseline (lambda_sparse=1.0,
lambda_prior=0.1, prior_margin=0.1).

Usage:
    python scripts/run_phase1_grid.py \
        --base_config configs/replogle.yaml \
        --output_dir results/grid_search \
        --n_epochs 40
"""
import os
import sys
import copy
import json
import argparse
import csv
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


GRID_LAMBDA_SPARSE = [1.0, 10.0]                         # Phase J: previous {0.01, 0.1} was 10-100× too weak.
                                                          # Loss contribution = lambda × A.abs().mean() ≈ 0.1×0.04 = 0.004 vs recon=0.54.
GRID_PRIOR = [(10.0, 0.5), (50.0, 0.5)]                  # Phase J: previous {1, 10} grid showed clear monotone effect — push higher.
GRID_LAMBDA_D_SPARSE = [1.0, 10.0]                       # Phase J: previous {0.01, 0.1} barely moved D_l1 (0.038→0.037).
                                                          # MultiOmics used 1000× larger; 1-10 is conservative scale-up.
# pert_loss_kind axis dropped in Phase F — MSE is the default; energy/mmd kept as eval-time options.


def make_run_id(lam_sparse, lam_prior, margin, lam_d_sparse):
    return (
        f"ls{lam_sparse}_lp{lam_prior}_m{margin}_ld{lam_d_sparse}"
        .replace('.', 'p')
    )


def run_one(config, lam_sparse, lam_prior, margin, lam_d_sparse,
            n_epochs, max_cells, run_id, out_dir, device,
            lam_sparse_mode='product'):
    """Train one grid cell and collect summary diagnostics into a flat dict."""
    cfg = copy.deepcopy(config)

    # ── overrides ──
    if max_cells is not None:
        cfg['data']['replogle_max_cells'] = max_cells
    cfg['phase1']['n_epochs'] = n_epochs
    cfg['phase1']['lambda_sparse'] = float(lam_sparse)
    cfg['phase1']['lambda_sparse_mode'] = str(lam_sparse_mode)
    cfg['phase1']['lambda_prior'] = float(lam_prior)
    cfg['phase1']['prior_margin'] = float(margin)
    cfg['phase1']['lambda_D_sparse'] = float(lam_d_sparse)
    # Phase F: pert_loss_kind is now fixed at 'mse' (read from the config — usually MSE).
    # All grid runs are on the new u-score / cell-sentence pipeline.
    cfg.setdefault('intervention', {})['scheme'] = 'u_score'
    # Force the bias-toward-discovery: re-enable DAGMA escalation if it was off.
    # (replogle_no_dag_constraint had rho=0; defaults from replogle.yaml are fine.)

    # Disable wandb for grid runs (too many runs to clutter the project)
    cfg.setdefault('wandb', {})['enabled'] = False
    cfg['wandb']['phase1_run_name'] = f'grid_{run_id}'

    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Redirect checkpoint location for this run by chdir'ing into a per-run dir.
    # train_vae writes to checkpoints/phase1_best.pt relative to CWD; we copy after.
    cwd0 = os.getcwd()
    os.chdir(run_dir)
    try:
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        from training.train_vae import train_vae
        from evaluation.dag_diagnostics import (
            analyze_dag, analyze_orthogonality, analyze_mutilation, analyze_u_sparsity,
        )

        t0 = time.time()
        model, data_dict = train_vae(cfg, device=device)
        elapsed = time.time() - t0

        # Diagnostics on the trained model
        pert_split = data_dict.get('pert_split', None)
        if pert_split is not None and isinstance(pert_split.get('train', None), set):
            pass  # already sets
        elif pert_split is not None:
            pert_split = {k: set(v) for k, v in pert_split.items()}

        dag_summary = analyze_dag(model, run_dir)
        ortho_summary = analyze_orthogonality(model, run_dir)
        mut_summary = analyze_mutilation(
            model, data_dict, pert_split, run_dir,
            n_dag_samples=8, threshold_frac=0.1, device=device,
        )
        u_summary = analyze_u_sparsity(model, data_dict, pert_split, run_dir, device=device)

        # Read training trajectory from the per-run CSV log
        recon_final = pert_final = ortho_final = float('nan')
        recon_phase_a_final = recon_active_final = float('nan')
        log_path = 'logs/phase1_training.csv'
        if os.path.isfile(log_path):
            with open(log_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                last = rows[-1]
                recon_final = float(last.get('recon', 'nan'))
                pert_final = float(last.get('pert', 'nan'))
                ortho_final = float(last.get('ortho', 'nan'))
                # Recon at the end of Phase A (epoch == reconstruction_phase_epochs):
                # used to detect the recon-explosion failure mode from job 2349145.
                phase_a_end = cfg['phase1']['reconstruction_phase_epochs']
                for r in rows:
                    if int(r.get('epoch', -1)) == phase_a_end:
                        recon_phase_a_final = float(r.get('recon', 'nan'))
                        break
                # Recon at the very end (DAG-active) — same as recon_final for default schedules.
                recon_active_final = recon_final
    finally:
        os.chdir(cwd0)

    summary = {
        'run_id': run_id,
        'lambda_sparse': lam_sparse,
        'lambda_sparse_mode': lam_sparse_mode,
        'lambda_prior': lam_prior,
        'prior_margin': margin,
        'lambda_D_sparse': lam_d_sparse,
        'pert_loss_kind': cfg['phase1'].get('pert_loss_kind', 'mse'),
        'n_epochs': n_epochs,
        'max_cells': max_cells,
        'elapsed_sec': elapsed,
        'mean_edge_prob': dag_summary['mean_edge_prob'],
        'max_edge_prob': dag_summary['max_edge_prob'],
        'theta_mean': dag_summary['theta_mean'],
        'theta_std': dag_summary['theta_std'],
        'edges_gt_0.1': dag_summary['p > 0.1'],
        'edges_gt_0.3': dag_summary['p > 0.3'],
        'edges_gt_0.5': dag_summary['p > 0.5'],
        'ortho_off_diag_mean_abs': ortho_summary['off_diag_mean_abs'],
        'ortho_col_norm_cv': ortho_summary['col_norm_cv'],
        'D_l1': ortho_summary.get('D_l1', float('nan')),
        'mean_programs_per_gene': ortho_summary.get('mean_programs_per_gene', float('nan')),
        'mean_genes_per_program': ortho_summary.get('mean_genes_per_program', float('nan')),
        'mean_n_affected_threshold': mut_summary.get('mean_n_affected', float('nan')),
        'mean_l0_gt_0p5_u': u_summary.get('mean_l0_gt_0.5', float('nan')),
        'mean_l1_over_l2_u': u_summary.get('mean_l1_over_l2', float('nan')),
        'recon_final': recon_final,
        'recon_phase_a_final': recon_phase_a_final,
        'recon_active_final': recon_active_final,
        'recon_blowup_ratio': (
            recon_active_final / recon_phase_a_final
            if recon_phase_a_final and recon_phase_a_final == recon_phase_a_final  # not NaN
                and recon_phase_a_final > 0 else float('nan')
        ),
        'pert_final': pert_final,
        'ortho_final': ortho_final,
    }

    # Per-run JSON dump (full)
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Phase 1 grid search (Replogle subset)')
    parser.add_argument('--base_config', type=str, default='configs/replogle.yaml')
    parser.add_argument('--output_dir', type=str, default='results/grid_search')
    parser.add_argument('--n_epochs', type=int, default=80,
                        help='Epochs per grid cell (default 80 — recon=20 + DAG warm-in=10 + 50 full)')
    parser.add_argument('--max_cells', type=int, default=None,
                        help='Optional cap on cells (default: full K562, ~186k)')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    # Use full K562 (~186k cells) by default — the cell-sentence loader handles
    # per-(gene, gem_group) block sampling internally.  No grid subsample.
    max_cells = args.max_cells   # None → no override → load_replogle keeps everything

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    summary_csv = os.path.join(out_dir, 'summary.csv')
    fieldnames = [
        'run_id', 'lambda_sparse', 'lambda_prior', 'prior_margin', 'lambda_D_sparse',
        'pert_loss_kind',
        'n_epochs', 'max_cells', 'elapsed_sec',
        'mean_edge_prob', 'max_edge_prob', 'theta_mean', 'theta_std',
        'edges_gt_0.1', 'edges_gt_0.3', 'edges_gt_0.5',
        'ortho_off_diag_mean_abs', 'ortho_col_norm_cv',
        'D_l1', 'mean_programs_per_gene', 'mean_genes_per_program',
        'mean_n_affected_threshold',
        'mean_l0_gt_0p5_u', 'mean_l1_over_l2_u',
        'recon_final', 'recon_phase_a_final', 'recon_active_final', 'recon_blowup_ratio',
        'pert_final', 'ortho_final',
    ]
    # Write header if file does not exist
    if not os.path.isfile(summary_csv):
        with open(summary_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    runs = []
    for lam_sparse in GRID_LAMBDA_SPARSE:
        for (lam_prior, margin) in GRID_PRIOR:
            for lam_d_sparse in GRID_LAMBDA_D_SPARSE:
                runs.append((lam_sparse, lam_prior, margin, lam_d_sparse))

    print(f"Grid: {len(runs)} runs, {args.n_epochs} epochs each, "
          f"max_cells={'config default' if max_cells is None else max_cells}")
    print(f"  axes: lambda_sparse={GRID_LAMBDA_SPARSE}")
    print(f"        (lambda_prior,margin)={GRID_PRIOR}")
    print(f"        lambda_D_sparse={GRID_LAMBDA_D_SPARSE}")
    print(f"  fixed: pert_loss_kind=mse, pert_n_dag_samples=1 (Phase F)")
    print(f"Output dir: {out_dir}")

    for i, (ls, lp, mg, ld) in enumerate(runs, 1):
        run_id = make_run_id(ls, lp, mg, ld)
        print(f"\n[{i}/{len(runs)}] === {run_id} ===")
        try:
            summary = run_one(
                config=base_config,
                lam_sparse=ls, lam_prior=lp, margin=mg, lam_d_sparse=ld,
                n_epochs=args.n_epochs, max_cells=max_cells,
                run_id=run_id, out_dir=out_dir, device=device,
            )
        except Exception as e:
            print(f"  RUN FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

        with open(summary_csv, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(summary)

        print(f"  edges>0.1: {summary['edges_gt_0.1']}, "
              f"mean_edge_prob: {summary['mean_edge_prob']:.4f}, "
              f"recon: {summary['recon_final']:.4f}")

    print(f"\nDone. See {summary_csv}.")


if __name__ == '__main__':
    main()
