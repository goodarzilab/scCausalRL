"""
Focused 2-cell mini-grid: λ_D_sparse ∈ {30, 100} with everything else pinned
to the Phase J grid 2369390 winner (lambda_sparse=10, lambda_prior=10, margin=0.5).

The main grid showed lambda_D_sparse=10 dropped mean_programs_per_gene from 62.6
to 17.9 — close to but slightly above the ≤15 target.  This sweep tests whether
pushing lambda_D higher gets us below 15 without trashing recon.

Outputs per cell to results/grid_search/<run_id>/ (same convention as the main
grid).  Appends rows to results/grid_search/summary.csv.

Usage:
    python scripts/run_phase1_minigrid_lD.py --device cuda
"""
import os
import sys
import copy
import json
import csv
import time
import argparse

import torch
import yaml

# Reuse the main grid driver's per-cell training + diagnostics function.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_phase1_grid import run_one, make_run_id


def main():
    parser = argparse.ArgumentParser(description='Phase J mini-grid: lambda_D ∈ {30, 100}')
    parser.add_argument('--base_config', type=str, default='configs/replogle_u_score.yaml')
    parser.add_argument('--output_dir', type=str, default='results/grid_search')
    parser.add_argument('--n_epochs', type=int, default=80)
    parser.add_argument('--max_cells', type=int, default=50000,
                        help='Cells per cell — keep at 50k for speed.')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    # Pin everything except lambda_D_sparse
    PIN_LAMBDA_SPARSE = 10.0
    PIN_PRIOR_W = 10.0
    PIN_MARGIN = 0.5
    SWEEP_LAMBDA_D = [30.0, 100.0]

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
    if not os.path.isfile(summary_csv):
        with open(summary_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"Mini-grid: lambda_D_sparse ∈ {SWEEP_LAMBDA_D}")
    print(f"  pinned: lambda_sparse={PIN_LAMBDA_SPARSE}, "
          f"(lambda_prior={PIN_PRIOR_W}, margin={PIN_MARGIN})")
    print(f"  n_epochs={args.n_epochs}, max_cells={args.max_cells}")
    print(f"  output_dir={out_dir}")
    print()

    for i, lam_d in enumerate(SWEEP_LAMBDA_D, 1):
        run_id = make_run_id(PIN_LAMBDA_SPARSE, PIN_PRIOR_W, PIN_MARGIN, lam_d)
        print(f"\n[{i}/{len(SWEEP_LAMBDA_D)}] === {run_id} ===")
        try:
            summary = run_one(
                config=base_config,
                lam_sparse=PIN_LAMBDA_SPARSE,
                lam_prior=PIN_PRIOR_W,
                margin=PIN_MARGIN,
                lam_d_sparse=lam_d,
                n_epochs=args.n_epochs,
                max_cells=args.max_cells,
                run_id=run_id,
                out_dir=out_dir,
                device=device,
            )
        except Exception as e:
            print(f"  RUN FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

        with open(summary_csv, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(summary)

        print(f"  edges>0.1: {summary['edges_gt_0.1']}, "
              f"programs/gene: {summary['mean_programs_per_gene']:.1f}, "
              f"recon: {summary['recon_final']:.4f}")

    print(f"\nDone. See {summary_csv}.")


if __name__ == '__main__':
    main()
