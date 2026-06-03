"""
Sweep over (lambda_sparse_mode, lambda_sparse) — 8 cells total.

Two L1 schemes for the DAG:
  - mode='product' (legacy): mean(|gates * weights|). Penalises sampled edge
    magnitudes; at training equilibrium this collapses WEIGHTS on
    topologically-confident edges (current pathology in epoch 150 of the
    promotion run — 2,568 confident edges, but mean |A_effective| ≈ 0.0003).
  - mode='gates' (new): mean(sigmoid(theta / tau)). Penalises edge EXISTENCE
    without touching weights. Hypothesis: surviving topology takes on its
    natural magnitude, the SCM solve actually propagates.

Cells:
  product × λ ∈ {0.1, 1.0, 10.0, 100.0}  — 4 cells (10.0 is the current default)
  gates   × λ ∈ {0.1, 1.0, 10.0, 100.0}  — 4 cells

Each cell runs at n_epochs=80, max_cells=50_000 (same scale as the
lambda_D_sparse mini-grid we already ran). Outputs land in
results/grid_search/<run_id>/ and rows are appended to
results/grid_search/summary.csv.

Run IDs use the convention `lsmode{mode}_ls{val}` (everything else pinned to
the Phase J winner: lambda_prior=10, margin=0.5, lambda_D_sparse=30).

Usage:
    python scripts/run_phase1_sweep_lsparse_mode.py --device cuda
"""
import os
import sys
import csv
import argparse

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_phase1_grid import run_one


def main():
    parser = argparse.ArgumentParser(description='Sweep lambda_sparse mode × weight')
    parser.add_argument('--base_config', type=str, default='configs/replogle_u_score.yaml')
    parser.add_argument('--output_dir', type=str, default='results/grid_search')
    parser.add_argument('--n_epochs', type=int, default=80)
    parser.add_argument('--max_cells', type=int, default=50000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--cell_index', type=int, default=None,
                        help="If set, run only the cell with this index (0..7) "
                             "and exit. Used by slurm-array sbatch to run cells "
                             "in parallel across GPUs.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    # Pinned (Phase J winner)
    PIN_PRIOR_W = 10.0
    PIN_MARGIN = 0.5
    PIN_LAMBDA_D = 30.0

    # Sweep axes
    MODES = ['product', 'gates']
    LAMBDAS = [0.1, 1.0, 10.0, 100.0]

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    # When running in parallel via slurm array, multiple cells writing to the
    # shared summary.csv would race — instead each cell writes its own row file
    # `<run_dir>/summary_row.csv` and a separate merge step (or shell command)
    # concatenates them after the array finishes. Sequential mode (no
    # cell_index) keeps appending to the shared file as before.
    summary_csv = os.path.join(out_dir, 'summary.csv')

    fieldnames = [
        'run_id', 'lambda_sparse', 'lambda_sparse_mode',
        'lambda_prior', 'prior_margin', 'lambda_D_sparse',
        'pert_loss_kind',
        'n_epochs', 'max_cells', 'elapsed_sec',
        'mean_edge_prob', 'max_edge_prob', 'theta_mean', 'theta_std',
        'edges_gt_0.1', 'edges_gt_0.3', 'edges_gt_0.5',
        'ortho_off_diag_mean_abs', 'ortho_col_norm_cv',
        'D_l1', 'mean_programs_per_gene', 'mean_genes_per_program',
        'mean_n_affected_threshold',
        'mean_l0_gt_0p5_u', 'mean_l1_over_l2_u',
        'recon_final', 'recon_phase_a_final', 'recon_active_final',
        'recon_blowup_ratio', 'pert_final', 'ortho_final',
    ]
    # Only initialise the shared summary.csv when running sequentially.
    # Slurm-array cells write per-run row files to avoid races.
    if args.cell_index is None and not os.path.isfile(summary_csv):
        with open(summary_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    cells_all = [(m, lam) for m in MODES for lam in LAMBDAS]
    if args.cell_index is not None:
        if not (0 <= args.cell_index < len(cells_all)):
            raise SystemExit(f"--cell_index must be in [0, {len(cells_all) - 1}], "
                             f"got {args.cell_index}")
        cells = [cells_all[args.cell_index]]
        start_idx = args.cell_index + 1
        print(f"\n=== lambda_sparse mode × weight sweep — running only cell "
              f"{args.cell_index} ({cells[0][0]}, lambda={cells[0][1]}) ===")
    else:
        cells = cells_all
        start_idx = 1
        print(f"\n=== lambda_sparse mode × weight sweep — {len(cells)} cells ===")
    print(f"  pinned: lambda_prior={PIN_PRIOR_W}, margin={PIN_MARGIN}, lambda_D_sparse={PIN_LAMBDA_D}")
    print(f"  n_epochs={args.n_epochs}, max_cells={args.max_cells}")
    print(f"  output: {out_dir}")
    for i, (mode, lam) in enumerate(cells_all):
        marker = " ← running" if args.cell_index == i or args.cell_index is None else ""
        print(f"    [{i}] mode={mode:>7s}  lambda_sparse={lam}{marker}")
    print()

    for i, (mode, lam) in enumerate(cells, start_idx):
        # Slug-safe run_id: lsmode{mode}_ls{val}_lp{pri}_m{mg}_ld{lamD}
        lam_slug = str(lam).replace('.', 'p')
        run_id = (
            f"lsmode_{mode}_ls{lam_slug}"
            f"_lp{str(PIN_PRIOR_W).replace('.', 'p')}"
            f"_m{str(PIN_MARGIN).replace('.', 'p')}"
            f"_ld{str(PIN_LAMBDA_D).replace('.', 'p')}"
        )
        print(f"\n[{i}/{len(cells)}] === {run_id} ===")
        try:
            summary = run_one(
                config=base_config,
                lam_sparse=lam,
                lam_prior=PIN_PRIOR_W,
                margin=PIN_MARGIN,
                lam_d_sparse=PIN_LAMBDA_D,
                n_epochs=args.n_epochs,
                max_cells=args.max_cells,
                run_id=run_id,
                out_dir=out_dir,
                device=device,
                lam_sparse_mode=mode,
            )
        except Exception as e:
            print(f"  RUN FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

        if args.cell_index is not None:
            # Parallel mode: write to per-run row file (no shared-file race)
            row_path = os.path.join(out_dir, summary['run_id'], 'summary_row.csv')
            with open(row_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(summary)
        else:
            # Sequential mode: legacy shared-file append (safe because one writer)
            with open(summary_csv, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(summary)

        print(f"  edges>0.5: {summary['edges_gt_0.5']}, "
              f"theta_std: {summary['theta_std']:.3f}, "
              f"programs/gene: {summary['mean_programs_per_gene']:.1f}, "
              f"recon: {summary['recon_final']:.4f}, "
              f"pert: {summary['pert_final']:.4f}")

    print(f"\n=== Sweep complete. See {summary_csv}. ===")


if __name__ == '__main__':
    main()
