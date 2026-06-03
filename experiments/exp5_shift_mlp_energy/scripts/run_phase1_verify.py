"""
Single-cell Phase-1 verification — run before launching the 8-cell grid.

Trains one grid cell at the recommended verification hyperparameters
(lambda_sparse=0.01, lambda_prior=1.0, margin=0.5, lambda_D=0.001) with
MSE + N=1 on 50k K562 cells for 80 epochs, then prints a pass/fail
acceptance summary against the criteria in docs/architecture_u_score.md §5.8.

Usage:
    python -u scripts/run_phase1_verify.py
"""
import os
import sys
import csv
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

# ── 1) load and override config ──────────────────────────────────────────
with open('configs/replogle_u_score.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['phase1']['lambda_sparse']   = 0.01
cfg['phase1']['lambda_prior']    = 1.0
cfg['phase1']['prior_margin']    = 0.5
cfg['phase1']['lambda_D_sparse'] = 0.001
cfg['phase1']['n_epochs']        = 80
cfg.setdefault('intervention', {})['scheme'] = 'u_score'
cfg.setdefault('wandb', {})['enabled'] = False

# ── 2) per-run output dir (mirrors grid_driver layout) ───────────────────
RUN_DIR = 'results/verify/cell_ls0p01_lp1p0_m0p5_ld0p001'
os.makedirs(f'{RUN_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{RUN_DIR}/logs', exist_ok=True)

cwd0 = os.getcwd()
os.chdir(RUN_DIR)
try:
    from training.train_vae import train_vae
    from evaluation.dag_diagnostics import (
        analyze_dag, analyze_orthogonality, analyze_mutilation, analyze_u_sparsity,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    print(f'Hyperparameters:')
    for k in ['lambda_sparse', 'lambda_prior', 'prior_margin', 'lambda_D_sparse',
              'pert_loss_kind', 'pert_n_dag_samples', 'cell_sentence_S',
              'min_block_cells', 'n_epochs']:
        print(f'  {k:25s} = {cfg["phase1"][k]}')
    print(f'  intervention.scheme        = {cfg["intervention"]["scheme"]}')
    print(f'  replogle_max_cells         = {cfg["data"]["replogle_max_cells"]}')
    print()

    t0 = time.time()
    model, data_dict = train_vae(cfg, device=device)
    elapsed = time.time() - t0
    print(f'\n=== TRAINING COMPLETE in {elapsed/60:.1f} minutes ===\n')

    # ── 3) diagnostics ────────────────────────────────────────────────────
    pert_split = data_dict.get('pert_split')
    if pert_split and isinstance(pert_split.get('train'), list):
        pert_split = {k: set(v) for k, v in pert_split.items()}

    dag_s = analyze_dag(model, '.')
    ortho_s = analyze_orthogonality(model, '.')
    mut_s = analyze_mutilation(model, data_dict, pert_split, '.',
                                n_dag_samples=8, threshold_frac=0.1, device=device)
    u_s = analyze_u_sparsity(model, data_dict, pert_split, '.', device=device)

    # Read recon trajectory from CSV log
    recon_a = recon_final = float('nan')
    with open('logs/phase1_training.csv') as f:
        rows = list(csv.DictReader(f))
    if rows:
        recon_final = float(rows[-1].get('recon', 'nan'))
        phase_a_end = cfg['phase1']['reconstruction_phase_epochs']
        for r in rows:
            if int(r.get('epoch', -1)) == phase_a_end:
                recon_a = float(r.get('recon', 'nan'))
                break

    # ── Factored-A statistics (the actual learned signed adjacency) ──────
    # Phase I (factored DAG): A = gate × weight × mask.  The gates (theta)
    # may not move much because Gumbel noise dominates; the WEIGHTS are
    # what carry the structural signal.  Acceptance below is based on the
    # signed A entries directly.
    import torch as _torch
    with _torch.no_grad():
        weights_t = model.dag.weights.detach()
        # Init was Xavier U(-r, r) with r = sqrt(6/(d+d)) = sqrt(3/d).
        # The expected init mean(|w|) is r/2 ≈ sqrt(3/(4d)).
        d_ = model.latent_dim
        expected_init_mean_abs_w = (3.0 / (4.0 * d_)) ** 0.5
        weights_mean_abs = weights_t.abs().mean().item()
        weights_movement = abs(weights_mean_abs - expected_init_mean_abs_w)
        # Sample one deterministic A (gate_hard * weight) and look at signed structure
        signed_A = model.dag.get_signed_edges().detach()  # [d, d]
        absA = signed_A.abs()
        edges_absA_gt_0p05 = int((absA > 0.05).sum().item())
        edges_absA_gt_0p10 = int((absA > 0.10).sum().item())
        absA_mean = absA.mean().item()
        absA_max = absA.max().item()

    metrics = {
        'theta_mean':              dag_s['theta_mean'],
        # gate-only stats (sigmoid(theta/tau)): less informative under factored A
        'gate_prob_mean':          dag_s['mean_edge_prob'],
        'gates_gt_0.5':            dag_s['p > 0.5'],
        # weight-based stats — the actual learned signed adjacency
        'weights_mean_abs':        weights_mean_abs,
        'weights_movement_from_init': weights_movement,
        'absA_mean':               absA_mean,
        'absA_max':                absA_max,
        'edges_absA_gt_0.05':      edges_absA_gt_0p05,
        'edges_absA_gt_0.10':      edges_absA_gt_0p10,
        'recon_phase_a_final':     recon_a,
        'recon_active_final':      recon_final,
        'recon_blowup_ratio':      recon_final / recon_a if recon_a > 0 else float('nan'),
        'ortho_off_diag_mean_abs': ortho_s['off_diag_mean_abs'],
        'mean_programs_per_gene':  ortho_s.get('mean_programs_per_gene', float('nan')),
        'mean_n_affected_threshold': mut_s.get('mean_n_affected', float('nan')),
    }

    # ── 4) acceptance check ───────────────────────────────────────────────
    # Phase I (factored A = gate × weight): the structural signal is in WEIGHTS,
    # not theta.  Acceptance is "weights moved appreciably off Xavier init AND
    # there's signed-A structure" (some edges meaningfully > 0).
    print('═══════════════════ ACCEPTANCE CHECK ═══════════════════')
    checks = {
        'weights moved ≥ 0.01 from init':
            metrics['weights_movement_from_init'] >= 0.01,
        'edges_|A|>0.10 ≥ 100  (signed structure):':
            metrics['edges_absA_gt_0.10'] >= 100,
        'recon_blowup_ratio ≤ 2.0':         metrics['recon_blowup_ratio'] <= 2.0,
        'mean_programs_per_gene ≤ 15':      metrics['mean_programs_per_gene'] <= 15,
        'ortho_off_diag_mean_abs ≤ 0.05':   metrics['ortho_off_diag_mean_abs'] <= 0.05,
    }
    for name, passed in checks.items():
        flag = '✓ PASS' if passed else '✗ FAIL'
        print(f'  {flag}  {name}')

    print()
    print('─── observed values ───')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k:35s} = {v:.4f}')
        else:
            print(f'  {k:35s} = {v}')

    overall = all(checks.values())
    print()
    if overall:
        print('▶▶▶  VERIFICATION PASSED — ok to submit the 8-cell grid')
        print('     sbatch scripts/train_phase1_grid.sbatch')
    else:
        n_fail = sum(1 for v in checks.values() if not v)
        print(f'▶▶▶  VERIFICATION FAILED ({n_fail}/5 checks failed) — DO NOT submit the grid')
        print('     debug the failing axis before running the sweep.')

    json.dump(metrics, open('verify_metrics.json', 'w'), indent=2)
finally:
    os.chdir(cwd0)

print(f'\nArtifacts: {RUN_DIR}/')
print(f'  checkpoints/phase1_best.pt')
print(f'  logs/phase1_training.csv')
print(f'  verify_metrics.json')
