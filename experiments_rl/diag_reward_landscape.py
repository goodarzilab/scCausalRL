"""Diagnostic: is the RL reward action-DEPENDENT, or flat (drift-saturated)?

If PPO-from-scratch collapsed because every action yields ~the same reward, then for a
FIXED target the potential Φ(action) should have near-zero spread across candidate genes,
and the true/MPC genes shouldn't stand out. We also test the fix hypothesis: does a
DRIFT-REMOVED reward (score Δ−drift instead of Δ) restore action-dependent spread?

For each of a few held-out combo targets:
  - Φ_raw(g) for ~300 random screened genes + the true combo's own genes + best-by-raw
  - Φ_resid(g) (drift-removed) for the same set
Reports the spread (std, max−mean, percentile of the true genes) under each reward.

Run (cpu_high_mem or overlap):  python diag_reward_landscape.py
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.setup_stack import add_stack_args, build_stack
from core.env import CausalLatentEnv


def main():
    ap = argparse.ArgumentParser()
    add_stack_args(ap)
    ap.add_argument('--n_rand', type=int, default=300)
    ap.add_argument('--n_targets', type=int, default=6)
    ap.add_argument('--n_dag', type=int, default=8)
    ap.add_argument('--n_ctrl', type=int, default=300)
    ap.add_argument('--output', default='rl0_mpc/results/diag')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    model, meta, op, data = build_stack(args, dev, args.n_dag)
    data.set_pool('screened')
    env = CausalLatentEnv(op, data, max_steps=1, alpha_pess=1.0, n_dag_samples=args.n_dag)
    G = len(data.gene_names)

    # drift = mean observed Δ over TRAIN singles (the shared differentiation program)
    drift = np.mean([data.de_single[g]['lfc'] for g in data.single_split['train']], axis=0)
    A = env.A_eval
    ctrl_recon = env.ctrl_recon.cpu().numpy()
    eps0 = data.eps_ctrl_mean.unsqueeze(0)

    def pred_delta(gene_idx):
        z, _, _ = op.apply(eps0, [gene_idx], A)
        return op.decode(z).squeeze(0).cpu().numpy() - ctrl_recon

    def match_on(pred, obs, sel):
        if sel.sum() < 3:
            return 0.0
        p, o = pred[sel], obs[sel]
        if p.std() < 1e-8 or o.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(p, o)[0, 1])

    rng = np.random.default_rng(0)
    targets = data.combo_split['test'][:args.n_targets]
    rows = []
    for key in targets:
        obs = data.de_combo[key]['lfc']
        true_gis = [data.gene_to_idx[g] for g in key]
        cand = list(rng.choice(data.candidate_idx, min(args.n_rand, len(data.candidate_idx)),
                               replace=False))
        for g in true_gis:
            if g not in cand:
                cand.append(g)
        thr = 0.05
        sel_raw = np.abs(obs) > thr
        obs_res = obs - drift
        sel_res = np.abs(obs_res) > thr
        phi_raw, phi_res = {}, {}
        for g in cand:
            pd_ = pred_delta(g)
            phi_raw[g] = match_on(pd_, obs, sel_raw)
            phi_res[g] = match_on(pd_ - drift, obs_res, sel_res)
        raw = np.array([phi_raw[g] for g in cand])
        res = np.array([phi_res[g] for g in cand])
        true_raw = np.mean([phi_raw[g] for g in true_gis])
        true_res = np.mean([phi_res[g] for g in true_gis])
        pct_raw = float((raw < true_raw).mean())
        pct_res = float((res < true_res).mean())
        rows.append({'target': '+'.join(key), 'n_cand': len(cand),
                     'raw_mean': raw.mean(), 'raw_std': raw.std(), 'raw_max': raw.max(),
                     'raw_range': raw.max() - raw.min(), 'true_raw': true_raw, 'true_raw_pctile': pct_raw,
                     'res_mean': res.mean(), 'res_std': res.std(), 'res_max': res.max(),
                     'res_range': res.max() - res.min(), 'true_res': true_res, 'true_res_pctile': pct_res})
        print(f"[{'+'.join(key):18s}] RAW: mean={raw.mean():.3f} std={raw.std():.3f} "
              f"range={raw.max()-raw.min():.3f} true@pctile={pct_raw:.2f}  ||  "
              f"RESID: mean={res.mean():.3f} std={res.std():.3f} range={res.max()-res.min():.3f} "
              f"true@pctile={pct_res:.2f}", flush=True)

    import statistics as st
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'target'}
    print("\n===== AGGREGATE (mean over targets) =====")
    print(f"  RAW reward   : std={agg['raw_std']:.3f}  range={agg['raw_range']:.3f}  "
          f"true-gene percentile={agg['true_raw_pctile']:.2f}")
    print(f"  RESID reward : std={agg['res_std']:.3f}  range={agg['res_range']:.3f}  "
          f"true-gene percentile={agg['true_res_pctile']:.2f}")
    print("  (flat raw std + true≈median percentile ⇒ reward is action-independent = PPO has no signal;")
    print("   larger resid std + true high percentile ⇒ drift-removed reward would give PPO signal)")
    json.dump({'targets': rows, 'aggregate': agg}, open(f'{args.output}/reward_landscape.json', 'w'), indent=2)
    print(f"saved → {args.output}/reward_landscape.json")


if __name__ == '__main__':
    main()
