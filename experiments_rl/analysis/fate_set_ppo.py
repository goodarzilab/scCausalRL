"""Learned fate-policy: set-PPO · RESID · curated trained to reach developmental LINEAGE centroids
(the same Leiden lineages as fate_rl MPC). Reuses SetPPO + CausalLatentEnv(reward_mode='resid')
+ the proven train() loop, but the train/eval targets are Lineage keys (the env already routes
them through observed_de/target_for). Eval is fate-native: nearest-lineage recall@1/@3 of the
policy's greedy set, compared head-to-head with fate-MPC (fate_rl_mpc.csv). Run via sbatch.
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import add_stack_args, build_stack
from core.env import CausalLatentEnv
from core.scoring import torch_match, Lineage
from rl2_set_ppo.set_ppo import SetPPO
from rl2_set_ppo.train import train, make_planner
TFLIN = ('/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/analysis/'
         'human_dev_annotation/joung_tf_lineage.csv')
CHUNK = 512


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--reward_mode', default='resid', choices=['raw', 'resid', 'blend'])
    ap.add_argument('--goals', default='centroid', choices=['centroid', 'driver'],
                    help="centroid: 5 lineage-centroid goals (few). driver: per confident-driver-TF "
                         "goals (~60 train, lineage-labeled) → diversity; eval on held-out drivers.")
    ap.add_argument('--pool', default='curated'); ap.add_argument('--iters', type=int, default=15000)
    ap.add_argument('--k_max', type=int, default=2); ap.add_argument('--n_dag', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0); ap.add_argument('--entropy_coeff', type=float, default=0.02)
    ap.add_argument('--entropy_final', type=float, default=0.005); ap.add_argument('--qmax', type=float, default=0.10)
    ap.add_argument('--output', default='rl2_set_ppo/results/joung_fate_resid_curated')
    ap.add_argument('--wandb', action='store_true'); ap.add_argument('--run_name', default='joung_fate_set_ppo')
    ap.add_argument('--device', default=None)
    a = ap.parse_args(); os.makedirs(a.output, exist_ok=True)
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, meta, op, data = build_stack(a, dev, a.n_dag); data.set_pool(a.pool)
    data.build_lineage_targets(TFLIN, qmax=a.qmax)
    keys = data.lineage_keys; names = [k.name for k in keys]; K = len(keys)
    # per-driver goals: each confident driver TF is its own goal (lineage-labeled). Train on the
    # lineage TRAIN drivers (diversity), eval on the HELD-OUT drivers (reach the right lineage?).
    drv2lin = {g: li for li, n in enumerate(names) for g in data.lineage_drivers[n]['train'] + data.lineage_drivers[n]['test']}
    train_drivers = [g for n in names for g in data.lineage_drivers[n]['train']]
    eval_drivers = [(g, li) for li, n in enumerate(names) for g in data.lineage_drivers[n]['test']]
    train_keys = keys if a.goals == 'centroid' else train_drivers
    env = CausalLatentEnv(op, data, max_steps=1, alpha_pess=a.alpha, n_dag_samples=a.n_dag,
                          reward_mode=a.reward_mode)

    wb = None
    if a.wandb:
        try:
            import wandb; wb = wandb.init(project='CausalBioRL', name=a.run_name,
                                          config={**vars(a), 'method': 'fate_set_ppo', 'K': K})
        except Exception as e:
            print(f"[wandb disabled: {e}]")

    agent = SetPPO(state_dim=op.d, n_candidates=len(data.candidate_idx), k_max=a.k_max,
                   entropy_coeff=a.entropy_coeff, device=dev)
    print(f"[fate-set-ppo] goals={a.goals} | {len(train_keys)} train goals "
          f"({K} lineages, {len(eval_drivers)} held-out drivers) | reward={a.reward_mode} pool={a.pool} "
          f"C={len(data.candidate_idx)} iters={a.iters}")
    # skip the combo quick_match periodic eval — fate eval is below
    train(op, data, env, agent, train_keys=train_keys, n_iters=a.iters, rng=np.random.default_rng(0),
          wb=wb, eval_planner=None, eval_keys=None, entropy_final=a.entropy_final)
    agent.save(os.path.join(a.output, 'fate_set_ppo.pt'))

    # ---- fate-native eval: nearest-lineage recall of the greedy policy set ----
    planner = make_planner(agent, op, data, env)
    A = op.mean_A(); ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]
    drift = torch.tensor(np.mean([data.de_single[g]['lfc'] for g in data.single_split['train']], 0),
                         dtype=torch.float32, device=dev)
    cent = torch.stack([torch.tensor(data.de_lineage[n]['lfc'], dtype=torch.float32, device=dev) - drift
                        for n in names])
    centn = cent / (cent.norm(dim=1, keepdim=True) + 1e-9)

    def pred_resid(gidx):
        S = op.shift_for(gidx).sum(0, keepdim=True)
        z = op.apply_many(data.eps_ctrl_mean, S, A)
        out = []
        for c in z.split(CHUNK):
            out.append(op.decode(c) - ctrl_recon)
        return torch.cat(out)[0] - drift

    # eval targets: centroid mode → the 5 lineage keys; driver mode → HELD-OUT drivers
    targets = list(zip(keys, range(K))) if a.goals == 'centroid' else [(g, li) for g, li in eval_drivers]
    rows = []
    for tgt, li in targets:
        genes = planner(tgt)                              # gene INDICES; tgt = Lineage or driver-TF str
        gset = {data.gene_names[i] for i in genes} if genes else set()
        rv = pred_resid(genes) if genes else None
        order = torch.argsort(-(centn @ (rv / (rv.norm() + 1e-9)))).tolist() if rv is not None else []
        copied = int((not isinstance(tgt, Lineage)) and tgt in gset)   # did it just re-pick the target TF?
        rows.append({'target': (tgt.name.split(':')[1][:12] if isinstance(tgt, Lineage) else tgt),
                     'lineage': names[li].split(':')[1][:12], 'chosen': ','.join(sorted(gset)) or '(none)',
                     'near@1': int(bool(order) and order[0] == li), 'near@3': int(li in order[:3]),
                     'copied_target': copied})
    df = pd.DataFrame(rows)
    print(f"\n{'='*92}\nFATE set-PPO · {a.reward_mode} · {a.pool} · goals={a.goals} — "
          f"{len(targets)} eval targets, {K} lineages")
    print(df.to_string(index=False))
    r1, r3 = df['near@1'].mean(), df['near@3'].mean(); rm = 0.0
    print(f"\nset-PPO  : nearest-lineage recall@1={r1:.2f} recall@3={r3:.2f} "
          f"copied-target={df['copied_target'].mean():.2f}  (chance@1={1/K:.2f})")
    mpcp = f'{HERE}/fate_rl_mpc.csv'
    if os.path.exists(mpcp):
        m = pd.read_csv(mpcp)
        print(f"fate-MPC : nearest-lineage recall@1={m['near@1'].mean():.2f} recall@3={m['near@3'].mean():.2f} "
              f"mean resid_match={m['resid_match'].mean():.3f}  (for reference)")
    df.to_csv(f'{a.output}/fate_set_ppo_eval.csv', index=False)
    if wb:
        wb.log({'fate/recall@1': r1, 'fate/recall@3': r3, 'fate/resid_match': rm}); wb.finish()
    print(f"saved -> {a.output}/")


if __name__ == '__main__':
    main()
