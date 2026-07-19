"""M3 — MPC → set-policy distillation (BC warm-start + PPO fine-tune).

Bridges the MPC oracle to a learned set policy: (1) collect MPC plans over the train
targets, (2) behavioral-clone the set policy to MPC's chosen set (cross-entropy on the
token sequence via SetPolicy.evaluate), (3) PPO fine-tune against the gene-space
reward so the policy can exceed the oracle by amortizing across targets. MPC plans are
cached to disk (MPC is the slow part). Distilled policy evaluated on the ladder.

Usage:
  python rl3_mpc_distill/distill.py --bc_epochs 200 --ppo_iters 3000 --output rl3_mpc_distill/results
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.setup_stack import add_stack_args, build_stack
from core.env import CausalLatentEnv
from rl0_mpc.mpc_planner import MPCPlanner
from rl2_set_ppo.set_ppo import SetPPO
from rl2_set_ppo.train import train as ppo_train, make_planner


def _tokens_from_genes(gene_idx, data, k_max):
    """MPC gene indices → padded token row (candidate-local g+1, STOP=0)."""
    local = sorted(data.candidate_idx.index(g) for g in gene_idx if g in data.candidate_idx)
    toks = [g + 1 for g in local][:k_max]
    toks = toks + [0] * (k_max - len(toks))               # STOP-pad
    return toks


def collect_mpc(op, data, env, mpc, keys, cache_path):
    """For each train key: state [z0;z_target] + MPC's chosen token row. Cached."""
    if os.path.exists(cache_path):
        d = torch.load(cache_path)
        print(f"[distill] loaded {len(d['tokens'])} cached MPC plans")
        return d['S'], d['T'], d['tokens']
    dd = op.d; S, T, TOK = [], [], []
    for i, key in enumerate(keys):
        obs = env.reset(key)
        chosen = mpc.plan(key)
        S.append(obs[:dd].cpu()); T.append(obs[dd:].cpu())
        TOK.append(_tokens_from_genes(chosen, data, mpc.max_set))
        if (i + 1) % 20 == 0:
            print(f"  collected {i+1}/{len(keys)} MPC plans", flush=True)
    S = torch.stack(S); T = torch.stack(T); TOK = torch.tensor(TOK, dtype=torch.long)
    torch.save({'S': S, 'T': T, 'tokens': TOK}, cache_path)
    return S, T, TOK


def behavioral_clone(agent, S, T, TOK, epochs=200, batch=64, lr=1e-3, device='cpu'):
    """Cross-entropy: maximize log π_set(MPC tokens | s). Uses SetPolicy.evaluate."""
    opt = torch.optim.Adam(agent.policy.parameters(), lr=lr)
    S, T, TOK = S.to(device), T.to(device), TOK.to(device)
    n = S.shape[0]
    for ep in range(1, epochs + 1):
        idx = torch.randperm(n, device=device)
        tot = 0.0
        for st in range(0, n, batch):
            b = idx[st:st + batch]
            logp, _ = agent.policy.evaluate(S[b], T[b], TOK[b])
            loss = -logp.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if ep % 50 == 0:
            print(f"  BC epoch {ep}  -logp={tot/max(1,(n//batch+1)):.3f}", flush=True)
    return agent


def main():
    ap = argparse.ArgumentParser()
    add_stack_args(ap)
    ap.add_argument('--bc_epochs', type=int, default=200)
    ap.add_argument('--ppo_iters', type=int, default=3000)
    ap.add_argument('--k_max', type=int, default=2)
    ap.add_argument('--n_dag', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--mpc_beam', type=int, default=4)
    ap.add_argument('--n_ctrl', type=int, default=300)
    ap.add_argument('--pool', default='full', choices=['full', 'screened', 'curated'])
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--run_name', default='set_ppo_distill')
    ap.add_argument('--output', default='rl3_mpc_distill/results')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model, meta, op, data = build_stack(args, dev, args.n_dag)
    data.set_pool(args.pool)
    env = CausalLatentEnv(op, data, max_steps=1, alpha_pess=args.alpha, n_dag_samples=args.n_dag)
    mpc = MPCPlanner(op, data, max_set=args.k_max, beam_width=args.mpc_beam,
                     n_dag_samples=args.n_dag, pessimism_alpha=args.alpha)
    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project='CausalBioRL', name=args.run_name,
                            config={**vars(args), 'method': 'set_ppo_distill', 'C': len(data.candidate_idx)})
        except Exception as e:
            print(f"[wandb disabled: {e}]")

    train_keys = list(data.single_split['train']) + list(data.combo_split['train'])
    print(f"[distill] collecting MPC plans for {len(train_keys)} train tasks...")
    S, T, TOK = collect_mpc(op, data, env, mpc, train_keys,
                            os.path.join(args.output, 'mpc_plans.pt'))

    agent = SetPPO(state_dim=op.d, n_candidates=len(data.candidate_idx), k_max=args.k_max, device=dev)
    print("[distill] behavioral cloning...")
    behavioral_clone(agent, S, T, TOK, epochs=args.bc_epochs, device=dev)
    agent.save(os.path.join(args.output, 'set_ppo_bc.pt'))

    if args.ppo_iters > 0:
        print("[distill] PPO fine-tuning...")
        ppo_train(op, data, env, agent, train_keys, n_iters=args.ppo_iters,
                  rng=np.random.default_rng(0), wb=wb)
        agent.save(os.path.join(args.output, 'set_ppo_distill.pt'))

    from core.validate import evaluate_planner
    from core.baselines import Planners
    planners = Planners(op, data)
    planner = make_planner(agent, op, data, env)
    df, summ = evaluate_planner(planner, data.combo_split['test'], planners, data, label='set_PPO_distill')
    df.to_csv(os.path.join(args.output, 'distill_test.csv'), index=False)
    print("\n===== MPC-distilled set-PPO on held-out combos =====")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()})
    if wb:
        wb.log({f'eval/{k}': v for k, v in summ.items() if isinstance(v, (int, float))})
        wb.finish()
    print(f"saved → {args.output}/")


if __name__ == "__main__":
    main()
