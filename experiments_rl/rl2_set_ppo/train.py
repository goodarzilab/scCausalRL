"""M2 — train the set-based PPO and evaluate on the value-add ladder.

Set-PPO applies the whole set in ONE env step (max_steps=1): the policy's set IS a
combo prediction, so synergy lives in the joint action (vs sequential PPO's imposed
order). Reward = the core env's gene-space DE-match. The trained policy is wrapped as
a uniform greedy planner for the ladder.

Usage:
  python rl2_set_ppo/train.py --iters 6000 --k_max 2 --output rl2_set_ppo/results
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.wm_data import WMData
from core.env import CausalLatentEnv
from rl2_set_ppo.set_ppo import SetPPO


def _split(obs, d):
    return obs[:d], obs[d:]


def train(op, data, env, agent, train_keys, n_iters=6000, n_steps_per_update=128,
          rng=None, log_every=500, wb=None):
    d = op.d
    rng = rng or np.random.default_rng(0)
    buf, returns, stats = 0, [], {}
    env.max_steps = 1                                   # the set is applied at once
    for it in range(1, n_iters + 1):
        key = train_keys[int(rng.integers(len(train_keys)))]
        obs = env.reset(key, rng=rng)
        s, tgt = _split(obs, d)
        tokens, genes, logp, v = agent.select(s, tgt)
        gene_idx = [data.candidate_idx[g] for g in genes]
        _, r, _, info = env.step(gene_idx)
        agent.store(s, tgt, tokens, r, logp, v)
        returns.append(r); buf += 1
        if buf >= n_steps_per_update:
            stats = agent.update(); buf = 0
        if it % log_every == 0:
            mret = float(np.mean(returns[-log_every:]))
            print(f"  iter {it:5d}  reward(mean@{log_every})={mret:.3f}  "
                  f"actor_loss={stats.get('actor_loss', 0):.3f}  entropy={stats.get('entropy', 0):.3f}",
                  flush=True)
            if wb:
                wb.log({'iter': it, 'reward': mret, 'actor_loss': stats.get('actor_loss', 0),
                        'critic_loss': stats.get('critic_loss', 0), 'entropy': stats.get('entropy', 0)})
    return agent


def make_planner(agent, op, data, env):
    """Greedy set rollout (argmax tokens, STOP-aware) → uniform planner."""
    d = op.d
    pol = agent.policy
    @torch.no_grad()
    def planner(target_key):
        env.max_steps = 1
        obs = env.reset(target_key)
        s, tgt = _split(obs, d)
        h = pol.encode(s.unsqueeze(0), tgt.unsqueeze(0))
        ctx = torch.zeros(1, pol.emb_dim, device=h.device)
        chosen = torch.zeros(1, pol.C, dtype=torch.bool, device=h.device)
        genes = []
        for i in range(pol.k_max):
            logits = pol._slot_logits(h, ctx, i, chosen)
            tok = int(logits.argmax())
            if tok == 0:
                break
            g = tok - 1
            genes.append(data.candidate_idx[g])
            chosen[0, g] = True
            ctx = ctx + pol.gene_emb(torch.tensor([g], device=h.device))
        return genes
    return planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--k_max', type=int, default=2)
    ap.add_argument('--n_dag', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--n_ctrl', type=int, default=300)
    ap.add_argument('--entropy_coeff', type=float, default=0.02)
    ap.add_argument('--pool', default='full', choices=['full', 'screened'])
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--run_name', default='set_ppo')
    ap.add_argument('--output', default='rl2_set_ppo/results')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model, meta = load_frozen_wm(device=dev)
    op = WorldModelOperator(model, meta, n_dag_samples=args.n_dag)
    data = WMData(model, meta, op, n_ctrl_cells=args.n_ctrl)
    data.set_pool(args.pool)
    env = CausalLatentEnv(op, data, max_steps=1, alpha_pess=args.alpha, n_dag_samples=args.n_dag)

    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project='CausalBioRL', name=args.run_name,
                            config={**vars(args), 'method': 'set_ppo', 'C': len(data.candidate_idx)})
        except Exception as e:
            print(f"[wandb disabled: {e}]")

    train_keys = list(data.single_split['train']) + list(data.combo_split['train'])
    print(f"[set-ppo] pool={args.pool} k_max={args.k_max} C={len(data.candidate_idx)}  train tasks: "
          f"{len(data.single_split['train'])} singles + {len(data.combo_split['train'])} combos")
    agent = SetPPO(state_dim=op.d, n_candidates=len(data.candidate_idx), k_max=args.k_max,
                   entropy_coeff=args.entropy_coeff, device=dev)
    train(op, data, env, agent, train_keys, n_iters=args.iters, rng=np.random.default_rng(0), wb=wb)
    agent.save(os.path.join(args.output, 'set_ppo.pt'))

    from core.validate import evaluate_planner
    from core.baselines import Planners
    planners = Planners(op, data)
    planner = make_planner(agent, op, data, env)
    df, summ = evaluate_planner(planner, data.combo_split['test'], planners, data, label='set_PPO')
    df.to_csv(os.path.join(args.output, 'setppo_test.csv'), index=False)
    print("\n===== set-PPO on held-out combos =====")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()})
    if wb:
        wb.log({f'eval/{k}': v for k, v in summ.items() if isinstance(v, (int, float))})
        wb.finish()
    print(f"saved → {args.output}/")


if __name__ == "__main__":
    main()
