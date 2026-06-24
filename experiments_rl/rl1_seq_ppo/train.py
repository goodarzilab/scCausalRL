"""M1 — Sequential PPO over the frozen world model (gene-space reward).

One categorical gene per step; a combo is reached as a SEQUENCE of single-gene steps
(ε threaded across steps → genuine multi-step counterfactual). Goal-conditioned
actor-critic on [z_t, z_target]; action head = #candidate genes (shrunk from the dead
run's ~4000). Reward = the core env's gene-space DE-match potential + ensemble
pessimism. The trained actor is wrapped as a uniform planner for the value-add ladder.

Usage:
  python rl1_seq_ppo/train.py --iters 6000 --output rl1_seq_ppo/results
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
from rl1_seq_ppo.ppo import PPO


def _split(obs, d):
    return obs[:d], obs[d:]


def train(op, data, env, ppo, train_keys, n_iters=6000, n_steps_per_update=128,
          rng=None, log_every=500, wb=None):
    d = op.d
    rng = rng or np.random.default_rng(0)
    step_buf, ep_returns, stats = 0, [], {}
    for it in range(1, n_iters + 1):
        key = train_keys[int(rng.integers(len(train_keys)))]
        k = 2 if isinstance(key, tuple) else 1
        env.max_steps = k
        obs = env.reset(key, rng=rng)
        ep_r = 0.0
        for _ in range(k):
            s, tgt = _split(obs, d)
            a, logp, v = ppo.select_action(s, tgt)
            obs2, r, done, info = env.step([data.candidate_idx[a]])
            ppo.store_transition(s, tgt, a, r, done, logp, v)
            obs = obs2; step_buf += 1; ep_r += r
            if done:
                break
        ep_returns.append(ep_r)
        if step_buf >= n_steps_per_update:
            s, tgt = _split(obs, d)
            stats = ppo.update(s, tgt); step_buf = 0
        if it % log_every == 0:
            mret = float(np.mean(ep_returns[-log_every:]))
            print(f"  iter {it:5d}  ep_return(mean@{log_every})={mret:.3f}  "
                  f"actor_loss={stats.get('actor_loss', 0):.3f}  entropy={stats.get('entropy', 0):.3f}",
                  flush=True)
            if wb:
                wb.log({'iter': it, 'ep_return': mret, 'actor_loss': stats.get('actor_loss', 0),
                        'critic_loss': stats.get('critic_loss', 0), 'entropy': stats.get('entropy', 0)})
    return ppo


def make_planner(ppo, op, data, env):
    """Wrap the trained actor as a uniform planner(target_key) -> gene_set (greedy,
    no-duplicate rollout)."""
    d = op.d
    @torch.no_grad()
    def planner(target_key):
        k = 2 if isinstance(target_key, tuple) else 1
        env.max_steps = k
        obs = env.reset(target_key)
        chosen_a, chosen_g = [], []
        for _ in range(k):
            s, tgt = _split(obs, d)
            dist = ppo.actor.forward(s.unsqueeze(0), tgt.unsqueeze(0))
            logits = dist.logits.squeeze(0).clone()
            for a in chosen_a:
                logits[a] = -1e9
            a = int(logits.argmax())
            chosen_a.append(a); chosen_g.append(data.candidate_idx[a])
            obs, _, done, _ = env.step([data.candidate_idx[a]])
            if done:
                break
        return chosen_g
    return planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--n_dag', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--max_steps', type=int, default=2)
    ap.add_argument('--n_ctrl', type=int, default=300)
    ap.add_argument('--entropy_coeff', type=float, default=0.02)
    ap.add_argument('--pool', default='full', choices=['full', 'screened'])
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--run_name', default='seq_ppo')
    ap.add_argument('--output', default='rl1_seq_ppo/results')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model, meta = load_frozen_wm(device=dev)
    op = WorldModelOperator(model, meta, n_dag_samples=args.n_dag)
    data = WMData(model, meta, op, n_ctrl_cells=args.n_ctrl)
    data.set_pool(args.pool)
    env = CausalLatentEnv(op, data, max_steps=args.max_steps, alpha_pess=args.alpha,
                          n_dag_samples=args.n_dag)

    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project='CausalBioRL', name=args.run_name,
                            config={**vars(args), 'method': 'seq_ppo', 'n_actions': len(data.candidate_idx)})
        except Exception as e:
            print(f"[wandb disabled: {e}]")

    # train on the TRAIN split (singles + combos); validate on the held-out combos
    train_keys = list(data.single_split['train']) + list(data.combo_split['train'])
    print(f"[seq-ppo] pool={args.pool} |actions|={len(data.candidate_idx)}  train tasks: "
          f"{len(data.single_split['train'])} singles + {len(data.combo_split['train'])} combos")
    ppo = PPO(state_dim=op.d, n_actions=len(data.candidate_idx),
              entropy_coeff=args.entropy_coeff, device=dev)
    train(op, data, env, ppo, train_keys, n_iters=args.iters, rng=np.random.default_rng(0), wb=wb)
    ppo.save(os.path.join(args.output, 'seq_ppo.pt'))

    # fast policy-only eval on held-out combos (no MPC — merge with the cached
    # baseline ladder from run_core_check for the full comparison)
    from core.validate import evaluate_planner
    from core.baselines import Planners
    planners = Planners(op, data)
    planner = make_planner(ppo, op, data, env)
    df, summ = evaluate_planner(planner, data.combo_split['test'], planners, data, label='seq_PPO')
    df.to_csv(os.path.join(args.output, 'seqppo_test.csv'), index=False)
    print("\n===== seq-PPO on held-out combos =====")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()})
    if wb:
        wb.log({f'eval/{k}': v for k, v in summ.items() if isinstance(v, (int, float))})
        wb.finish()
    print(f"saved → {args.output}/")


if __name__ == "__main__":
    main()
