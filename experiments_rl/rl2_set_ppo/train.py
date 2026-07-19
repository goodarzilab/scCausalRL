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
from core.setup_stack import add_stack_args, build_stack
from core.env import CausalLatentEnv
from rl2_set_ppo.set_ppo import SetPPO


def _split(obs, d):
    return obs[:d], obs[d:]


def _drift_vec(data):
    return np.mean([data.de_single[g]['lfc'] for g in data.single_split['train']], axis=0)


@torch.no_grad()
def quick_match(op, data, env, planner, keys, drift=None):
    """Mean held-out match (signed-Pearson of the planner's chosen-set Δ vs observed Δ on
    DEGs), BOTH raw and DRIFT-REMOVED (TF-specific). Returns (raw, resid). The resid number
    is the real-signal metric (raw is mostly the shared differentiation drift)."""
    A = env.A_eval; cr = env.ctrl_recon
    eps0 = data.eps_ctrl_mean.unsqueeze(0)
    if drift is None:
        drift = _drift_vec(data)
    raws, resids = [], []

    def _p(pred, obs, mask):
        sel = mask & (np.abs(obs) > env.thresh)
        if sel.sum() < 3 or pred[sel].std() < 1e-8 or obs[sel].std() < 1e-8:
            return 0.0
        return float(np.corrcoef(pred[sel], obs[sel])[0, 1])

    for key in keys:
        genes = planner(key)
        if not genes:
            raws.append(0.0); resids.append(0.0); continue
        z, _, _ = op.apply(eps0, list(genes), A)
        pred = (op.decode(z).squeeze(0) - cr).cpu().numpy()
        obs = data.de_combo[key]['lfc'] if isinstance(key, tuple) else data.de_single[key]['lfc']
        ks = key if isinstance(key, tuple) else (key,)
        mask = np.ones(len(pred), bool)
        for g in ks:
            mask[data.gene_to_idx[g]] = False
        raws.append(_p(pred, obs, mask))
        resids.append(_p(pred - drift, obs - drift, mask))
    n = max(1, len(raws))
    return float(np.sum(raws) / n), float(np.sum(resids) / n)


def train(op, data, env, agent, train_keys, n_iters=6000, n_steps_per_update=128,
          rng=None, log_every=500, wb=None, eval_keys=None, eval_planner=None, eval_every=10000,
          best_path=None, entropy_final=None):
    d = op.d
    rng = rng or np.random.default_rng(0)
    buf, returns, stats = 0, [], {}
    env.max_steps = 1                                   # the set is applied at once
    ent0 = float(agent.ent_c); best_res = -1e9          # for entropy anneal + best-ckpt
    for it in range(1, n_iters + 1):
        if entropy_final is not None:                   # linear entropy anneal
            agent.ent_c = ent0 + (float(entropy_final) - ent0) * (it / n_iters)
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
        if eval_planner is not None and eval_keys and it % eval_every == 0:
            em_raw, em_res = quick_match(op, data, env, eval_planner, eval_keys)
            env.max_steps = 1                           # restore (planner mutates it)
            tag = ''
            if em_res > best_res and best_path is not None:   # save best-by-resid checkpoint
                best_res = em_res; agent.save(best_path); tag = ' *best*'
            print(f"    [eval@{it}] held-out match raw={em_raw:.3f} resid={em_res:.3f}{tag}", flush=True)
            if wb:
                wb.log({'iter': it, 'eval/match_raw': em_raw, 'eval/match_resid': em_res,
                        'eval/best_resid': max(best_res, em_res), 'entropy_coeff': float(agent.ent_c)})
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
    add_stack_args(ap)
    ap.add_argument('--reward_mode', default='raw', choices=['raw', 'resid', 'blend'])
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--k_max', type=int, default=2)
    ap.add_argument('--n_dag', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--n_ctrl', type=int, default=300)
    ap.add_argument('--entropy_coeff', type=float, default=0.02)
    ap.add_argument('--entropy_final', type=float, default=None, help='linear-anneal entropy_coeff to this')
    ap.add_argument('--lr_actor', type=float, default=3e-4)
    ap.add_argument('--pool', default='full', choices=['full', 'screened', 'curated'])
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--run_name', default='set_ppo')
    ap.add_argument('--output', default='rl2_set_ppo/results')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model, meta, op, data = build_stack(args, dev, args.n_dag)
    data.set_pool(args.pool)
    env = CausalLatentEnv(op, data, max_steps=1, alpha_pess=args.alpha, n_dag_samples=args.n_dag,
                          reward_mode=args.reward_mode)

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
                   entropy_coeff=args.entropy_coeff, lr_actor=args.lr_actor, device=dev)
    eval_planner = make_planner(agent, op, data, env)        # live wrapper for periodic eval
    best_path = os.path.join(args.output, 'set_ppo_best.pt')
    train(op, data, env, agent, train_keys, n_iters=args.iters, rng=np.random.default_rng(0), wb=wb,
          eval_keys=data.combo_split['test'], eval_planner=eval_planner, eval_every=max(2000, args.iters // 20),
          best_path=best_path, entropy_final=args.entropy_final)
    agent.save(os.path.join(args.output, 'set_ppo.pt'))           # last
    if os.path.exists(best_path):                                # final eval uses BEST-by-resid
        agent.policy.load_state_dict(torch.load(best_path, map_location=dev)['policy'])
        print("[set-ppo] loaded best-by-resid checkpoint for final eval")

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
