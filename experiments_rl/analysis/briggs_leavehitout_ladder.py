"""Leak-free RL ladder (Briggs): fit the reward (readout + Mahalanobis support) ONLY on a TRAIN set that
EXCLUDES the recovery-target TFs (P1 u P2), keep those TFs in the candidate pool, and ask EVERY planner
whether it still rediscovers them. This removes the leakage that inflated in-sample recovery. All planners
(MPC-beam, MPC-greedy, distill, BC-init+PPO, from-scratch set-PPO) use the SAME hit-excluded reward.
Reports per planner: held-out P1/P2 recovery (CLEAN) + generic-driver rate (drivers NOT held out => context)
+ distinct cocktails. Prod WM, clean joung_ctrl readout background. Run via sbatch.
"""
import os, sys, argparse, numpy as np, scipy.sparse as sp, torch
RL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, RL)
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.briggs_fate_reward import FateReward
from core.env import CausalLatentEnv
from rl2_set_ppo.set_ppo import SetPPO
from rl2_set_ppo.set_policy import SetPolicy
from analysis.briggs_setppo_bcfinetune import mpc_beam, FateData
from analysis.briggs_fate_set_ppo import P1, P2, DRIVERS
EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{EXP20}/results/grid/v2b_prod_deg8_all40/checkpoints/phase1_epoch40.pt')
    ap.add_argument('--kmax', type=int, default=7); ap.add_argument('--n_ctrl', type=int, default=256)
    ap.add_argument('--n_dag', type=int, default=6); ap.add_argument('--bc_epochs', type=int, default=500)
    ap.add_argument('--ft_iters', type=int, default=3000); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    TAG = a.ckpt.split('results/grid/')[1].split('/')[0]; RB = f'{EXP20}/results/briggs/readout_ridge_{TAG}'
    HELD = P1 | P2                                                       # 12 recovery-target TFs held out of the reward
    model, meta = load_frozen_wm(a.ckpt, EXP20, a.device); op = WorldModelOperator(model, meta, n_dag_samples=a.n_dag)
    g2i = meta['gene_to_idx']; d = model.latent_dim
    sys.path.insert(0, EXP20); from data.load_joung import load_joung
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt",
                    split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    _, _, eps = model.encoder(xc); eps0 = eps.mean(0)
    # hit-excluded reward: holdhits readouts (fit on non-P1/P2 combos) + support excluding P1/P2
    R = FateReward(op, [f'{RB}_rho_cao_holdhits.npz', f'{RB}_rho_briggs_holdhits.npz'], eps0, lam=1.0, n_dag=a.n_dag)
    R.set_support(f'{EXP20}/results/briggs/readout_feats_{TAG}_ans_joungctrl.npz', 2.0, exclude_tfs=list(HELD))
    import csv
    reliable = set(open(V + "/effective_tf_genes.txt").read().split()); briggs = set()
    for r in csv.DictReader(open(f'{EXP20}/results/briggs/observed_combo_targets.csv')): briggs.update(r['combo'].split('|'))
    pool = sorted({g for g in (reliable | briggs) if g in g2i}); pidx = [g2i[g] for g in pool]; i2g = {g2i[g]: g for g in pool}
    z0 = op.scm_forward(eps0.unsqueeze(0), op.mean_A()).squeeze(0); data = FateData(eps0, z0, pidx)
    env = CausalLatentEnv(op, data, max_steps=1, n_dag_samples=a.n_dag, fate_reward=R, reward_clip=1000.0)
    cpos = {g: i for i, g in enumerate(pidx)}
    print(f"[leave-hit-out ladder] pool={len(pool)} (P1/P2 kept in pool) | HELD-OUT of reward: {sorted(HELD)}", flush=True)

    def recov(cocktails):                                               # cocktails: list of gene-idx iterables
        prop = set()
        for s in cocktails: prop.update(i2g[i] for i in s if i in i2g)
        return dict(heldP1=len(prop & P1), heldP2=len(prop & P2), drv_rate=round(len(prop & DRIVERS) / max(1, len(prop)), 3),
                    ndistinct=len(cocktails), prop=len(prop))

    def sample_top(policy, n=2000, topN=40):
        seen = {}
        for _ in range(n):
            obs = env.reset(None); _, genes, _, _ = policy.sample(obs[:d], obs[d:])
            gg = [pidx[g] for g in genes]
            if gg: seen.setdefault(frozenset(gg), None)
        for k in list(seen):
            rr, _, _ = R.pessimistic([list(k)]); seen[k] = float(rr[0])
        return [list(k) for k, _ in sorted(seen.items(), key=lambda kv: -kv[1])[:topN]]

    demos = mpc_beam(R, pidx, beam=16, kmax=a.kmax, topN=40)             # MPC expert cocktails (on hit-excluded reward)
    rows = {}
    rows['MPC (beam)'] = recov(demos)
    rows['MPC-greedy'] = recov(mpc_beam(R, pidx, beam=1, kmax=a.kmax, topN=40))

    S = torch.stack([env.reset(None)[:d]] * len(demos)); T = torch.stack([env.reset(None)[d:]] * len(demos))
    TOK = torch.tensor([[cpos[g] + 1 for g in s][:a.kmax] + [0] * (a.kmax - len(s)) for s in demos], dtype=torch.long)
    # distill (BC only)
    ag_d = SetPPO(state_dim=d, n_candidates=len(pool), k_max=a.kmax, device=a.device)
    od = torch.optim.Adam(ag_d.policy.parameters(), 1e-3)
    for _ in range(a.bc_epochs):
        lp, e = ag_d.policy.evaluate(S, T, TOK); (-lp.mean() - 1e-3 * e.mean()).backward(); od.step(); od.zero_grad()
    rows['distill (BC)'] = recov(sample_top(ag_d.policy))
    # BC-init + PPO
    ag_b = SetPPO(state_dim=d, n_candidates=len(pool), k_max=a.kmax, entropy_coeff=0.02, device=a.device)
    ob = torch.optim.Adam(ag_b.policy.parameters(), 1e-3)
    for _ in range(a.bc_epochs):
        lp, e = ag_b.policy.evaluate(S, T, TOK); (-lp.mean() - 1e-3 * e.mean()).backward(); ob.step(); ob.zero_grad()
    for it in range(a.ft_iters):
        obs = env.reset(None); tok, gen, lg, v = ag_b.select(obs[:d], obs[d:])
        _, rw, _, _ = env.step([pidx[g] for g in gen]); ag_b.store(obs[:d], obs[d:], tok, rw, lg, v)
        if len(ag_b._buf) >= 128: ag_b.update()
    rows['BC-init+PPO'] = recov(sample_top(ag_b.policy))
    # from-scratch set-PPO
    ag_s = SetPPO(state_dim=d, n_candidates=len(pool), k_max=a.kmax, entropy_coeff=0.02, device=a.device)
    for it in range(a.ft_iters):
        obs = env.reset(None); tok, gen, lg, v = ag_s.select(obs[:d], obs[d:])
        _, rw, _, _ = env.step([pidx[g] for g in gen]); ag_s.store(obs[:d], obs[d:], tok, rw, lg, v)
        if len(ag_s._buf) >= 128: ag_s.update()
    rows['set-PPO (scratch)'] = recov(sample_top(ag_s.policy))

    print(f"\n{'='*78}\nLEAVE-HIT-OUT RL LADDER (P1/P2 held out of reward; prod WM) — held-out recovery is CLEAN", flush=True)
    print(f"{'planner':>18} | {'held-P1/6':>9} {'held-P2/6':>9} {'drv_rate':>9} {'distinct':>9}", flush=True)
    for k, v in rows.items():
        print(f"{k:>18} | {v['heldP1']:>9} {v['heldP2']:>9} {v['drv_rate']:>9} {v['ndistinct']:>9}", flush=True)
    print(f"\n(driver rate context: drivers were NOT held out, so that column is semi-in-sample; "
          f"held-P1/P2 is the clean leak-free number. chance driver rate ~0.03)", flush=True)


if __name__ == '__main__':
    main()
