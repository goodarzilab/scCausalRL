"""M3 — MPC-distill (behavior cloning) for the Briggs HSPC fate objective. From-scratch set-PPO (M2)
can't learn: the pessimistic reward is sparse (random cocktails are off-manifold -> clipped -> no gradient).
Distill sidesteps that by imitating the M0 fate-MPC's chosen cocktails (the de-novo candidate table) into
the SetPolicy via supervised cross-entropy, then evaluates recovery. Completes the value-add ladder:
  M0 planner  ~  M3 distill  >>  M2 from-scratch PPO.
Run: python analysis/briggs_fate_distill.py  (reuses de-novo candidates as BC targets)
"""
import os, sys, argparse, csv, numpy as np, scipy.sparse as sp, torch
RL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, RL)
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.briggs_fate_reward import FateReward
from core.env import CausalLatentEnv
from rl2_set_ppo.set_policy import SetPolicy
from analysis.briggs_fate_set_ppo import FateData, P1, P2, DRIVERS
EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{EXP20}/results/grid/v2b_deg3_t25_frozen_r32_s1/checkpoints/phase1_epoch40.pt')
    RB = f'{EXP20}/results/briggs/readout_ridge_v2b_deg3_t25_frozen_r32_s1'
    ap.add_argument('--targets', default=f'{EXP20}/results/briggs/denovo_candidates.csv')
    ap.add_argument('--kmax', type=int, default=7); ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--n_ctrl', type=int, default=256); ap.add_argument('--n_dag', type=int, default=6); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    model, meta = load_frozen_wm(a.ckpt, EXP20, a.device)
    op = WorldModelOperator(model, meta, n_dag_samples=a.n_dag); g2i = meta['gene_to_idx']; d = op.d
    sys.path.insert(0, EXP20); from data.load_joung import load_joung
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None,
                    effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    _, _, eps = model.encoder(xc); eps0 = eps.mean(0)
    R = FateReward(op, [f'{RB}_rho_cao.npz', f'{RB}_rho_briggs.npz'], eps0, lam=1.0, n_dag=a.n_dag).set_support(f'{EXP20}/results/briggs/readout_feats_v2b_deg3_t25_frozen_r32_s1.npz', 2.0)

    reliable = set(open(V + "/effective_tf_genes.txt").read().split()); briggs = set()
    for r in csv.DictReader(open(f'{EXP20}/results/briggs/observed_combo_targets.csv')): briggs.update(r['combo'].split('|'))
    pool = sorted({g for g in (reliable | briggs) if g in g2i}); pidx = [g2i[g] for g in pool]
    g2pos = {g: p for p, g in enumerate(pool)}; i2g = {g2i[g]: g for g in pool}
    z0 = op.scm_forward(eps0.unsqueeze(0), op.mean_A()).squeeze(0)
    data = FateData(eps0, z0, pidx)
    env = CausalLatentEnv(op, data, max_steps=1, n_dag_samples=a.n_dag, fate_reward=R)
    obs = env.reset(None); s_emb, t_emb = obs[:d], obs[d:]

    # BC targets: planner cocktails -> token rows (token = pool_pos+1, STOP=0, pad to kmax)
    rows = []
    for r in csv.DictReader(open(a.targets)):
        genes = [g for g in r['combo'].split('|') if g in g2pos]
        if not genes: continue
        toks = [g2pos[g] + 1 for g in genes][:a.kmax] + [0] * (a.kmax - min(len(genes), a.kmax))
        rows.append(toks[:a.kmax])
    TOK = torch.tensor(rows, dtype=torch.long, device=a.device)
    print(f"[distill] {len(rows)} BC target cocktails | pool={len(pool)} kmax={a.kmax}", flush=True)

    policy = SetPolicy(d, len(pool), k_max=a.kmax).to(a.device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    S = s_emb.unsqueeze(0).expand(len(rows), -1); T = t_emb.unsqueeze(0).expand(len(rows), -1)
    for ep in range(a.epochs):
        policy.train(); logp, ent = policy.evaluate(S, T, TOK)
        loss = -(logp.mean()) - 0.001 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 100 == 0: print(f"  ep {ep+1} BC logp {float(logp.mean()):+.3f}", flush=True)

    # eval: sample cocktails, recovery
    policy.eval(); seen = {}
    for _ in range(2000):
        toks, genes, _, _ = policy.sample(s_emb, t_emb)
        gi = [data.candidate_idx[g] for g in genes]
        if gi: seen.setdefault(frozenset(gi), None)
    for key in list(seen):
        rr, _, _ = R.pessimistic([list(key)]); seen[key] = float(rr[0])
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:20]
    proposed = set()
    print(f"\n[M3 distill TOP cocktails]", flush=True)
    observed = set(frozenset(r['combo'].split('|')) for r in csv.DictReader(open(f'{EXP20}/results/briggs/observed_combo_targets.csv')))
    for key, rv in ranked:
        names = sorted(i2g[i] for i in key); proposed.update(names)
        novel = frozenset(names) not in observed
        ov1 = len(set(names) & P1); ov2 = len(set(names) & P2); dv = len(set(names) & DRIVERS)
        tag = (f" P1∩{ov1}" if ov1 else "") + (f" P2∩{ov2}" if ov2 else "") + (f" drv∩{dv}" if dv else "")
        print(f"  {'*' if novel else ' '} k={len(key)} r~ {rv:+.3f} | {names}{tag}", flush=True)
    base = len([g for g in pool if g in DRIVERS]) / len(pool)
    print(f"\n[M3 distill recovery] proposed={len(proposed)} | driver rate {len(proposed&DRIVERS)/max(1,len(proposed)):.2f} vs base {base:.2f}", flush=True)
    print(f"  P1: {sorted(proposed & P1)} | P2: {sorted(proposed & P2)} | drivers: {sorted(proposed & DRIVERS)}", flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
