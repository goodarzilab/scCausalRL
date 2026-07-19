"""M2 — set-PPO learned policy for the Briggs HSPC fate objective (amortized planner). Same merged reward
as the M0 fate-MPC: r~ = min-readout(z_post) - lam*DAG_std - gamma*Mahalanobis, single fixed HSPC goal.
Trains SetPolicy over the tiered pool, then samples cocktails from the greedy/stochastic policy and scores
RECOVERY (P1/P2 members, drivers) exactly like fate-MPC — the value-add-ladder head-to-head (does a learned
policy match the planner?). Run via sbatch (CPU).
"""
import os, sys, argparse, numpy as np, scipy.sparse as sp, torch
RL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, RL)
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.briggs_fate_reward import FateReward
from core.env import CausalLatentEnv
from rl2_set_ppo.set_ppo import SetPPO
EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
P1 = {'TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2'}
P2 = {'DDIT3', 'MZF1', 'HOXB3', 'MIER1', 'SIRT6', 'ZNF85'}
DRIVERS = {'ETV2', 'GATA2', 'RUNX1', 'LMO2', 'TAL1', 'LYL1', 'GFI1B', 'SPI1', 'FLI1', 'ERG', 'MEIS1',
           'HOXA9', 'GATA1', 'MYB', 'HHEX', 'RUNX3', 'IKZF1', 'CEBPA', 'KLF1', 'MECOM'}


class FateData:  # minimal stack shim the env/policy need (avoids WMDataJoung vocab defaults)
    def __init__(self, eps_ctrl_mean, z0, candidate_idx):
        self.eps_ctrl_mean = eps_ctrl_mean; self.z0 = z0; self.candidate_idx = candidate_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{EXP20}/results/grid/v2b_deg3_t25_frozen_r32_s1/checkpoints/phase1_epoch40.pt')
    RB = f'{EXP20}/results/briggs/readout_ridge_v2b_deg3_t25_frozen_r32_s1'
    ap.add_argument('--readout_cao', default=f'{RB}_rho_cao.npz'); ap.add_argument('--readout_briggs', default=f'{RB}_rho_briggs.npz')
    ap.add_argument('--support_feats', default=f'{EXP20}/results/briggs/readout_feats_v2b_deg3_t25_frozen_r32_s1.npz')
    ap.add_argument('--lam', type=float, default=1.0); ap.add_argument('--support_gamma', type=float, default=2.0)
    ap.add_argument('--kmax', type=int, default=7); ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--n_dag', type=int, default=6); ap.add_argument('--n_ctrl', type=int, default=256)
    ap.add_argument('--entropy', type=float, default=0.02); ap.add_argument('--update_every', type=int, default=128)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--out', default=f'{EXP20}/results/briggs/setppo_candidates.csv')
    a = ap.parse_args()

    model, meta = load_frozen_wm(a.ckpt, EXP20, a.device)
    op = WorldModelOperator(model, meta, n_dag_samples=a.n_dag)
    g2i = meta['gene_to_idx']; d = op.d
    sys.path.insert(0, EXP20); from data.load_joung import load_joung
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None,
                    effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    _, _, eps = model.encoder(xc); eps0 = eps.mean(0)
    R = FateReward(op, [a.readout_cao, a.readout_briggs], eps0, lam=a.lam, n_dag=a.n_dag).set_support(a.support_feats, a.support_gamma)

    # tiered pool
    reliable = set(open(V + "/effective_tf_genes.txt").read().split())
    import csv
    briggs = set()
    for r in csv.DictReader(open(f'{EXP20}/results/briggs/observed_combo_targets.csv')):
        briggs.update(r['combo'].split('|'))
    pool = sorted({g for g in (reliable | briggs) if g in g2i}); pidx = [g2i[g] for g in pool]; i2g = {g2i[g]: g for g in pool}
    z0 = op.scm_forward(eps0.unsqueeze(0), op.mean_A()).squeeze(0)
    data = FateData(eps0, z0, pidx)
    env = CausalLatentEnv(op, data, max_steps=1, n_dag_samples=a.n_dag, fate_reward=R)
    agent = SetPPO(state_dim=d, n_candidates=len(pool), k_max=a.kmax, entropy_coeff=a.entropy, device=a.device)
    print(f"[set-PPO fate] pool={len(pool)} kmax={a.kmax} iters={a.iters} lam={a.lam} gamma={a.support_gamma}", flush=True)

    # train (single fixed HSPC goal)
    running = []
    for it in range(1, a.iters + 1):
        obs = env.reset(None)
        s_emb, t_emb = obs[:d], obs[d:]
        tokens, genes, logp, v = agent.select(s_emb, t_emb)
        gene_idx = [data.candidate_idx[g] for g in genes]
        _, r, _, info = env.step(gene_idx)
        agent.store(s_emb, t_emb, tokens, r, logp, v); running.append(r)
        if len(agent._buf) >= a.update_every:
            st = agent.update()
            if it % (a.update_every * 8) == 0:
                print(f"  iter {it:5d} reward(mean) {np.mean(running[-500:]):+.3f} actor {st.get('actor_loss',0):+.3f} ent {st.get('entropy',0):.3f}", flush=True)

    # eval: sample many cocktails from the trained policy, rank by r~, recovery
    agent.policy.eval()
    seen = {}
    for _ in range(2000):
        obs = env.reset(None); s_emb, t_emb = obs[:d], obs[d:]
        _, genes, _, _ = agent.select(s_emb, t_emb)
        gi = [data.candidate_idx[g] for g in genes]
        if not gi: continue
        key = frozenset(gi)
        if key not in seen:
            rr, base, u = R.pessimistic([list(gi)]); seen[key] = float(rr[0])
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:40]
    rows, proposed = [], set()
    print(f"\n[set-PPO TOP cocktails by r~]  (* novel vs Briggs)", flush=True)
    observed = set(frozenset(r['combo'].split('|')) for r in csv.DictReader(open(f'{EXP20}/results/briggs/observed_combo_targets.csv')))
    for key, rv in ranked[:20]:
        names = sorted(i2g[i] for i in key); proposed.update(names)
        novel = frozenset(names) not in observed
        ov1 = len(set(names) & P1); ov2 = len(set(names) & P2); dv = len(set(names) & DRIVERS)
        tag = (f" P1∩{ov1}" if ov1 else "") + (f" P2∩{ov2}" if ov2 else "") + (f" drv∩{dv}" if dv else "")
        print(f"  {'*' if novel else ' '} k={len(key)} r~ {rv:+.3f} | {names}{tag}", flush=True)
        rows.append(dict(combo='|'.join(names), k=len(key), r_pess=round(rv, 4), novel=int(novel),
                         P1_overlap=ov1, P2_overlap=ov2, driver_overlap=dv))
    base_rate = len([g for g in pool if g in DRIVERS]) / len(pool)
    print(f"\n[set-PPO recovery] proposed={len(proposed)} | driver rate {len(proposed&DRIVERS)/max(1,len(proposed)):.2f} vs base {base_rate:.2f}", flush=True)
    print(f"  P1 recovered: {sorted(proposed & P1)} | P2: {sorted(proposed & P2)} | drivers: {sorted(proposed & DRIVERS)}", flush=True)
    with open(a.out, 'w', newline='') as f:
        import csv as _csv; w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('wrote', a.out, flush=True)


if __name__ == '__main__':
    main()
