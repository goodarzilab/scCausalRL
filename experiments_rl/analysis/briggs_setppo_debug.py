"""Debug/sweep the Briggs fate set-PPO — WHY does from-scratch PPO fail on the pessimistic reward, and
what fixes it? Instruments training (reward mean/std/min/max, entropy, actor_loss, most-picked TFs over
time) and sweeps the axes: reward-clip width, pessimism strength (lam,gamma), and k_max (3/4/7). Leading
hypothesis: the hard [-10,10] reward clip flattens the sparse pessimistic reward (off-manifold cocktails
all hit the floor -> zero advantage -> no gradient); widening it (or softening pessimism) restores the
Mahalanobis gradient that pulls the policy on-manifold. Run via sbatch (CPU).
"""
import os, sys, argparse, numpy as np, scipy.sparse as sp, torch
from collections import Counter
RL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, RL)
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.briggs_fate_reward import FateReward
from core.env import CausalLatentEnv
from rl2_set_ppo.set_ppo import SetPPO
from analysis.briggs_fate_set_ppo import FateData, P1, P2, DRIVERS
EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"

CONFIGS = [  # (name, kmax, lam, gamma, clip)
    ('A_pess_clip10_k7', 7, 1.0, 2.0, 10.0),      # current FAILING baseline
    ('B_pess_wideclip_k7', 7, 1.0, 2.0, 1000.0),  # same reward, widen clip (test the clip hypothesis)
    ('C_fateonly_k7', 7, 0.0, 0.0, 10.0),         # no pessimism (dense fate only)
    ('D_soft_wideclip_k7', 7, 0.3, 0.5, 1000.0),  # soft pessimism + wide clip
    ('E_soft_wideclip_k3', 3, 0.3, 0.5, 1000.0),  # + smaller k
    ('F_pess_wideclip_k3', 3, 1.0, 2.0, 1000.0),  # strong pessimism, wide clip, small k
    ('G_fateonly_k3', 3, 0.0, 0.0, 10.0),         # dense fate, small k
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{EXP20}/results/grid/v2b_deg3_t25_frozen_r32_all40/checkpoints/phase1_epoch40.pt')
    ap.add_argument('--iters', type=int, default=2500); ap.add_argument('--n_ctrl', type=int, default=128)
    ap.add_argument('--n_dag', type=int, default=4); ap.add_argument('--update_every', type=int, default=128); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    TAG = 'v2b_deg3_t25_frozen_r32_all40'
    RB = f'{EXP20}/results/briggs/readout_ridge_{TAG}'

    model, meta = load_frozen_wm(a.ckpt, EXP20, a.device); g2i = meta['gene_to_idx']; d = model.latent_dim
    sys.path.insert(0, EXP20); from data.load_joung import load_joung
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt",
                    split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    reliable = set(open(V + "/effective_tf_genes.txt").read().split()); import csv
    briggs = set()
    for r in csv.DictReader(open(f'{EXP20}/results/briggs/observed_combo_targets.csv')): briggs.update(r['combo'].split('|'))
    pool = sorted({g for g in (reliable | briggs) if g in g2i}); pidx = [g2i[g] for g in pool]; i2g = {g2i[g]: g for g in pool}

    for name, kmax, lam, gamma, clip in CONFIGS:
        op = WorldModelOperator(model, meta, n_dag_samples=a.n_dag)
        _, _, eps = model.encoder(xc); eps0 = eps.mean(0)
        R = FateReward(op, [f'{RB}_rho_cao.npz', f'{RB}_rho_briggs.npz'], eps0, lam=lam, n_dag=a.n_dag)
        R.set_support(f'{EXP20}/results/briggs/readout_feats_{TAG}_ans.npz', gamma)
        z0 = op.scm_forward(eps0.unsqueeze(0), op.mean_A()).squeeze(0); data = FateData(eps0, z0, pidx)
        env = CausalLatentEnv(op, data, max_steps=1, n_dag_samples=a.n_dag, fate_reward=R, reward_clip=clip)
        agent = SetPPO(state_dim=d, n_candidates=len(pool), k_max=kmax, device=a.device)
        print(f"\n########## {name}  (k={kmax} lam={lam} gamma={gamma} clip={clip}) ##########", flush=True)
        rews = []; picks = Counter(); early = Counter()
        for it in range(1, a.iters + 1):
            obs = env.reset(None); s_emb, t_emb = obs[:d], obs[d:]
            tokens, genes, logp, v = agent.select(s_emb, t_emb)
            gi = [data.candidate_idx[g] for g in genes]
            _, rw, _, info = env.step(gi); rews.append(rw)
            for g in genes: picks[i2g[data.candidate_idx[g]]] += 1
            if it <= 500:
                for g in genes: early[i2g[data.candidate_idx[g]]] += 1
            if len(agent._buf) >= a.update_every:
                st = agent.update()
                if it % (a.update_every * 5) == 0:
                    win = np.array(rews[-500:])
                    print(f"  it {it:5d} | reward mean {win.mean():+.2f} std {win.std():.2f} "
                          f"[{win.min():+.1f},{win.max():+.1f}] | ent {st.get('entropy',0):.2f} actor {st.get('actor_loss',0):+.3f}", flush=True)
        # diagnostics
        print(f"  [early picks it<=500] {[g for g,_ in early.most_common(6)]}", flush=True)
        print(f"  [all picks]           {[g for g,_ in picks.most_common(6)]}", flush=True)
        # recovery from the trained policy
        agent.policy.eval(); seen = {}
        for _ in range(1000):
            obs = env.reset(None); _, genes, _, _ = agent.select(obs[:d], obs[d:])
            gg = [data.candidate_idx[g] for g in genes]
            if gg: seen.setdefault(frozenset(gg), None)
        prop = set()
        for key in list(seen)[:40]: prop.update(i2g[i] for i in key)
        drate = len(prop & DRIVERS) / max(1, len(prop))
        print(f"  [RESULT {name}] final reward {np.mean(rews[-500:]):+.2f} | driver rate {drate:.2f} (base 0.03) "
              f"| P1∩{len(prop&P1)} P2∩{len(prop&P2)} drv∩{len(prop&DRIVERS)} | learned={'YES' if np.mean(rews[-500:])>np.mean(rews[:500])+0.5 else 'no'}", flush=True)


if __name__ == '__main__':
    main()
