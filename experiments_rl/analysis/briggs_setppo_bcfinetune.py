"""Briggs fate set-PPO: BC-init + PPO fine-tune. From-scratch PPO can't explore this design space
(swept: no clip/pessimism/k config learns). The fix = WARM-START from the planner (distill), then let
PPO refine. Steps: (1) run fate-MPC to get expert cocktails, (2) BC the SetPolicy to imitate them (= M3
distill), (3) continue PPO on the pessimistic fate reward (wide clip so the gradient survives). Reports
recovery at BC-init vs after fine-tune — does PPO improve on distill (0.47) or wander off?
Run via sbatch (CPU).
"""
import os, sys, argparse, numpy as np, scipy.sparse as sp, torch
RL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, RL)
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.briggs_fate_reward import FateReward
from core.env import CausalLatentEnv
from rl2_set_ppo.set_ppo import SetPPO
from analysis.briggs_fate_set_ppo import FateData, P1, P2, DRIVERS
from core.wblog import wb_init, wb_log, wb_summary, wb_finish
EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"


def mpc_beam(R, pidx, beam, kmax, topN):
    """fate-MPC beam over the pool -> top-N cocktails (expert demos), by r~."""
    singles = [[i] for i in pidx]
    r, _, _ = R.pessimistic(singles); order = np.argsort(-r.cpu().numpy())[:beam]
    beams = [(singles[o], float(r[o])) for o in order]; best = list(beams)
    for _ in range(2, kmax + 1):
        cand = []
        for bset, _ in beams:
            exp = [bset + [i] for i in pidx if i not in bset]
            rr, _, _ = R.pessimistic(exp); rr = rr.cpu().numpy()
            for o in np.argsort(-rr)[:beam]: cand.append((exp[o], float(rr[o])))
        seen = {}
        for s, rv in sorted(cand, key=lambda x: -x[1]):
            k = frozenset(s)
            if k not in seen: seen[k] = (s, rv)
            if len(seen) >= beam: break
        beams = list(seen.values()); best.extend(beams)
    dd = {frozenset(s): (s, rv) for s, rv in best}
    return [s for s, rv in sorted(dd.values(), key=lambda x: -x[1])][:topN]


def recovery(policy, env, R, data, i2g, d, n=2000):
    """SAMPLED recovery (matches the distill eval): sample cocktails, dedup, rank by r~, top-40 driver rate."""
    policy.eval(); seen = {}
    for _ in range(n):
        obs = env.reset(None); _, genes, _, _ = policy.sample(obs[:d], obs[d:])   # STOCHASTIC
        g = [data.candidate_idx[x] for x in genes]
        if g: seen.setdefault(frozenset(g), None)
    for key in list(seen):
        rr, _, _ = R.pessimistic([list(key)]); seen[key] = float(rr[0])
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:40]
    prop = set(); mahas = []
    for key, _ in ranked:
        prop.update(i2g[i] for i in key)
        mahas.append(float(R.mahalanobis(R.op.apply_sets(R.eps0, [list(key)], R.A_eval))[0]))  # off-manifold check
    return dict(driver=len(prop & DRIVERS) / max(1, len(prop)), P1=len(prop & P1), P2=len(prop & P2),
                drv=len(prop & DRIVERS), ncocktails=len(seen), maha=round(float(np.mean(mahas)), 2) if mahas else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{EXP20}/results/grid/v2b_deg3_t25_frozen_r32_all40/checkpoints/phase1_epoch40.pt')
    ap.add_argument('--kmax', type=int, default=7); ap.add_argument('--bc_epochs', type=int, default=500)
    ap.add_argument('--ft_iters', type=int, default=3000); ap.add_argument('--n_ctrl', type=int, default=256)
    ap.add_argument('--n_dag', type=int, default=6); ap.add_argument('--device', default='cpu')
    ap.add_argument('--readout', default=None, help='single briggs readout npz (clean bg); default = deg3_t25 min-ensemble')
    ap.add_argument('--support_feats', default=None); ap.add_argument('--ent_coeff', type=float, default=0.02)
    ap.add_argument('--gamma', type=float, default=2.0, help='Mahalanobis support penalty weight (0 disables)')
    ap.add_argument('--curve_csv', default=None, help='write per-update (reward, actor_loss, critic_loss, entropy) curve')
    ap.add_argument('--wandb', action='store_true'); ap.add_argument('--run_name', default=None)
    a = ap.parse_args()
    wb = wb_init(a.run_name or f'bcppo_g{a.gamma}_e{a.ent_coeff}', vars(a), tags=['briggs', 'bcppo']) if a.wandb else None
    TAG = a.ckpt.split('results/grid/')[1].split('/')[0]; RB = f'{EXP20}/results/briggs/readout_ridge_{TAG}'
    model, meta = load_frozen_wm(a.ckpt, EXP20, a.device); g2i = meta['gene_to_idx']; d = model.latent_dim
    op = WorldModelOperator(model, meta, n_dag_samples=a.n_dag)
    sys.path.insert(0, EXP20); from data.load_joung import load_joung
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt",
                    split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    _, _, eps = model.encoder(xc); eps0 = eps.mean(0)
    readouts = [a.readout] if a.readout else [f'{RB}_rho_cao.npz', f'{RB}_rho_briggs.npz']
    R = FateReward(op, readouts, eps0, lam=1.0, n_dag=a.n_dag)
    R.set_support(a.support_feats or f'{EXP20}/results/briggs/readout_feats_{TAG}_ans.npz', a.gamma)
    import csv
    reliable = set(open(V + "/effective_tf_genes.txt").read().split()); briggs = set()
    for r in csv.DictReader(open(f'{EXP20}/results/briggs/observed_combo_targets.csv')): briggs.update(r['combo'].split('|'))
    pool = sorted({g for g in (reliable | briggs) if g in g2i}); pidx = [g2i[g] for g in pool]; i2g = {g2i[g]: g for g in pool}
    z0 = op.scm_forward(eps0.unsqueeze(0), op.mean_A()).squeeze(0); data = FateData(eps0, z0, pidx)
    env = CausalLatentEnv(op, data, max_steps=1, n_dag_samples=a.n_dag, fate_reward=R, reward_clip=1000.0)  # WIDE clip
    cpos = {g: i for i, g in enumerate(pidx)}
    print(f"[bc+ft] pool={len(pool)} kmax={a.kmax} bc_epochs={a.bc_epochs} ft_iters={a.ft_iters}", flush=True)

    # (1) expert demos from fate-MPC
    demos = mpc_beam(R, pidx, beam=16, kmax=a.kmax, topN=40)
    print(f"[bc+ft] {len(demos)} MPC expert cocktails", flush=True)

    agent = SetPPO(state_dim=d, n_candidates=len(pool), k_max=a.kmax, entropy_coeff=a.ent_coeff, device=a.device)
    print(f"[bc+ft] ent_coeff={a.ent_coeff} readout={'briggs-only' if a.readout else 'min-ensemble'}", flush=True)
    obs = env.reset(None); S = obs[:d].unsqueeze(0).expand(len(demos), -1); T = obs[d:].unsqueeze(0).expand(len(demos), -1)
    TOK = torch.tensor([[cpos[g] + 1 for g in s][:a.kmax] + [0] * (a.kmax - len(s)) for s in demos], dtype=torch.long, device=a.device)

    # (2) BC-init (= M3 distill)
    opt_bc = torch.optim.Adam(agent.policy.parameters(), 1e-3)
    for ep in range(a.bc_epochs):
        logp, ent = agent.policy.evaluate(S, T, TOK); loss = -logp.mean() - 0.001 * ent.mean()
        opt_bc.zero_grad(); loss.backward(); opt_bc.step()
    rec_bc = recovery(agent.policy, env, R, data, i2g, d)
    print(f"[BC-init (=distill)] driver rate {rec_bc['driver']:.2f} | P1∩{rec_bc['P1']} P2∩{rec_bc['P2']} drv∩{rec_bc['drv']} | {rec_bc['ncocktails']} distinct | maha {rec_bc['maha']}", flush=True)

    # (3) PPO fine-tune from the BC'd policy
    r0 = []; curve = []
    for it in range(1, a.ft_iters + 1):
        obs = env.reset(None); tokens, genes, logp, v = agent.select(obs[:d], obs[d:])
        gi = [data.candidate_idx[g] for g in genes]
        _, rw, _, _ = env.step(gi); agent.store(obs[:d], obs[d:], tokens, rw, logp, v); r0.append(rw)
        if len(agent._buf) >= 128:
            st = agent.update()
            curve.append((it, float(np.mean(r0[-128:])), st.get('actor_loss',0.0), st.get('critic_loss',0.0), st.get('entropy',0.0), st.get('explained_variance',0.0), st.get('reward_std',0.0), st.get('approx_kl',0.0), st.get('clip_frac',0.0)))
            wb_log(wb, {'ppo/reward': float(np.mean(r0[-128:])), 'ppo/actor_loss': st.get('actor_loss', 0.0),
                        'ppo/critic_loss': st.get('critic_loss', 0.0), 'ppo/entropy': st.get('entropy', 0.0),
                        'ppo/explained_variance': st.get('explained_variance', 0.0), 'ppo/reward_std': st.get('reward_std', 0.0),
                        'ppo/approx_kl': st.get('approx_kl', 0.0), 'ppo/clip_frac': st.get('clip_frac', 0.0)}, step=it)
            if it % 640 == 0:
                print(f"  ft it {it:5d} reward {np.mean(r0[-500:]):+.2f} ent {st.get('entropy',0):.2f}", flush=True)
    if getattr(a, 'curve_csv', None):
        import csv as _csv
        with open(a.curve_csv, 'w', newline='') as f:
            w = _csv.writer(f); w.writerow(['iter','reward','actor_loss','critic_loss','entropy','explained_variance','reward_std','approx_kl','clip_frac']); w.writerows(curve)
        print(f"[curve] wrote {a.curve_csv} ({len(curve)} updates)", flush=True)
    rec_ft = recovery(agent.policy, env, R, data, i2g, d)
    print(f"\n[POST-FINETUNE] driver rate {rec_ft['driver']:.2f} | P1∩{rec_ft['P1']} P2∩{rec_ft['P2']} drv∩{rec_ft['drv']} | {rec_ft['ncocktails']} distinct | maha {rec_ft['maha']}", flush=True)
    print(f"[PARETO ent={a.ent_coeff} gamma={a.gamma}] distill(div {rec_bc['ncocktails']}, drv {rec_bc['driver']:.2f}) -> "
          f"BC+PPO(div {rec_ft['ncocktails']}, drv {rec_ft['driver']:.2f})", flush=True)
    print(f"[VERDICT] distill {rec_bc['driver']:.2f} -> BC+PPO {rec_ft['driver']:.2f}  "
          f"({'IMPROVED' if rec_ft['driver'] > rec_bc['driver'] + 0.02 else 'no gain / degraded'})", flush=True)
    wb_summary(wb, {'bc/driver_rate': rec_bc['driver'], 'bc/P1': rec_bc['P1'], 'bc/P2': rec_bc['P2'], 'bc/distinct': rec_bc['ncocktails'], 'bc/maha': rec_bc['maha'],
                    'ft/driver_rate': rec_ft['driver'], 'ft/P1': rec_ft['P1'], 'ft/P2': rec_ft['P2'], 'ft/distinct': rec_ft['ncocktails'], 'ft/maha': rec_ft['maha']})
    wb_finish(wb)


if __name__ == '__main__':
    main()
