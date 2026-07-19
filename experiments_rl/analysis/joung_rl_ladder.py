"""Joung-RL value-add ladder — ALL planners + baselines on the fetal-lineage fate benchmark, ONE uniform
eval so every cell of the table is computed (recall@1, recall@3, resid_match, driver-hit, held-out-driver).
Planners: MPC (beam), MPC-greedy (k2), MPC-greedy (k1), set-PPO, MPC-distill (BC of MPC), seq-PPO.
Baselines: random k-sets (avg), 1-driver oracle. In-distribution test: does the WM+RL recover the TFs the
Cao/Joung atlas says drive each fate? Run via scripts/joung_ladder.sbatch.
"""
import os, sys, argparse, numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import add_stack_args, build_stack
from core.scoring import torch_match, Lineage
from core.env import CausalLatentEnv
from analysis.fate_rl import fate_mpc, pred_resid, TFLIN
from rl2_set_ppo.set_ppo import SetPPO
from rl2_set_ppo.set_policy import SetPolicy
from rl2_set_ppo.train import train as train_set, make_planner as planner_set
from rl1_seq_ppo.ppo import PPO
from rl1_seq_ppo.train import train as train_seq


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--pool', default='curated'); ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--qmax', type=float, default=0.15); ap.add_argument('--min_drivers', type=int, default=2)
    ap.add_argument('--iters', type=int, default=4000); ap.add_argument('--device', default=None)
    ap.add_argument('--lineage_csv', default=TFLIN, help='TF->target CSV; pass a per-celltype CSV for the finer benchmark')
    a = ap.parse_args()
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, meta, op, data = build_stack(a, dev, n_dag=8); data.set_pool(a.pool)
    data.build_lineage_targets(a.lineage_csv, qmax=a.qmax, min_drivers=a.min_drivers)
    keys = data.lineage_keys; names = [k.name for k in keys]; K = len(keys)
    A = op.mean_A(); ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]
    drift = torch.tensor(np.mean([data.de_single[g]['lfc'] for g in data.single_split['train']], 0), dtype=torch.float32, device=dev)
    cent = torch.stack([torch.tensor(data.de_lineage[n]['lfc'], dtype=torch.float32, device=dev) - drift for n in names])
    centn = cent / (cent.norm(dim=1, keepdim=True) + 1e-9)
    thr = data.thresh; cand = data.candidate_idx; gn = data.gene_names; g2i = data.gene_to_idx
    rng = np.random.default_rng(0)
    print(f"[ladder] {K} lineages, pool={a.pool} |cand|={len(cand)}", flush=True)

    def metrics(planner_idx, n_rep=1):
        """planner_idx: li -> list of gene indices. n_rep>1 averages (for stochastic baselines)."""
        acc = np.zeros(5)
        for _ in range(n_rep):
            r1 = r3 = rm = dh = hd = 0.0
            for li, nm in enumerate(names):
                gidx = [g for g in planner_idx(li) if 0 <= g < len(gn)]
                if not gidx:
                    continue
                rv = pred_resid(op, data, gidx, drift, ctrl_recon, A)
                order = torch.argsort(-(centn @ (rv / (rv.norm() + 1e-9)))).tolist()
                r1 += order[0] == li; r3 += li in order[:3]
                o = cent[li]; sel = torch.where(o.abs() > thr)[0]; rm += float(torch_match(rv.unsqueeze(0), o[sel], sel)[0])
                gset = {gn[i] for i in gidx}; drv = data.lineage_drivers[nm]
                dh += len(gset & set(drv['train'] + drv['test'])) > 0
                hd += len(gset & set(drv['test'])) > 0
            acc += np.array([r1, r3, rm, dh, hd])
        acc /= (n_rep * K)
        return dict(recall1=acc[0], recall3=acc[1], resid=acc[2], drv_hit=acc[3], held=acc[4])

    def mpc_p(beam, maxset):
        def p(li):
            o = cent[li]; sel = torch.where(o.abs() > thr)[0]
            chosen, _ = fate_mpc(op, data, cand, o[sel], sel, drift, ctrl_recon, A, beam=beam, max_set=maxset)
            return chosen
        return p

    rows = {}
    rows['MPC (beam8, k2)'] = metrics(mpc_p(a.beam, 2))
    rows['MPC-greedy (k2)'] = metrics(mpc_p(1, 2))
    rows['MPC-greedy (k1)'] = metrics(mpc_p(1, 1))
    print("[ladder] MPC variants done", flush=True)

    # --- set-PPO (trained on lineage centroids) ---
    try:
        env = CausalLatentEnv(op, data, max_steps=1, reward_mode='resid', n_dag_samples=8)
        agent = SetPPO(state_dim=op.d, n_candidates=len(cand), k_max=2, device=dev)
        train_set(op, data, env, agent, train_keys=keys, n_iters=a.iters, rng=np.random.default_rng(0),
                  eval_planner=None, eval_keys=None)
        sp = planner_set(agent, op, data, env)
        rows['set-PPO'] = metrics(lambda li: sp(keys[li]))
        print("[ladder] set-PPO done", flush=True)
    except Exception as e:
        print(f"[ladder] set-PPO FAILED: {e}", flush=True)

    # --- MPC-distill (BC the MPC-beam picks into a SetPolicy) ---
    try:
        env2 = CausalLatentEnv(op, data, max_steps=1, reward_mode='resid', n_dag_samples=8)
        cpos = {g: i for i, g in enumerate(cand)}; beam_p = mpc_p(a.beam, 2)
        Ss, Ts, TOKs = [], [], []
        for li in range(K):
            obs = env2.reset(keys[li]); Ss.append(obs[:op.d]); Ts.append(obs[op.d:])
            picks = [cpos[g] for g in beam_p(li) if g in cpos][:2]
            TOKs.append([p + 1 for p in picks] + [0] * (2 - len(picks)))
        S = torch.stack(Ss); T = torch.stack(Ts); TOK = torch.tensor(TOKs, dtype=torch.long, device=dev)
        pol = SetPolicy(op.d, len(cand), k_max=2).to(dev); opt = torch.optim.Adam(pol.parameters(), 1e-3)
        for _ in range(400):
            logp, ent = pol.evaluate(S, T, TOK); loss = -logp.mean() - 0.001 * ent.mean()
            opt.zero_grad(); loss.backward(); opt.step()

        @torch.no_grad()
        def distill_p(li):
            obs = env2.reset(keys[li]); s, tg = obs[:op.d], obs[op.d:]
            h = pol.encode(s.unsqueeze(0), tg.unsqueeze(0)); ctx = torch.zeros(1, pol.emb_dim, device=dev)
            chosen = torch.zeros(1, pol.C, dtype=torch.bool, device=dev); out = []
            for i in range(pol.k_max):
                tok = int(pol._slot_logits(h, ctx, i, chosen).argmax())
                if tok == 0: break
                g = tok - 1; out.append(cand[g]); chosen[0, g] = True
                ctx = ctx + pol.gene_emb(torch.tensor([g], device=dev))
            return out
        rows['MPC-distill (BC)'] = metrics(distill_p)
        print("[ladder] MPC-distill done", flush=True)
    except Exception as e:
        print(f"[ladder] MPC-distill FAILED: {e}", flush=True)

    # --- BC-init + PPO fine-tune (BC the MPC picks, then PPO refine; higher entropy for diversity) ---
    try:
        env4 = CausalLatentEnv(op, data, max_steps=1, reward_mode='resid', n_dag_samples=8, reward_clip=1000.0)
        agent4 = SetPPO(state_dim=op.d, n_candidates=len(cand), k_max=2, entropy_coeff=0.05, device=dev)
        cpos = {g: i for i, g in enumerate(cand)}; beam_p = mpc_p(a.beam, 2)
        Ss, Ts, TOKs = [], [], []
        for li in range(K):
            obs = env4.reset(keys[li]); Ss.append(obs[:op.d]); Ts.append(obs[op.d:])
            picks = [cpos[g] for g in beam_p(li) if g in cpos][:2]
            TOKs.append([p + 1 for p in picks] + [0] * (2 - len(picks)))
        S = torch.stack(Ss); T = torch.stack(Ts); TOK = torch.tensor(TOKs, dtype=torch.long, device=dev)
        optbc = torch.optim.Adam(agent4.policy.parameters(), 1e-3)
        for _ in range(400):                                              # BC-init on per-lineage MPC picks
            logp, ent = agent4.policy.evaluate(S, T, TOK); loss = -logp.mean() - 0.001 * ent.mean()
            optbc.zero_grad(); loss.backward(); optbc.step()
        rng2 = np.random.default_rng(1)
        for it in range(a.iters):                                         # PPO fine-tune across lineages
            li = int(rng2.integers(K)); obs = env4.reset(keys[li])
            tokens, genes, logp, v = agent4.select(obs[:op.d], obs[op.d:])
            gi = [cand[g] for g in genes]; _, rw, _, _ = env4.step(gi)
            agent4.store(obs[:op.d], obs[op.d:], tokens, rw, logp, v)
            if len(agent4._buf) >= 128: agent4.update()
        sp4 = planner_set(agent4, op, data, env4)
        rows['BC-init+PPO'] = metrics(lambda li: sp4(keys[li]))
        print("[ladder] BC-init+PPO done", flush=True)
    except Exception as e:
        print(f"[ladder] BC-init+PPO FAILED: {e}", flush=True)

    # --- seq-PPO (trained; forced 2-step greedy rollout for lineage targets) ---
    try:
        envq = CausalLatentEnv(op, data, max_steps=2, reward_mode='resid', n_dag_samples=8)
        ppo = PPO(state_dim=op.d, n_actions=len(cand), device=dev)
        train_seq(op, data, envq, ppo, keys, n_iters=a.iters, rng=np.random.default_rng(0))

        @torch.no_grad()
        def seq_p(li):
            envq.max_steps = 2; obs = envq.reset(keys[li]); ca, cg = [], []
            for _ in range(2):
                s, tg = obs[:op.d], obs[op.d:]
                logits = ppo.actor.forward(s.unsqueeze(0), tg.unsqueeze(0)).logits.squeeze(0).clone()
                for x in ca: logits[x] = -1e9
                x = int(logits.argmax()); ca.append(x); cg.append(cand[x])
                obs, _, done, _ = envq.step([cand[x]])
                if done: break
            return cg
        rows['seq-PPO (k2)'] = metrics(seq_p)
        print("[ladder] seq-PPO done", flush=True)
    except Exception as e:
        print(f"[ladder] seq-PPO FAILED: {e}", flush=True)

    # --- baselines (complete metrics) ---
    rows['random (k2)'] = metrics(lambda li: [int(x) for x in rng.choice(cand, 2, replace=False)], n_rep=200)
    rows['1-driver oracle'] = metrics(lambda li: [g2i[data.lineage_drivers[names[li]]['train'][0]]])
    rows['chance'] = dict(recall1=1 / K, recall3=3 / K, resid=0.0, drv_hit=float('nan'), held=float('nan'))

    df = pd.DataFrame(rows).T[['recall1', 'recall3', 'resid', 'drv_hit', 'held']]
    df.columns = ['recall@1', 'recall@3', 'resid_match', 'driver_hit_frac', 'held_driver_frac']
    print(f"\n{'='*88}\nJOUNG-RL VALUE-ADD LADDER — {K} fetal lineages, pool={a.pool}, in-distribution", flush=True)
    print(df.round(3).to_string(), flush=True)
    df.round(4).to_csv(f'{HERE}/joung_rl_ladder.csv')
    print(f"\nsaved -> joung_rl_ladder.csv", flush=True)


if __name__ == '__main__':
    main()
