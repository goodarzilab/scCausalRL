"""Fate-target RL on the Joung WM: instead of "reach held-out combo #N", the goal is "reach
developmental LINEAGE L" (Leiden cluster over Cao cell types). Target = residual centroid of L's
confident driver TFs. Planner = MPC (the proven method) over the curated pool, scoring the
DRIFT-REMOVED match to the centroid (the only axis with fate signal). Eval is fate-native:
  resid_match     signed-Pearson(chosen-set Δ−drift, L centroid−drift) on the centroid's DEGs
  nearest-lineage of K centroids, is L the planner's argmax (recall@1) / top-3 (recall@3)
  driver_recovery did the planner pick TFs the atlas annotates to L (incl. HELD-OUT drivers)?
Baselines: random curated sets; single best-driver oracle. Run via scripts/fate_rl.sbatch.
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import add_stack_args, build_stack
from core.scoring import torch_match, Lineage
TFLIN = ('/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/analysis/'
         'human_dev_annotation/joung_tf_lineage.csv')
CHUNK = 512


def decode_delta(op, z, ctrl_recon):
    out = []
    for c in z.split(CHUNK):
        out.append(op.decode(c) - ctrl_recon)
    return torch.cat(out)


def fate_mpc(op, data, cand_idx, o_t, sel_idx, drift, ctrl_recon, A, beam=8, max_set=2):
    """Beam MPC maximizing resid-match to a lineage centroid. Returns chosen gene indices."""
    S_all = op.shift_for(cand_idx)                                   # [C,d]
    C = S_all.shape[0]; eps0 = data.eps_ctrl_mean

    def match(S_batch):
        z = op.apply_many(eps0, S_batch, A)                          # [K,d]
        return torch_match(decode_delta(op, z, ctrl_recon) - drift, o_t, sel_idx)
    m1 = match(S_all)                                                # [C]
    top = torch.topk(m1, min(beam, C)).indices.tolist()
    finalists = [([top[0]], float(m1[top[0]]))]
    if max_set >= 2 and C >= 2:
        for bi in top:
            m2 = match(S_all[bi].unsqueeze(0) + S_all)               # [C]
            m2[bi] = -1e9
            j = int(torch.argmax(m2)); finalists.append(([bi, j], float(m2[j])))
    best = max(finalists, key=lambda x: x[1])
    return [cand_idx[i] for i in best[0]], best[1]


def pred_resid(op, data, gene_idx, drift, ctrl_recon, A):
    """resid Δ [G] for a gene set."""
    S = op.shift_for(gene_idx).sum(0, keepdim=True)
    z = op.apply_many(data.eps_ctrl_mean, S, A)
    return (decode_delta(op, z, ctrl_recon)[0] - drift)


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--pool', default='curated'); ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--qmax', type=float, default=0.10); ap.add_argument('--device', default=None)
    ap.add_argument('--min_drivers', type=int, default=4, help='min confident drivers for a lineage to be a target (lower => more lineages)')
    a = ap.parse_args()
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, meta, op, data = build_stack(a, dev, n_dag=8); data.set_pool(a.pool)
    data.build_lineage_targets(TFLIN, qmax=a.qmax, min_drivers=a.min_drivers)
    keys = data.lineage_keys; K = len(keys); names = [k.name for k in keys]
    A = op.mean_A(); ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]
    drift = torch.tensor(np.mean([data.de_single[g]['lfc'] for g in data.single_split['train']], 0),
                         dtype=torch.float32, device=dev)
    thr = data.thresh; cand = data.candidate_idx

    # lineage centroid residuals [K,G] + per-lineage DEG selection (for nearest-lineage + MPC target)
    cent = torch.stack([torch.tensor(data.de_lineage[n]['lfc'], dtype=torch.float32, device=dev) - drift
                        for n in names])                              # [K,G]
    centn = cent / (cent.norm(dim=1, keepdim=True) + 1e-9)
    rng = np.random.default_rng(0)

    def nearest(resid_vec):
        cs = (centn @ (resid_vec / (resid_vec.norm() + 1e-9)))        # [K]
        order = torch.argsort(-cs).tolist()
        return order

    rows = []
    for li, (k, name) in enumerate(zip(keys, names)):
        o_full = cent[li]; sel = torch.where(o_full.abs() > thr)[0]
        o_t = o_full[sel]
        chosen, mscore = fate_mpc(op, data, cand, o_t, sel, drift, ctrl_recon, A, beam=a.beam)
        rv = pred_resid(op, data, chosen, drift, ctrl_recon, A)
        order = nearest(rv); r1 = order[0] == li; r3 = li in order[:3]
        gset = {data.gene_names[i] for i in chosen}
        drv = data.lineage_drivers[name]
        rec = len(gset & set(drv['train'] + drv['test'])); rec_held = len(gset & set(drv['test']))
        rows.append({'lineage': name.split(':')[0] + ':' + name.split(':')[1][:14], 'n_drv': len(drv['train']) + len(drv['test']),
                     'chosen': ','.join(sorted(gset)), 'resid_match': round(mscore, 3),
                     'near@1': int(r1), 'near@3': int(r3), 'drv_hit': rec, 'held_drv_hit': rec_held})

    # baselines: random curated sets (mean over 200), single best-driver oracle
    rand_r1 = rand_match = 0.0; NR = 200
    for _ in range(NR):
        li = rng.integers(K); name = names[li]
        gset = [int(x) for x in rng.choice(cand, size=rng.integers(1, 3), replace=False)]
        rv = pred_resid(op, data, gset, drift, ctrl_recon, A)
        rand_r1 += nearest(rv)[0] == li
        o_full = cent[li]; sel = torch.where(o_full.abs() > thr)[0]
        rand_match += float(torch_match(rv.unsqueeze(0), o_full[sel], sel)[0])
    orc_r1 = 0
    for li, name in enumerate(names):
        g = data.lineage_drivers[name]['train'][0]
        rv = pred_resid(op, data, [data.gene_to_idx[g]], drift, ctrl_recon, A)
        orc_r1 += nearest(rv)[0] == li

    df = pd.DataFrame(rows)
    print(f"\n{'='*100}\nFATE-TARGET MPC — {K} lineages, pool={a.pool} (|cand|={len(cand)}), drift-removed reward")
    print(df.to_string(index=False))
    print(f"\nMPC      : nearest-lineage recall@1={df['near@1'].mean():.2f}  recall@3={df['near@3'].mean():.2f}  "
          f"mean resid_match={df['resid_match'].mean():.3f}  driver-hit lineages={ (df['drv_hit']>0).sum() }/{K} "
          f"(held-out drivers hit in { (df['held_drv_hit']>0).sum() }/{K})")
    print(f"random   : nearest-lineage recall@1={rand_r1/NR:.2f}  mean resid_match={rand_match/NR:.3f}  (chance@1={1/K:.2f})")
    print(f"1-driver : nearest-lineage recall@1={orc_r1/K:.2f}  (single known driver TF, the per-TF ceiling)")
    df.to_csv(f'{HERE}/fate_rl_mpc.csv', index=False); print(f"saved -> fate_rl_mpc.csv")


if __name__ == '__main__':
    main()
