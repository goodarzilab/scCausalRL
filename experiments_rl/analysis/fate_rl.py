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


def fate_mpc(op, data, cand_idx, o_t, sel_idx, drift, ctrl_recon, A, beam=8, max_set=2, eps0=None, return_ranked=False, return_sets=False):
    """Beam MPC (greedy-forward) maximizing resid-match to a lineage centroid, for ANY max_set (k). Keeps a beam of
    partial sets; at each of the k steps, extends every beam member by every candidate and prunes to `beam`. Returns
    the best set of size <= max_set. Cost grows ~ O(max_set · beam · C) forward solves — the k-sweep crossover knob.
    `eps0` overrides the starting control latent (default = Joung data.eps_ctrl_mean) — pass a Briggs-frame control to
    plan from the Briggs frame; pair with the matching `ctrl_recon` so the predicted Δ is in the same frame.
    `return_ranked` (EVAL-ONLY breadth-match): also return a top-K ranked candidate shortlist so a freq-ranked
    driver-recall gives MPC the SAME budget as the policy's best-of-N. Default is the original 2-tuple return, so
    every other caller (mpc_p, build_demos, briggs_frame, joung_rl_ladder) is unchanged."""
    S_all = op.shift_for(cand_idx)                                   # [C,d]
    C = S_all.shape[0]; eps0 = data.eps_ctrl_mean if eps0 is None else eps0

    def match(S_batch):
        z = op.apply_many(eps0, S_batch, A)                          # [M,d]
        return torch_match(decode_delta(op, z, ctrl_recon) - drift, o_t, sel_idx)
    m1 = match(S_all)                                                # [C]
    seen = {i: float(m1[i]) for i in range(C)}                       # best model-score each candidate ever attained (for the ranked shortlist)
    top = torch.topk(m1, min(beam, C)).indices.tolist()
    beams = [([i], S_all[i].clone(), float(m1[i])) for i in top]     # (gene idxs, cumulative shift, score)
    best = max(beams, key=lambda b: b[2])
    for _ in range(2, int(max_set) + 1):                             # grow the set one TF at a time
        ext = []
        for idxs, cur, _ in beams:
            ms = match(cur.unsqueeze(0) + S_all)                     # [C] add each candidate to this partial set
            for i in idxs:
                ms[i] = -1e9                                         # no repeats
            for j in torch.topk(ms, min(beam, C)).indices.tolist():
                sj = float(ms[j])
                if sj > seen[j]: seen[j] = sj                        # track best-ever extension score per candidate
                ext.append((idxs + [j], cur + S_all[j], sj))
        if not ext:
            break
        ext.sort(key=lambda b: -b[2]); beams = ext[:beam]           # prune to beam width
        if beams[0][2] > best[2]:
            best = beams[0]
    chosen = [cand_idx[i] for i in best[0]]
    if return_ranked:
        rest = [i for i in sorted(seen, key=lambda k: -seen[k]) if i not in best[0]]
        ranked_pos = list(dict.fromkeys(list(best[0]) + rest))       # chosen best-set first, then remaining by best-ever score
        return chosen, best[2], [cand_idx[i] for i in ranked_pos]
    if return_sets:                                                  # beam-top-N: the final beam FRONTIER as N distinct complete cocktails (score-ranked) — the fair "search asked for N candidates" diversity baseline
        sets = [([cand_idx[i] for i in idxs], sc) for idxs, _, sc in sorted(beams, key=lambda b: -b[2])]
        return chosen, best[2], sets
    return chosen, best[2]


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
