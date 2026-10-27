"""M0 fate-objective MPC over the frozen exp20 WM — beam search for the k<=kmax TF cocktail maximizing the
merged Briggs fate reward r~ = Ridge_fate(z_post) - lambda*u (u = DAG-ensemble std [+ interaction blowup]).
Tiered pool = reliable-495 UNION Briggs-388 on-axis (zero-shot for non-reliable), so known drivers
(GATA2/TAL1/ERG/...) are reachable. Reports the top cocktails + recovery signal: overlap with P1/P2 members
and known HSPC drivers, enrichment of proposals vs random, and how pessimism (lambda) reshapes picks.
Run via sbatch (CPU). Args: --ckpt --readout --kmax --beam --lam --support_beta.
"""
import os, sys, argparse, numpy as np, scipy.sparse as sp, torch
RL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, RL)
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.briggs_fate_reward import FateReward
from core.wblog import wb_init, wb_summary, wb_finish
torch.set_grad_enabled(False)
EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
P1 = {'TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2'}
P2 = {'DDIT3', 'MZF1', 'HOXB3', 'MIER1', 'SIRT6', 'ZNF85'}
DRIVERS = {'ETV2', 'GATA2', 'RUNX1', 'LMO2', 'TAL1', 'LYL1', 'GFI1B', 'SPI1', 'FLI1', 'ERG', 'MEIS1',
           'HOXA9', 'GATA1', 'MYB', 'HHEX', 'RUNX3', 'IKZF1', 'CEBPA', 'KLF1', 'MECOM', 'LMO4', 'MYB'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{EXP20}/results/grid/v2b_deg3_t25_frozen_r32_s1/checkpoints/phase1_epoch40.pt')
    RB = f'{EXP20}/results/briggs/readout_ridge_v2b_deg3_t25_frozen_r32_s1'
    ap.add_argument('--readout_cao', default=f'{RB}_rho_cao.npz')
    ap.add_argument('--readout_briggs', default=f'{RB}_rho_briggs.npz')
    ap.add_argument('--ensemble', type=int, default=1, help='1=min over both readouts (cross-readout pessimism), 0=Cao only')
    ap.add_argument('--kmax', type=int, default=7); ap.add_argument('--beam', type=int, default=12)
    ap.add_argument('--lam', type=float, default=1.0); ap.add_argument('--support_beta', type=float, default=0.0)
    ap.add_argument('--support_gamma', type=float, default=0.0, help='Mahalanobis off-manifold penalty (MOReL/MOPO)')
    ap.add_argument('--support_feats', default=f'{EXP20}/results/briggs/readout_feats_v2b_deg3_t25_frozen_r32_s1.npz')
    ap.add_argument('--exclude_tfs', default='', help='leave-hit-out: TFs dropped from support manifold (leak-free recovery test)')
    ap.add_argument('--n_ctrl', type=int, default=256); ap.add_argument('--n_dag', type=int, default=8); ap.add_argument('--device', default='cpu')
    ap.add_argument('--topN', type=int, default=15); ap.add_argument('--div_jaccard', type=float, default=1.0,
                    help='<1.0 enables diversity filter: skip a cocktail if Jaccard>=thr to an already-selected one')
    ap.add_argument('--out', default=None, help='write candidate CSV (de-novo mode)')
    ap.add_argument('--wandb', action='store_true'); ap.add_argument('--run_name', default=None)
    a = ap.parse_args()

    model, meta = load_frozen_wm(a.ckpt, EXP20, a.device)
    op = WorldModelOperator(model, meta, n_dag_samples=a.n_dag)
    g2i = meta['gene_to_idx']
    sys.path.insert(0, EXP20); from data.load_joung import load_joung
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None,
                    effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    _, _, eps = model.encoder(xc); eps0 = eps.mean(0)
    readouts = [a.readout_cao, a.readout_briggs] if a.ensemble else [a.readout_cao]
    R = FateReward(op, readouts, eps0, lam=a.lam, n_dag=a.n_dag, support_beta=a.support_beta)
    excl = [t for t in a.exclude_tfs.split(',') if t]
    R.set_support(a.support_feats, a.support_gamma, exclude_tfs=excl or None)   # always load (maha for reporting); penalty active iff gamma>0
    print(f"[support] Mahalanobis loaded; penalty gamma={a.support_gamma}; leave-hit-out TFs={len(excl)}", flush=True)

    # tiered pool: reliable-495 UNION Briggs-388 on-axis
    reliable = set(open(V + "/effective_tf_genes.txt").read().split())
    briggs_tfs = set()
    bt = f'{EXP20}/results/briggs/observed_combo_targets.csv'
    import csv
    for r in csv.DictReader(open(bt)):
        briggs_tfs.update(r['combo'].split('|'))
    pool = sorted({g for g in (reliable | briggs_tfs) if g in g2i})
    pidx = [g2i[g] for g in pool]; i2g = {g2i[g]: g for g in pool}
    A_batch = op.sample_dags(a.n_dag)
    print(f"[fate-MPC] pool={len(pool)} (reliable∩axis {len({g for g in reliable if g in g2i})}, "
          f"briggs∩axis {len({g for g in briggs_tfs if g in g2i})}) | kmax={a.kmax} beam={a.beam} lam={a.lam} sb={a.support_beta}", flush=True)

    def pess(sets):
        r, base, u = R.pessimistic(sets, A_batch=A_batch); return r.cpu().numpy(), base.cpu().numpy(), u.cpu().numpy()

    # beam search
    singles = [[i] for i in pidx]
    r, base, u = pess(singles)
    order = np.argsort(-r)[:a.beam]
    beams = [(singles[o], float(r[o]), float(base[o])) for o in order]
    best = list(beams)
    for depth in range(2, a.kmax + 1):
        cand = []
        for bset, _, _ in beams:
            exp = [bset + [i] for i in pidx if i not in bset]
            rr, bb, uu = pess(exp)
            for j, o in enumerate(np.argsort(-rr)[:a.beam]):
                cand.append((exp[o], float(rr[o]), float(bb[o])))
        # dedup by frozenset, keep top-beam
        seen = {}
        for s, rv, bv in sorted(cand, key=lambda x: -x[1]):
            key = frozenset(s)
            if key not in seen: seen[key] = (s, rv, bv)
            if len(seen) >= a.beam: break
        beams = list(seen.values()); best.extend(beams)
        top = beams[0]
        print(f"  depth {depth}: best {[i2g[i] for i in top[0]]}  r~ {top[1]:+.3f} fate {top[2]:+.3f}", flush=True)

    best = sorted({frozenset(s): (s, rv, bv) for s, rv, bv in best}.values(), key=lambda x: -x[1])
    # novelty: is the EXACT cocktail among Briggs's observed combos?
    observed = set()
    for r in csv.DictReader(open(bt)):
        observed.add(frozenset(r['combo'].split('|')))
    # optional diversity filter (greedy: skip if too similar to an already-selected cocktail)
    picked = []
    for s, rv, bv in best:
        names = frozenset(i2g[i] for i in s)
        if a.div_jaccard < 1.0 and any(len(names & p) / len(names | p) >= a.div_jaccard for p, _, _, _ in picked):
            continue
        picked.append((names, s, rv, bv))
        if len(picked) >= a.topN: break

    rows = []
    print(f"\n[TOP {a.topN} cocktails by r~]  (cao/brg=per-readout; maha=support dist; * = novel vs Briggs)", flush=True)
    for names, s, rv, bv in picked:
        z = op.apply_sets(eps0, [s], R.A_eval)
        pr = R._fate_all(z)[:, 0].cpu().numpy(); mh = float(R.mahalanobis(z)[0])
        nm = sorted(names); novel = frozenset(names) not in observed
        ov1 = len(names & P1); ov2 = len(names & P2); dv = len(names & DRIVERS)
        brg = float(pr[1]) if len(pr) > 1 else float('nan')
        tag = (f" P1∩{ov1}" if ov1 else "") + (f" P2∩{ov2}" if ov2 else "") + (f" drv∩{dv}" if dv else "")
        print(f"  {'*' if novel else ' '} k={len(s)} r~ {rv:+.3f} [cao {pr[0]:+.2f} brg {brg:+.2f} maha {mh:.2f}] | {nm}{tag}", flush=True)
        rows.append(dict(combo='|'.join(nm), k=len(s), r_pess=round(rv, 4), fate_cao=round(float(pr[0]), 4),
                         fate_briggs=round(brg, 4), maha=round(mh, 3), novel=int(novel),
                         P1_overlap=ov1, P2_overlap=ov2, driver_overlap=dv))
    if a.out:
        import os as _os; _os.makedirs(_os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"wrote {a.out}", flush=True)

    # recovery: driver enrichment among proposed TFs vs pool base rate
    proposed = set()
    for names, s, _, _ in picked:
        proposed.update(names)
    base_rate = len([g for g in pool if g in DRIVERS]) / len(pool)
    prop_rate = len(proposed & DRIVERS) / max(1, len(proposed))
    print(f"\n[recovery] proposed TFs={len(proposed)} | driver rate {prop_rate:.2f} vs pool base {base_rate:.2f}", flush=True)
    print(f"  P1 members recovered: {sorted(proposed & P1)}", flush=True)
    print(f"  P2 members recovered: {sorted(proposed & P2)}", flush=True)
    print(f"  drivers recovered:    {sorted(proposed & DRIVERS)}", flush=True)
    if a.wandb:
        wb = wb_init(a.run_name or f'briggs_mpc_k{a.kmax}_g{a.support_gamma}', vars(a), tags=['briggs', 'mpc'])
        wb_summary(wb, {'recovery/driver_rate': prop_rate, 'recovery/base_rate': base_rate,
                        'recovery/P1_recovered': len(proposed & P1), 'recovery/P2_recovered': len(proposed & P2),
                        'recovery/drivers_recovered': len(proposed & DRIVERS), 'recovery/proposed_tfs': len(proposed)})
        wb_finish(wb)
    print("done", flush=True)


if __name__ == '__main__':
    main()
