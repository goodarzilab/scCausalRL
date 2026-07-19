"""Combinatorial (multi-TF) prediction for a v2b exp20 model, scored on the v2b∩combo gene
intersection (the Joung combo experiment GSE217066 measured only ~8.5k genes; ~2.8k overlap v2b).

For each held-out TF SET (pairs + triples from joung_combo_proc.h5ad), compare:
  • do_op    — structural shift-mutilate on the SET: intervene_shift_mutilate([g1,g2,..]) (training-coherent)
  • additive — the model's OWN sum of single-TF predicted Δ (no epistasis; the baseline to beat)
  • add_obs  — sum of OBSERVED single-TF Δ (de_single) — the oracle additive null
against the observed combo Δ. Metrics (drift-removed, on the intersection genes): disc_resid
(cosine discrimination among sets), pear (mean per-set Pearson), pear_noKO (excl. the set's own genes).

Run: python evaluation/combo_eval_v2b.py --checkpoint <ckpt> --tag <tag> [--min_cells 20 --n_ctrl 500]
"""
import os, sys, argparse, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp, anndata as ad
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung

VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = os.path.join(VDIR, "joung_singles_proc_v2b.h5ad"); EFF = os.path.join(VDIR, "effective_tf_genes.txt")
COMBO = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2/joung_combo_proc_v2b.h5ad"  # v2b file (4366 genes) = the file TRAINING uses; was exp17 joung_combo_proc.h5ad (2816-gene subset)
FILL = {'GFP', 'MCHERRY', 'EGFP', 'MECHERRY'}


def parse_tfs(tf_set):
    out = []
    for tok in str(tf_set).split(','):
        nm = tok.split('-')[-1].upper()
        if nm and nm not in FILL: out.append(nm)
    return list(dict.fromkeys(out))


def zrow(M):
    M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def disc(pred, obs):                       # cosine discrimination among sets (drift-removed upstream)
    C = zrow(pred) @ zrow(obs).T; d = np.diag(C)[:, None]; T = len(pred)
    return float(((d > C).sum(1) / (T - 1)).mean()) if T > 1 else float('nan')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', required=True); ap.add_argument('--tag', default='combo')
    ap.add_argument('--min_cells', type=int, default=20); ap.add_argument('--n_ctrl', type=int, default=500)
    ap.add_argument('--device', default='cpu'); a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)
    model, gn, ck = load_shift_model(a.checkpoint, dev)
    p1 = (ck.get('config', {}) or {}).get('phase1', {}) or {}
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); UMODE = p1.get('pert_mutilate_u_mode', 'shift')
    UMAX = float(p1.get('pert_mutilate_u_max', 0.5))
    gidx = {g: i for i, g in enumerate(gn)}; G = len(gn)

    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
    de = dd['de_single']
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=dev)
    _, _, eps = model.encoder(xc)
    A = model.dag.sample_A(1).squeeze(0); z_prev = model.dag.scm_forward(eps, A)
    pert_dec = getattr(model, 'pert_decoder', None) or model.decoder
    ctrl_recon = pert_dec(z_prev).mean(0).cpu().numpy()

    # ---- observed combo Δ on the v2b axis (reindex combo proc; intersection genes only) ----
    C = ad.read_h5ad(COMBO)
    Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
    Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4)          # lognorm CP10k
    cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
    inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter])
    mask = np.array([gidx[g] for g in inter])                            # v2b positions with combo data
    print(f"[combo] {len(inter)}/{G} v2b genes have combo data (intersection scored)", flush=True)
    tfset = C.obs['TF_set'].astype(str).values
    combo_ref = Xk.mean(0)                                               # reference (cancels in drift-removal)

    sets = []
    for ts in dict.fromkeys(tfset):
        tfs = parse_tfs(ts)
        tfs = [t for t in tfs if t in gidx]                             # constituent TFs must be in v2b vocab
        if len(tfs) < 2: continue
        m = tfset == ts
        if m.sum() < a.min_cells: continue
        obs_full = Xk[m].mean(0) - combo_ref                            # [combo_genes]
        obs = np.zeros(G, np.float32); obs[mask] = obs_full[col]        # onto v2b axis (intersection filled)
        sets.append(dict(name=ts, tfs=tfs, k=len(tfs), obs=obs))
    print(f"[combo] usable sets: {len(sets)} ({sum(s['k']==2 for s in sets)} pairs, {sum(s['k']>=3 for s in sets)} triples)", flush=True)

    def single_pred(g):                                                # model's single-TF predicted Δ (do-op)
        zp, _ = model.intervene_shift_mutilate(eps, [gidx[g]], A, z_prev, u_mode=UMODE, u_clip_max=UMAX, shift_beta=BETA)
        return pert_dec(zp).mean(0).cpu().numpy() - ctrl_recon

    spred = {}
    for s in sets:
        for g in s['tfs']:
            if g not in spred: spred[g] = single_pred(g)

    for s in sets:
        zp, _ = model.intervene_shift_mutilate(eps, [gidx[g] for g in s['tfs']], A, z_prev,
                                               u_mode=UMODE, u_clip_max=UMAX, shift_beta=BETA)
        s['do'] = pert_dec(zp).mean(0).cpu().numpy() - ctrl_recon                       # structural do-op
        s['add'] = np.sum([spred[g] for g in s['tfs']], axis=0)                         # model additive
        s['addobs'] = np.sum([np.asarray(de[g], np.float32) for g in s['tfs'] if g in de], axis=0) \
            if any(g in de for g in s['tfs']) else np.zeros(G, np.float32)              # observed additive null

    def score(key, subset):
        rows = [s for s in sets if subset(s)]
        if len(rows) < 2: return None
        P = np.stack([s[key][mask] for s in rows]); O = np.stack([s['obs'][mask] for s in rows])
        Pd = P - P.mean(0); Od = O - O.mean(0)                                          # drift-remove (mean over sets)
        d = disc(Pd, Od)
        pear = float(np.mean([np.corrcoef(Pd[i], Od[i])[0, 1] for i in range(len(rows))
                              if Pd[i].std() > 1e-9 and Od[i].std() > 1e-9]))
        # noKO: exclude the set's own constituent-gene columns
        pnk = []
        for i, s in enumerate(rows):
            keep = np.array([j for j, g in enumerate(inter) if g not in set(s['tfs'])])
            x, y = Pd[i][keep], Od[i][keep]
            if x.std() > 1e-9 and y.std() > 1e-9: pnk.append(np.corrcoef(x, y)[0, 1])
        return dict(n=len(rows), disc=round(d, 4), pear=round(pear, 4), pear_noKO=round(float(np.mean(pnk)), 4))

    def per_set_vals(key, rows):                       # per-set disc (frac beaten) + pearson, for error bars
        P = np.stack([s[key][mask] for s in rows]); O = np.stack([s['obs'][mask] for s in rows])
        Pd = zrow(P - P.mean(0)); Od = zrow(O - O.mean(0))
        Cmat = Pd @ Od.T; dg = np.diag(Cmat)[:, None]; T = len(rows)
        disc_ps = ((dg > Cmat).sum(1) / (T - 1)) if T > 1 else np.full(T, np.nan)
        pear_ps = np.array([np.corrcoef(Pd[i], Od[i])[0, 1] if Pd[i].std() > 1e-9 and Od[i].std() > 1e-9 else np.nan
                            for i in range(T)])
        return disc_ps.tolist(), pear_ps.tolist()

    out = {}
    for split, sub in [('pairs', lambda s: s['k'] == 2), ('triples', lambda s: s['k'] >= 3), ('all', lambda s: True)]:
        out[split] = {m: score(m, sub) for m in ('do', 'add', 'addobs')}
    # per-set values (pairs) for figure error bars
    prs = [s for s in sets if s['k'] == 2]
    out['pairs_per_set'] = {m: dict(zip(('disc', 'pear'), per_set_vals(m, prs))) for m in ('do', 'add', 'addobs')}

    # ---- epistasis quartiles (pairs; E = ||obs - obsadd|| / ||obs|| on intersection; Q4 = most non-additive) ----
    E = np.array([np.linalg.norm(s['obs'][mask] - s['addobs'][mask]) / (np.linalg.norm(s['obs'][mask]) + 1e-9)
                  for s in prs])
    order = np.argsort(E); quarts = np.array_split(order, 4); qout = {}
    print("\n==== COMBO PREDICTION vs ADDITIVE BASELINES (drift-removed, intersection genes) ====")
    print(f"{'split':8}{'method':14}{'n':>4}{'disc_resid':>12}{'pear':>8}{'pear_noKO':>11}")
    for split in ('pairs', 'triples', 'all'):
        for m, lab in [('do', 'do-op'), ('add', 'model-add'), ('addobs', 'obs-add(null)')]:
            r = out[split][m]
            if r: print(f"{split:8}{lab:14}{r['n']:>4}{r['disc']:>12}{r['pear']:>8}{r['pear_noKO']:>11}")
        print()
    print("==== EPISTASIS QUARTILES (pairs, Q4 = most non-additive) — do-op vs additive ====")
    print(f"{'Q':4}{'n':>3}{'meanE':>7}{'do disc/pear':>16}{'add disc/pear':>16}{'obsadd disc/pear':>18}{'gap(do-add)pear':>16}")
    for qi, idx in enumerate(quarts, 1):
        rows = [prs[i] for i in idx]
        def qs(key):
            P = np.stack([r[key][mask] for r in rows]); O = np.stack([r['obs'][mask] for r in rows])
            Pd = P - P.mean(0); Od = O - O.mean(0)
            pe = float(np.mean([np.corrcoef(Pd[i], Od[i])[0, 1] for i in range(len(rows))
                                if Pd[i].std() > 1e-9 and Od[i].std() > 1e-9]))
            return disc(zrow(Pd), zrow(Od)), pe
        do_d, do_p = qs('do'); ad_d, ad_p = qs('add'); ob_d, ob_p = qs('addobs'); me = float(E[idx].mean())
        print(f"Q{qi:<3}{len(rows):>3}{me:>7.3f}{f'{do_d:.3f}/{do_p:.3f}':>16}{f'{ad_d:.3f}/{ad_p:.3f}':>16}"
              f"{f'{ob_d:.3f}/{ob_p:.3f}':>18}{do_p-ad_p:>16.3f}")
        qout[f'Q{qi}'] = dict(n=len(rows), meanE=round(me, 3), do=[round(do_d, 3), round(do_p, 3)],
                              add=[round(ad_d, 3), round(ad_p, 3)], addobs=[round(ob_d, 3), round(ob_p, 3)])
    out['epistasis_quartiles'] = qout
    od = f"results/combo_eval/{a.tag}"; os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/summary.json", 'w'), indent=2)
    print(f"\nwrote {od}/summary.json", flush=True)


if __name__ == '__main__':
    main()
