"""CONSOLIDATED single-cell WM eval for the Joung+Briggs joint models — ONE eval, consistent metrics.

Metrics = the canonical exp20 set (combo_eval_full / baseline_eval): DRIFT-REMOVED disc / pearson / allDEG.
  drift  = mean single-TF Δ of that context's dataset (the shared TF-OE direction; baseline_eval 'perturb_mean').
  resid  = Δ − drift ; scored so raw shared-drift correlation is removed and only TF-specific signal counts.
  pearson = corr(resid_pred, resid_obs) over observed genes ; allDEG = same on |resid_obs|>thr ; disc = retrieval.
Predictions, all do-op from the CORRECT control (Joung ctrl for Joung sets, Briggs pluripotent ctrl for Briggs):
  WM+head    = Σ compute_shift + Σ_{i<j} interaction, mutilated solve
  WM-nohead  = head off (SCM composes summed single shifts)
  obs-add    = Σ_g de_single[g] from the SAME dataset (fair OBSERVATIONAL additive — not cross-dataset)
  model-add  = Σ_g (single do-op Δ) from the same control (model's own additive; no co-composition)
Groups: {Joung,Briggs} × {singles, combos} × {train, held-out} (held-out combos split by dataset)."""
import os, sys, argparse, json
import numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung_briggs import load_joung_briggs, _is_answerkey
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"


def metrics(P, O, drift, mask, thr, ko=None):
    """P,O: [n,G] pred/obs Δ. Drift-removed disc/pearson/allDEG (means over n items). CANONICAL spec:
    disc uses the full masked profile (KO/OE gene INCLUDED); pearson + allDEG EXCLUDE each item's own
    perturbed (KO) gene(s) via `ko` (list of full-G gene-index arrays, one per item) — the OE gene is the
    largest, trivially-predicted mover and inflates singles scores if kept. thr on |resid_obs|."""
    if len(P) == 0: return {}
    base = np.where(mask)[0]
    Pd = P - drift; Od = O - drift                                        # full-G residual
    def z(a): s = a.std(1, keepdims=True); return (a - a.mean(1, keepdims=True)) / np.where(s > 1e-9, s, 1)
    Pb, Ob = Pd[:, base], Od[:, base]
    Pz, Oz = z(Pb), z(Ob); Cm = Pz @ Oz.T / Pz.shape[1]; dg = np.diag(Cm)
    disc = np.mean([(dg[i] > np.delete(Cm[i], i)).mean() for i in range(len(P))]) if len(P) > 1 else np.nan
    pear, adeg = [], []
    for i in range(len(P)):
        cols = base if ko is None else base[~np.isin(base, ko[i])]       # drop this item's KO/OE gene(s)
        pi, oi = Pd[i, cols], Od[i, cols]
        pear.append(np.corrcoef(pi, oi)[0, 1] if pi.std() > 1e-9 and oi.std() > 1e-9 else np.nan)
        d = np.abs(oi) > thr
        adeg.append(np.corrcoef(pi[d], oi[d])[0, 1] if d.sum() > 2 and pi[d].std() > 1e-9 else np.nan)
    def _cl(L): return [None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x) for x in L]
    return dict(n=len(P), disc=float(disc), pearson=float(np.nanmean(pear)), allDEG=float(np.nanmean(adeg)),
                pearson_items=_cl(pear), allDEG_items=_cl(adeg))   # per-perturbation values for error bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--device', default='cpu')
    ap.add_argument('--tag', default='model'); ap.add_argument('--n_ctrl', type=int, default=4000)  # canonical: >=4000
    ap.add_argument('--thr', type=float, default=0.1); ap.add_argument('--combo_kmax', type=int, default=8)
    ap.add_argument('--min_combo_cells', type=int, default=10); ap.add_argument('--out', default=None)
    ap.add_argument('--heldout_tfs', default=None, help='comma-sep TF list OR path to a file (one TF/line): '
                    'score these EXACT Joung singles as held-out (base+arms on one identical set), overriding the split')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, gn, ck = load_shift_model(a.checkpoint, dev)
    cfg = ck.get('config', {}) or {}; p1 = cfg.get('phase1', {}) or {}
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); gidx = {g: i for i, g in enumerate(gn)}; G = len(gn)
    has_head = getattr(model, 'interaction_U', None) is not None
    seed = int(cfg.get('data', {}).get('split_seed', 1))
    test_keys = {tuple(sorted(map(str, k))) for k in (p1.get('combo_test_keys') or [])}
    train_keys = {tuple(sorted(map(str, k))) for k in (p1.get('combo_train_keys') or [])}

    dcfg = cfg.get('data', {})
    # Reproduce the TRAINING data prep from the checkpoint so the singles pert_split, the exact combo sets,
    # and the Briggs modality (clean vs permissive) all match what this model was trained/split on. Without
    # this a clean-trained model (briggs_clean=True) is scored against the permissive Briggs set, the held-out
    # combos with <a.min_combo_cells cells are silently dropped from scoring, and the held-out-singles split
    # (protect=combo_tfs) drifts. CLI values are the fallback only.
    mcc = int(dcfg.get('min_combo_cells') or a.min_combo_cells)
    kmax = int(dcfg.get('combo_kmax') or a.combo_kmax)
    briggs_clean = bool(dcfg.get('briggs_clean', False))
    holdout_ak = bool(dcfg.get('holdout_answerkey', True))
    jb = load_joung_briggs(joung_proc_path=f"{V}/joung_singles_proc_v2b.h5ad",
                           joung_combo_proc_path=f"{V}/joung_combo_proc_v2b.h5ad",
                           min_cells_per_pert=int(dcfg.get('min_cells_per_pert') or 5),
                           min_combo_cells=mcc, combo_kmax=kmax, split_seed=seed,
                           train_frac=float(dcfg.get('single_pert_train_frac', 1.0)),
                           val_frac=float(dcfg.get('single_pert_val_frac', 0.0)),
                           effective_tfs=f"{V}/effective_tf_genes.txt",
                           holdout_answerkey=holdout_ak, briggs_clean=briggs_clean)
    print(f"[eval-data] reproduced from ckpt: briggs_clean={briggs_clean} min_combo_cells={mcc} "
          f"combo_kmax={kmax} holdout_answerkey={holdout_ak}", flush=True)
    ad = jb['adata']; obs = ad.obs; X = ad.X; X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    de_sJ, de_sB = jb['de_single'], jb['de_single_briggs']; de_combo = ad.uns['de_combo']; cdset = jb['combo_dataset']
    maskB = jb['modality_gene_mask'][1].astype(bool); maskJ = np.ones(G, bool)
    psplit = jb['pert_split']; train_singles = set(psplit['train'])
    # canonical: drift = mean over TRAIN singles only (leak-free; matches bigtable). Briggs singles are
    # eval-only (never trained as singles) so driftB stays mean over all Briggs singles.
    _jtr = [g for g in de_sJ if g in train_singles] or list(de_sJ)
    driftJ = np.mean([de_sJ[g] for g in _jtr], 0).astype(np.float32)
    driftB = np.mean([de_sB[g] for g in de_sB], 0).astype(np.float32) if de_sB else np.zeros(G, np.float32)

    combo_dec = getattr(model, 'combo_decoder', None)                    # secondary combo head (|S|>=2 rows)
    def ctrl_eps(mod):
        idx = np.where((obs['modality_id'].values == mod) & (obs['is_control'].values))[0]
        rng = np.random.default_rng(0); sel = idx[rng.choice(len(idx), min(a.n_ctrl, len(idx)), replace=False)]
        xc = torch.tensor(X[sel], device=dev); m = torch.full((len(sel),), mod, dtype=torch.long, device=dev)
        with torch.no_grad():
            _, _, eps = model.encoder(xc, modality_id=m, batch_id=m)
            A = model.dag.sample_A(1).squeeze(0); zp = model.dag.scm_forward(eps, A)
            cr = model.decoder(zp).mean(0).cpu().numpy()
            crc = combo_dec(zp).mean(0).cpu().numpy() if combo_dec is not None else cr   # combo-head ctrl recon
        return eps, A, cr, crc
    epsJ, AJ, crJ, crcJ = ctrl_eps(0); epsB, AB, crB, crcB = ctrl_eps(1)

    def dop(gi, head, eps, A, cr, crc, mod):
        # mod = context modality (0=Joung, 1=Briggs). MUST pass it to compute_shift so Exp1's per-modality
        # shift scale fires. Combo sets (|S|>=2) decode through the secondary combo_decoder when present
        # (both prediction and the ctrl baseline crc), so combo Δ is measured in the combo head's output.
        g = torch.tensor(gi, device=dev); mid = torch.full((len(gi),), int(mod), dtype=torch.long, device=dev)
        S = model.compute_shift(g, modality_id=mid).sum(0)
        if head and has_head:
            si = model.compute_interaction(g)
            if si is not None: S = S + si
        u = (BETA * S.abs()).clamp(max=1.0)
        with torch.no_grad():
            zp, _, _ = model.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u)
            if combo_dec is not None and len(gi) >= 2:
                return combo_dec(zp).mean(0).cpu().numpy() - crc
            return model.decoder(zp).mean(0).cpu().numpy() - cr

    def preds_for(keys, ctx):                                            # ctx: (eps,A,cr,crc,de_single,drift,mask,name)
        eps, A, cr, crc, de_s, drift, mask, name = ctx
        mod = 0 if name == 'joung' else 1
        O, WH, WN, OA, MA, KO = [], [], [], [], [], []
        for key in keys:
            gi = [gidx[g] for g in key if g in gidx]
            if len(gi) < 1: continue
            O.append(np.asarray(de_combo[key] if len(key) > 1 else de_s[key[0]], np.float32))
            WH.append(dop(gi, True, eps, A, cr, crc, mod)); WN.append(dop(gi, False, eps, A, cr, crc, mod))
            oa = np.zeros(G, np.float32)
            for g in key:
                if g in de_s: oa = oa + de_s[g]
            OA.append(oa)
            MA.append(sum(dop([gidx[g]], True, eps, A, cr, crc, mod) for g in key if g in gidx))
            KO.append(np.asarray(gi, dtype=int))                          # per-item KO/OE gene idxs (excluded from pear/allDEG)
        stk = lambda L: np.stack(L) if L else np.zeros((0, G), np.float32)
        return stk(O), stk(WH), stk(WN), stk(OA), stk(MA), drift, mask, KO

    ctxJ = (epsJ, AJ, crJ, crcJ, de_sJ, driftJ, maskJ, 'joung')
    ctxB = (epsB, AB, crB, crcB, de_sB, driftB, maskB, 'briggs')

    def emit(name, keys, ctx):
        O, WH, WN, OA, MA, drift, mask, KO = preds_for(keys, ctx)
        if len(O) == 0: print(f"  {name}: (none)", flush=True); return None
        r = {v: metrics(P, O, drift, mask, a.thr, ko=KO) for v, P in [('WM+head', WH), ('WM-nohead', WN), ('obs-add', OA), ('model-add', MA)]}
        print(f"  {name} (n={len(O)}):", flush=True)
        for v in ('WM+head', 'WM-nohead', 'obs-add', 'model-add'):
            print(f"     {v:11}: disc {r[v]['disc']:+.3f} | pearson {r[v]['pearson']:+.3f} | allDEG {r[v]['allDEG']:+.3f}", flush=True)
        return r

    RES = {'tag': a.tag, 'has_head': has_head, 'groups': {}}
    print(f"\n===== CONSOLIDATED EVAL: {a.tag} (drift-removed disc/pearson/allDEG) =====", flush=True)
    # SINGLES. --heldout_tfs pins the held-out set to a FIXED TF list (so base + every finetune arm are
    # scored on the identical singles despite their differing combo-TF-protect splits).
    if a.heldout_tfs:
        raw = open(a.heldout_tfs).read() if os.path.exists(a.heldout_tfs) else a.heldout_tfs
        HO = {t.strip() for t in raw.replace(',', '\n').split() if t.strip()}
        js_tr = [(g,) for g in de_sJ if g not in HO]; js_te = [(g,) for g in de_sJ if g in HO]
        print(f"[eval-data] FIXED held-out singles list: {len(js_te)}/{len(HO)} present in vocab", flush=True)
    else:
        js_tr = [(g,) for g in de_sJ if g in train_singles]; js_te = [(g,) for g in de_sJ if g not in train_singles]
    print("[Joung singles]", flush=True)
    RES['groups']['joung_singles_train'] = emit('  train', js_tr, ctxJ)
    if js_te: RES['groups']['joung_singles_heldout'] = emit('  held-out', js_te, ctxJ)
    print("[Briggs singles]", flush=True)
    RES['groups']['briggs_singles'] = emit('  all', [(g,) for g in de_sB], ctxB)
    # COMBOS split by dataset × train/held-out, by k
    for dsname, ctx in [('joung', ctxJ), ('briggs', ctxB)]:
        ck_keys = [k for k in de_combo if cdset.get(k) == dsname and len(k) >= 2]
        for split, S in [('held-out', test_keys), ('train', train_keys)]:
            keys = [k for k in ck_keys if k in S]
            print(f"[{dsname} combos — {split}]", flush=True)
            RES['groups'][f'{dsname}_combos_{split}'] = emit('  all-k', keys, ctx)
            for lo, hi, nm in [(2, 3, 'k2-3'), (4, 5, 'k4-5'), (6, 8, 'k6-8')]:
                kk = [k for k in keys if lo <= len(k) <= hi]
                if kk: RES['groups'][f'{dsname}_combos_{split}_{nm}'] = emit(f'  {nm}', kk, ctx)
    # ANSWER-KEY Briggs combos (P1/P2/Lange/... literature HSPC cocktails): held out of training ENTIRELY
    # via holdout_answerkey, so absent from combo_test_keys — score them as a dedicated held-out group (the
    # biologically important recovery target). de_combo retains them regardless.
    ak = [k for k in de_combo if cdset.get(k) == 'briggs' and len(k) >= 2 and _is_answerkey(k)]
    if ak:
        print("[Briggs combos — ANSWER-KEY (held out of training)]", flush=True)
        RES['groups']['briggs_combos_answerkey'] = emit('  all-k', ak, ctxB)
        for lo, hi, nm in [(2, 3, 'k2-3'), (4, 5, 'k4-5'), (6, 8, 'k6-8')]:
            kk = [k for k in ak if lo <= len(k) <= hi]
            if kk: RES['groups'][f'briggs_combos_answerkey_{nm}'] = emit(f'  {nm}', kk, ctxB)
    if a.out:
        json.dump(RES, open(a.out, 'w'), indent=1); print(f"\nsaved -> {a.out}", flush=True)
    print("\n[jb-consolidated-eval done]", flush=True)


if __name__ == '__main__':
    main()
