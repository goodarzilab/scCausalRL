"""Panel-A retrieval baselines on the EXACT held-out single-TF set of jb_ft_replay_co8.

Reuses the canonical drift-removed metric from jb_eval_consolidated (`metrics`) and the SAME data prep
(load_joung_briggs with the checkpoint's split_seed / single_pert_train_frac / briggs_clean), so the WM+head
and WM-nohead numbers reproduce consolidated_jb_ft_replay_co8.json EXACTLY (joung held-out 0.831/0.481,
briggs 0.710/0.201) — i.e. every method lands on ONE identical held-out set with ONE identical metric.

Adds two model-free retrieval copies (logic reused from evaluation/singles_generalization_probe.py and
evaluation/eval_frontier.py::esm_knn_singles), library = Joung TRAIN singles, scored with the SAME metric():
  esm_knn   : copy the observed Δ of the single nearest TRAIN TF in the ShiftEncoder's own ESM input
              space (model.pert_label; falls back to data/joung_esm2_emb.npz if pert_label is empty).
  coexp_knn : copy the mean observed Δ of the k nearest TRAIN TFs by control co-expression (measured in the
              query group's OWN control context — Joung ctrl for Joung held-out, Briggs ctrl for Briggs OOD).
Drift-removed throughout (metric subtracts the group drift from BOTH prediction and observed); the query TF's
own OE gene is excluded from allDEG/pearson exactly as jb_eval_consolidated does (`ko`).
Emits results/briggs/panelA_baselines_co8.json.
"""
import os, sys, argparse, json
import numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.jb_eval_consolidated import metrics                      # canonical drift-removed disc/allDEG
from data.load_joung_briggs import load_joung_briggs
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
_ESM = os.path.join(os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')


def _unit(E):
    return E / (np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9)


def disc_items(P, O, drift, mask):
    """Per-item discrimination score — the exact per-i quantity metrics() averages into `disc`.
    disc_i = fraction of other items whose observed residual the item's OWN prediction beats
    (diag corr > off-diag corr). Full masked profile, KO/OE gene INCLUDED (matches metrics disc)."""
    base = np.where(mask)[0]
    Pd, Od = P - drift, O - drift

    def z(a):
        s = a.std(1, keepdims=True)
        return (a - a.mean(1, keepdims=True)) / np.where(s > 1e-9, s, 1)

    Pz, Oz = z(Pd[:, base]), z(Od[:, base])
    Cm = Pz @ Oz.T / Pz.shape[1]
    dg = np.diag(Cm)
    return [float((dg[i] > np.delete(Cm[i], i)).mean()) for i in range(len(P))] if len(P) > 1 else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--device', default='cpu')
    ap.add_argument('--tag', default='jb_ft_replay_co8'); ap.add_argument('--n_ctrl', type=int, default=4000)
    ap.add_argument('--thr', type=float, default=0.1); ap.add_argument('--knn_k', type=int, default=10)
    ap.add_argument('--n_coexp_ctrl', type=int, default=5000)
    ap.add_argument('--heldout_tfs', required=True, help='same fixed held-out singles file used by the consolidated eval')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)
    model, gn, ck = load_shift_model(a.checkpoint, dev)
    cfg = ck.get('config', {}) or {}; p1 = cfg.get('phase1', {}) or {}; dcfg = cfg.get('data', {})
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); gidx = {g: i for i, g in enumerate(gn)}; G = len(gn)
    has_head = getattr(model, 'interaction_U', None) is not None
    seed = int(dcfg.get('split_seed', 1))

    # --- reproduce the EXACT training data prep from the checkpoint (same as jb_eval_consolidated) ---
    mcc = int(dcfg.get('min_combo_cells') or 5); kmax = int(dcfg.get('combo_kmax') or 8)
    briggs_clean = bool(dcfg.get('briggs_clean', False)); holdout_ak = bool(dcfg.get('holdout_answerkey', True))
    jb = load_joung_briggs(joung_proc_path=f"{V}/joung_singles_proc_v2b.h5ad",
                           joung_combo_proc_path=f"{V}/joung_combo_proc_v2b.h5ad",
                           min_cells_per_pert=int(dcfg.get('min_cells_per_pert') or 5),
                           min_combo_cells=mcc, combo_kmax=kmax, split_seed=seed,
                           train_frac=float(dcfg.get('single_pert_train_frac', 1.0)),
                           val_frac=float(dcfg.get('single_pert_val_frac', 0.0)),
                           effective_tfs=f"{V}/effective_tf_genes.txt",
                           holdout_answerkey=holdout_ak, briggs_clean=briggs_clean)
    ad = jb['adata']; obs = ad.obs; X = ad.X; X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    de_sJ, de_sB = jb['de_single'], jb['de_single_briggs']
    maskB = jb['modality_gene_mask'][1].astype(bool); maskJ = np.ones(G, bool)
    train_singles = set(jb['pert_split']['train'])
    _jtr = [g for g in de_sJ if g in train_singles] or list(de_sJ)
    driftJ = np.mean([de_sJ[g] for g in _jtr], 0).astype(np.float32)
    driftB = np.mean([de_sB[g] for g in de_sB], 0).astype(np.float32) if de_sB else np.zeros(G, np.float32)
    print(f"[data] train singles={len(_jtr)} | briggs singles={len(de_sB)} | genes={G} | has_head={has_head}", flush=True)

    # --- control eps per modality (0=Joung, 1=Briggs) — same as jb_eval_consolidated.ctrl_eps ---
    def ctrl_eps(mod):
        idx = np.where((obs['modality_id'].values == mod) & (obs['is_control'].values))[0]
        rng = np.random.default_rng(0); sel = idx[rng.choice(len(idx), min(a.n_ctrl, len(idx)), replace=False)]
        xc = torch.tensor(X[sel], device=dev); m = torch.full((len(sel),), mod, dtype=torch.long, device=dev)
        _, _, eps = model.encoder(xc, modality_id=m, batch_id=m)
        A = model.dag.sample_A(1).squeeze(0); zp = model.dag.scm_forward(eps, A)
        cr = model.decoder(zp).mean(0).cpu().numpy()
        return eps, A, cr
    epsJ, AJ, crJ = ctrl_eps(0); epsB, AB, crB = ctrl_eps(1)

    def dop(gi, head, eps, A, cr, mod):                                   # single-TF do-op Δ (k=1 -> no combo head)
        g = torch.tensor(gi, device=dev); mid = torch.full((len(gi),), int(mod), dtype=torch.long, device=dev)
        S = model.compute_shift(g, modality_id=mid).sum(0)
        if head and has_head:
            si = model.compute_interaction(g)
            if si is not None: S = S + si
        u = (BETA * S.abs()).clamp(max=1.0)
        zp, _, _ = model.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u)
        return model.decoder(zp).mean(0).cpu().numpy() - cr

    # --- retrieval library = Joung TRAIN singles (what the model saw); ESM + coexpr neighbor sources ---
    train_tfs = [g for g in de_sJ if g in train_singles and g in gidx]
    tr_arr = np.array(train_tfs); tr_idx = np.array([gidx[g] for g in train_tfs])
    DE_tr = np.stack([np.asarray(de_sJ[g], np.float32) for g in train_tfs])   # observed Joung Δ of library
    lab = model.pert_label.detach().cpu().numpy() if getattr(model, 'pert_label', None) is not None else np.zeros((G, 1))
    esm_src = 'model.pert_label'
    if np.abs(lab).sum() == 0:                                            # fall back to standalone ESM2 npz
        z = np.load(_ESM, allow_pickle=True); emb = {str(g): e for g, e in zip(z['genes'], z['emb'])}
        dim = len(next(iter(emb.values()))); lab = np.stack([emb.get(g, np.zeros(dim)) for g in gn]); esm_src = 'joung_esm2_emb.npz'
    Ltr = _unit(lab[tr_idx]); print(f"[esm] neighbor source = {esm_src}", flush=True)

    def coexp_ctrl(mod):                                                  # standardized control cell x gene matrix
        idx = np.where((obs['modality_id'].values == mod) & (obs['is_control'].values))[0]
        rng = np.random.default_rng(1); sel = idx[rng.choice(len(idx), min(a.n_coexp_ctrl, len(idx)), replace=False)]
        Z = X[sel].astype(np.float32); Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        return Z, Z.shape[0]
    ZcJ, nCJ = coexp_ctrl(0); ZcB, nCB = coexp_ctrl(1)

    def esm_copy(query_tfs, ex_self):                                     # nearest TRAIN TF in ESM space
        P = []
        for g in query_tfs:
            sims = Ltr @ _unit(lab[gidx[g]])
            if ex_self:                                                   # never copy the query itself
                sims = np.where(tr_arr == g, -np.inf, sims)
            P.append(DE_tr[int(np.argmax(sims))])
        return np.stack(P)

    def coexp_copy(query_tfs, Zc, nC):                                    # mean Δ of k nearest TRAIN TFs by coexpr
        P = []
        for g in query_tfs:
            sims = (Zc[:, tr_idx].T @ Zc[:, gidx[g]]) / nC
            nbrs = [j for j in np.argsort(-sims) if tr_arr[j] != g][:a.knn_k]
            P.append(DE_tr[nbrs].mean(0))
        return np.stack(P)

    # --- the two panel-A held-out groups ---
    raw = open(a.heldout_tfs).read()
    HO = {t.strip() for t in raw.replace(',', '\n').split() if t.strip()}
    joung_ho = [g for g in de_sJ if g in HO and g in gidx]
    briggs_all = [g for g in de_sB if g in gidx]
    groups = {
        'joung_singles_heldout': dict(tfs=joung_ho, de=de_sJ, drift=driftJ, mask=maskJ,
                                       eps=epsJ, A=AJ, cr=crJ, mod=0, Zc=ZcJ, nC=nCJ, ex_self=True),
        'briggs_singles':        dict(tfs=briggs_all, de=de_sB, drift=driftB, mask=maskB,
                                       eps=epsB, A=AB, cr=crB, mod=1, Zc=ZcB, nC=nCB, ex_self=True),
    }
    EXPECT = {'joung_singles_heldout': (0.831, 0.481), 'briggs_singles': (0.710, 0.201)}   # consolidated WM+head disc/allDEG
    RES = {'tag': a.tag, 'esm_source': esm_src, 'n_train_library': len(train_tfs),
           'note': 'drift-removed; WM+head/-nohead identical on singles (head is a no-op on k=1); '
                   'retrieval library = Joung TRAIN singles; metric = jb_eval_consolidated.metrics '
                   '(same drift/mask/thr/KO-exclusion as consolidated_jb_ft_replay_co8.json).',
           'groups': {}}
    for name, c in groups.items():
        tfs = c['tfs']
        O = np.stack([np.asarray(c['de'][g], np.float32) for g in tfs])
        KO = [np.array([gidx[g]], dtype=int) for g in tfs]
        WH = np.stack([dop([gidx[g]], True, c['eps'], c['A'], c['cr'], c['mod']) for g in tfs])
        WN = np.stack([dop([gidx[g]], False, c['eps'], c['A'], c['cr'], c['mod']) for g in tfs])
        PE = esm_copy(tfs, c['ex_self']); PC = coexp_copy(tfs, c['Zc'], c['nC'])
        preds = {'WM+head': WH, 'WM-nohead': WN, 'esm_knn': PE, 'coexp_knn': PC}
        mm = {m: metrics(P, O, c['drift'], c['mask'], a.thr, ko=KO) for m, P in preds.items()}
        # per-TF item vectors so the figure can bootstrap a 95% CI over the held-out TFs.
        # disc_items = the exact per-i quantity metrics() means into `disc`; allDEG_items come
        # straight from metrics() (per-TF drift-removed DEG Pearson, KO gene excluded, None=undef).
        out = {m: {'n': d['n'], 'disc': d['disc'], 'allDEG': d['allDEG'], 'pearson': d['pearson'],
                   '_items': {'disc': disc_items(preds[m], O, c['drift'], c['mask']),
                              'allDEG': d.get('allDEG_items', []),
                              'pearson': d.get('pearson_items', [])}}
               for m, d in mm.items()}
        out['n'] = len(tfs); out['tfs'] = list(tfs); RES['groups'][name] = out
        ed, ea = EXPECT[name]; wh = out['WM+head']
        print(f"\n[{name}] n={len(tfs)}", flush=True)
        for m in ('WM+head', 'WM-nohead', 'esm_knn', 'coexp_knn'):
            print(f"   {m:11}: disc {out[m]['disc']:+.3f} | allDEG {out[m]['allDEG']:+.3f}", flush=True)
        print(f"   >> VERIFY WM+head vs consolidated: disc {wh['disc']:.3f} (exp {ed}) | "
              f"allDEG {wh['allDEG']:.3f} (exp {ea}) | "
              f"match={'OK' if abs(wh['disc']-ed)<0.01 and abs(wh['allDEG']-ea)<0.01 else 'MISMATCH'}", flush=True)
    if a.out:
        json.dump(RES, open(a.out, 'w'), indent=1); print(f"\nsaved -> {a.out}", flush=True)
    print("[panelA-baselines done]", flush=True)


if __name__ == '__main__':
    main()
