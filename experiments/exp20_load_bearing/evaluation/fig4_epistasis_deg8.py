"""Fig 4 (epistasis) recompute on the deg8 `co8` world-model family — the SAME WM the RL planner uses.

Replaces the old deg2 (v2b_mmd1_deg2 / _Jcombo_ho50) epistasis panels with the deg8 co8 models:
  combo-trained (held-out split)  = results/grid/jb_ft_replay_co8/checkpoints/ckpt.pt   (epoch60; the
        held-out-split sibling of the RL planner's jb_prod_wm; single_pert_train_frac 0.85, split_seed 1,
        combos held out via combo_test_keys). This is the checkpoint used for held-out PREDICTION (Fig 3).
  singles-only                    = results/grid/jb_singles_base_co8/checkpoints/ckpt.pt (epoch60; 0 combo
        train keys — its interaction head is UNtrained, so it is the "no combo training" control).

Do-operator + data prep are copied verbatim from evaluation/jb_eval_consolidated.py so predictions match the
held-out prediction eval exactly (per-modality shift scale, mutilated SCM solve, interaction head).

NET EPISTASIS (do-operator definition, matching _fig3_dump_points.py / briggs_epistasis_enrichment.py):
  For a combo S of size k, over the combo's OBSERVED drift-removed DE genes (|O - drift| > DEGT):
      eps_obs (S)  = (O   - Σ_g de_single[g])   + (k-1)·drift     # observed:  combo Δ − Σ observed single Δ
      eps_pred(S)  = (DM  - Σ_g do_op({g}))     + (k-1)·drift     # predicted: combo do-op − Σ single do-op
      net_o(S) = mean_{DE} eps_obs ;  net_m(S) = mean_{DE} eps_pred
  The +(k-1)·drift term drift-removes the additive baseline's k-fold OVER-counting of the shared TF-OE drift
  program (each single carries the drift; summing k of them counts it k times while the true combo expresses
  it ~once). Without it the raw combo−Σsingle epistasis is dominated by k (size-confounded) — see the memo
  and briggs_epistasis_enrichment.py. This term isolates TF-SPECIFIC interaction and generalises the k=2
  '+drift' correction of the old _fig3_dump_points.py to arbitrary k SYMMETRICALLY (same drift on both axes).
  We ALSO dump the raw (no-correction) net for transparency.

USABLE held-out combos = combo_test_keys whose EVERY member TF has an OBSERVED single-TF Δ in the same
context (de_single_briggs for Briggs, de_single for Joung) so an observed additive baseline exists.

Writes results/briggs/fig4_epistasis_deg8.json.  Run on SLURM cpu (see scripts/fig4_epi_deg8.sbatch).
"""
import os, sys, json, argparse
import numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung_briggs import load_joung_briggs
from scipy.stats import spearmanr, wilcoxon

V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
DEGT = 0.15
CKPTS = dict(combo='results/grid/jb_ft_replay_co8/checkpoints/ckpt.pt',
             singles='results/grid/jb_singles_base_co8/checkpoints/ckpt.pt')


def boot_ci_r(x, y, nboot=5000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); N = len(x)
    if N < 3: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); rs = []
    for _ in range(nboot):
        i = rng.integers(0, N, N); xi, yi = x[i], y[i]
        if xi.std() > 1e-9 and yi.std() > 1e-9:
            rs.append(np.corrcoef(xi, yi)[0, 1])
    if not rs: return (np.nan, np.nan)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def build_model_ops(ckpt_path, jb, n_ctrl, dev):
    """Load a checkpoint and return (predict_delta, ctx-by-modality, gidx, has_head). predict_delta(gi, head,
    mod) returns the model's predicted Δ (full-G) for gene-index list gi in modality `mod` (0=Joung,1=Briggs)."""
    model, gn, ck = load_shift_model(ckpt_path, dev)
    cfg = ck.get('config', {}) or {}; p1 = cfg.get('phase1', {}) or {}
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); gidx = {g: i for i, g in enumerate(gn)}
    has_head = getattr(model, 'interaction_U', None) is not None
    combo_dec = getattr(model, 'combo_decoder', None)
    ad = jb['adata']; obs = ad.obs; X = ad.X
    X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)

    def ctrl_ctx(mod):
        idx = np.where((obs['modality_id'].values == mod) & (obs['is_control'].values))[0]
        rng = np.random.default_rng(0); sel = idx[rng.choice(len(idx), min(n_ctrl, len(idx)), replace=False)]
        xc = torch.tensor(X[sel], device=dev); m = torch.full((len(sel),), mod, dtype=torch.long, device=dev)
        with torch.no_grad():
            _, _, eps = model.encoder(xc, modality_id=m, batch_id=m)
            A = model.dag.sample_A(1).squeeze(0); zp = model.dag.scm_forward(eps, A)
            cr = model.decoder(zp).mean(0).cpu().numpy()
            crc = combo_dec(zp).mean(0).cpu().numpy() if combo_dec is not None else cr
        return dict(eps=eps, A=A, cr=cr, crc=crc, mod=mod)

    ctx = {0: ctrl_ctx(0), 1: ctrl_ctx(1)}

    def predict_delta(gi, head, mod):
        c = ctx[mod]; g = torch.tensor(gi, device=dev)
        mid = torch.full((len(gi),), int(mod), dtype=torch.long, device=dev)
        S = model.compute_shift(g, modality_id=mid).sum(0)
        if head and has_head:
            si = model.compute_interaction(g)
            if si is not None: S = S + si
        u = (BETA * S.abs()).clamp(max=1.0)
        with torch.no_grad():
            zp, _, _ = model.dag.scm_intervene_u(c['eps'], c['A'], c['eps'] + S.unsqueeze(0), u)
            if combo_dec is not None and len(gi) >= 2:
                return combo_dec(zp).mean(0).cpu().numpy() - c['crc']
            return model.decoder(zp).mean(0).cpu().numpy() - c['cr']

    return predict_delta, gidx, has_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_ctrl', type=int, default=4000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='results/briggs/fig4_epistasis_deg8.json')
    a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)
    torch.manual_seed(0); np.random.seed(0)

    # ---- reproduce the co8 training data prep (both deg8 models share this exact config) ----
    _, _, ck = load_shift_model(CKPTS['combo'], 'cpu'); cfg = ck.get('config', {}) or {}
    d = cfg.get('data', {}) or {}; p1 = cfg.get('phase1', {}) or {}
    seed = int(d.get('split_seed', 1))
    test_keys = {tuple(sorted(map(str, k))) for k in (p1.get('combo_test_keys') or [])}
    jb = load_joung_briggs(joung_proc_path=f"{V}/joung_singles_proc_v2b.h5ad",
                           joung_combo_proc_path=f"{V}/joung_combo_proc_v2b.h5ad",
                           min_cells_per_pert=int(d.get('min_cells_per_pert') or 5),
                           min_combo_cells=int(d.get('min_combo_cells') or 5),
                           combo_kmax=int(d.get('combo_kmax') or 8), split_seed=seed,
                           train_frac=float(d.get('single_pert_train_frac', 1.0)),
                           val_frac=float(d.get('single_pert_val_frac', 0.0)),
                           effective_tfs=f"{V}/effective_tf_genes.txt",
                           holdout_answerkey=bool(d.get('holdout_answerkey', True)),
                           briggs_clean=bool(d.get('briggs_clean', False)))
    G = len(jb['gene_names'])
    de_sJ, de_sB = jb['de_single'], jb['de_single_briggs']
    de_combo = jb['adata'].uns['de_combo']; cdset = jb['combo_dataset']
    maskB = jb['modality_gene_mask'][1].astype(bool)
    train_singles = set(jb['pert_split']['train'])
    _jtr = [g for g in de_sJ if g in train_singles] or list(de_sJ)
    driftJ = np.mean([de_sJ[g] for g in _jtr], 0).astype(np.float32)
    driftB = (np.mean([de_sB[g] for g in de_sB], 0).astype(np.float32) if de_sB else np.zeros(G, np.float32))
    print(f"[fig4] G={G} | Joung singles={len(de_sJ)} | Briggs singles={len(de_sB)} | "
          f"test combos={len(test_keys)}", flush=True)

    # ---- build both models' do-op ----
    predict_combo, gidxC, headC = build_model_ops(CKPTS['combo'], jb, a.n_ctrl, dev)
    predict_sing, gidxS, headS = build_model_ops(CKPTS['singles'], jb, a.n_ctrl, dev)
    assert gidxC == gidxS, "gene axes differ between the two co8 models"
    gidx = gidxC
    print(f"[fig4] combo-model head={headC} | singles-model head={headS}", flush=True)

    # single-TF do-op caches (both models), for whichever members appear
    single_cache = {'combo': {}, 'singles': {}}

    def sdop(which, g, mod):
        c = single_cache[which]
        key = (g, mod)
        if key not in c:
            fn = predict_combo if which == 'combo' else predict_sing
            c[key] = fn([gidx[g]], True, mod)
        return c[key]

    def process(dataset):
        """Return per-combo epistasis records for held-out combos of `dataset` whose every member TF has an
        observed single-TF Δ in that context (so an observed additive baseline is defined)."""
        ctx_mod = 0 if dataset == 'joung' else 1
        de_s = de_sJ if dataset == 'joung' else de_sB
        drift = driftJ if dataset == 'joung' else driftB
        base = np.ones(G, bool) if dataset == 'joung' else maskB
        keys = sorted(k for k in de_combo if cdset.get(k) == dataset and len(k) >= 2 and k in test_keys
                      and all((g in gidx and g in de_s) for g in k))
        recs = []
        for k in keys:
            gi = [gidx[g] for g in k]
            kk = len(k)
            O = np.asarray(de_combo[k], np.float32)
            OAo = np.sum([np.asarray(de_s[g], np.float32) for g in k], 0)          # observed additive
            DMc = predict_combo(gi, True, ctx_mod)                                 # combo-model combo do-op
            MAc = np.sum([sdop('combo', g, ctx_mod) for g in k], 0)                # combo-model additive
            DMs = predict_sing(gi, True, ctx_mod)                                  # singles-model combo do-op
            MAs = np.sum([sdop('singles', g, ctx_mod) for g in k], 0)              # singles-model additive
            de = (np.abs(O - drift) > DEGT) & base
            if de.sum() < 3:
                continue
            corr = (kk - 1) * drift                                                # symmetric drift-removal
            def netv(P, PA, add_corr=True):
                e = (P - PA) + (corr if add_corr else 0.0)
                return float(e[de].mean())
            rec = dict(key='+'.join(k), tfs=list(k), k=kk, n_de=int(de.sum()),
                       net_o=netv(O, OAo), net_m_combo=netv(DMc, MAc), net_m_singles=netv(DMs, MAs),
                       net_o_raw=netv(O, OAo, False), net_m_combo_raw=netv(DMc, MAc, False),
                       # per-DE-gene do-op vs additive vs observed (for the panel-D example)
                       r_do=pear(DMc[de], O[de]), r_add=pear(MAc[de], O[de]))
            # stash per-gene arrays only for candidate panel-D combos (kept small: only if r_do decent)
            if rec['r_do'] is not np.nan and np.isfinite(rec['r_do']) and rec['r_do'] > 0.15 and kk <= 4:
                rec['_genes'] = dict(obs=(O - drift)[de].tolist(), do=(DMc - drift)[de].tolist(),
                                     add=(MAc - drift)[de].tolist())
            recs.append(rec)
        return recs

    out = dict(checkpoints=CKPTS, DEGT=DEGT, n_ctrl=int(a.n_ctrl), split_seed=seed,
               head=dict(combo=headC, singles=headS), datasets={})
    for dataset in ('briggs', 'joung'):
        recs = process(dataset)
        no = np.array([r['net_o'] for r in recs]); mc = np.array([r['net_m_combo'] for r in recs])
        ms = np.array([r['net_m_singles'] for r in recs])
        no_raw = np.array([r['net_o_raw'] for r in recs]); mc_raw = np.array([r['net_m_combo_raw'] for r in recs])
        r_combo = pear(no, mc); r_sing = pear(no, ms); r_raw = pear(no_raw, mc_raw)
        lo, hi = boot_ci_r(no, mc) if len(recs) >= 3 else (np.nan, np.nan)
        rs = float(spearmanr(no, mc).correlation) if len(recs) >= 3 else np.nan
        # per-k concordance (panel-C candidate)
        perk = {}
        for lo_k, hi_k, nm in [(2, 3, 'k2-3'), (4, 5, 'k4-5'), (6, 8, 'k6-8')]:
            idx = [i for i, r in enumerate(recs) if lo_k <= r['k'] <= hi_k]
            if len(idx) >= 3:
                xo = no[idx]; ym = mc[idx]
                l2, h2 = boot_ci_r(xo, ym)
                perk[nm] = dict(r=pear(xo, ym), n=len(idx), r_lo=l2, r_hi=h2)
        ds = dict(n=len(recs), r_combo=r_combo, r_singles=r_sing, r_combo_spearman=rs,
                  r_combo_ci=[lo, hi], r_combo_raw=r_raw,
                  net_o=no.tolist(), net_m_combo=mc.tolist(), net_m_singles=ms.tolist(),
                  keys=[r['key'] for r in recs], ks=[r['k'] for r in recs],
                  n_de=[r['n_de'] for r in recs], per_k=perk, recs=recs)
        out['datasets'][dataset] = ds
        print(f"\n[{dataset}] usable held-out combos with observed singles: n={len(recs)}", flush=True)
        print(f"   r(net_o, net_m_combo)  = {r_combo:+.3f}   CI[{lo:+.3f},{hi:+.3f}]  spearman={rs:+.3f}", flush=True)
        print(f"   r(net_o, net_m_singles)= {r_sing:+.3f}   (singles-only control)", flush=True)
        print(f"   r RAW (no drift-remove)= {r_raw:+.3f}", flush=True)
        for nm, v in perk.items():
            print(f"   per-k {nm}: r={v['r']:+.3f} n={v['n']}", flush=True)

    # choose panel-A set: Briggs if it has enough usable combos, else Joung
    nB = out['datasets']['briggs']['n']; nJ = out['datasets']['joung']['n']
    out['panelA_set'] = 'briggs' if nB >= 8 else 'joung'
    print(f"\n[fig4] panel-A set = {out['panelA_set']}  (briggs n={nB}, joung n={nJ})", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'))
    print(f"[fig4] wrote {a.out}", flush=True)


if __name__ == '__main__':
    main()
