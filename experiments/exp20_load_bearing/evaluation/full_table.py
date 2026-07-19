"""Complete eval table across ALL completed prior-off v2b arms, loading the proc ONCE.

For each arm: TRAIN + HELD-OUT disc_resid (drift-removed cosine discrimination), pear_noKO
(off-target Pearson excluding the perturbed gene), A_gap (1 - mean-centered cos(Δ_A, Δ_{A=0}),
per-TF A sampling — matches agap_probe / the tradeoff figure), and |S| test/train ratio.
Reuses the EXACT do-op + metrics from train_vs_heldout.py so numbers are comparable.

Auto-discovers arms: results/grid/v2b_*/checkpoints/phase1_epoch{100,60}.pt, skipping
*_prioron / *_partial (archived) and JN* (multi-context, needs a different loader).

Run: python evaluation/full_table.py --device cpu
"""
import os, sys, glob, argparse, csv, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.heldout_zeroshot_eval import gene_space, _solve
from data.load_joung import load_joung

VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = os.path.join(VDIR, "joung_singles_proc_v2b.h5ad")
EFF = os.path.join(VDIR, "effective_tf_genes.txt")


def discover():
    arms = []
    for d in sorted(glob.glob("results/grid/v2b_*")):
        t = os.path.basename(d)
        if t.endswith("_prioron") or t.endswith("_partial") or t.startswith("v2b_JN"):
            continue
        ck = next((f"{d}/checkpoints/phase1_epoch{e}.pt" for e in (100, 60)
                   if os.path.exists(f"{d}/checkpoints/phase1_epoch{e}.pt")), None)
        if ck:
            arms.append((t, ck))
    return arms


def family(cfg, tag):
    m = cfg.get('model', {}); p1 = cfg.get('phase1', {})
    dec = m.get('decoder_type', 'linear')
    if p1.get('pert_hard_self_gate'):
        return 'loc+masked' if dec == 'masked_go' else 'localized'
    return {'masked_go': 'masked', 'frozen_go': 'frozen', 'linear': 'learned'}.get(dec, dec)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu')
    ap.add_argument('--n_ctrl', type=int, default=400); a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)

    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
    de = dd['de_single']
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc_np = Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]]
    train0 = [g for g in dd['pert_split']['train'] if g in de]
    test0 = [g for g in dd['pert_split']['test'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in train0]).mean(0)
    print(f"[data] {len(train0)} train / {len(test0)} test TFs | drift ||{np.linalg.norm(drift):.2f}||", flush=True)

    def _zr(M):
        M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

    def disc_pnk(pred, obs, genes, gidx):
        # OLD metric: drift-removed cosine discrimination (per-TF frac of others beaten).
        pr, ob = pred - drift, obs - drift
        C = _zr(pr) @ _zr(ob).T; dgn = np.diag(C)[:, None]; T = len(genes)
        disc_ps = (dgn > C).sum(1) / (T - 1) if T > 1 else np.full(T, np.nan)
        pnk_dr, pnk_raw, degauroc, effspear = [], [], [], []   # drift-removed pnk + NEW magnitude metrics
        for k, gg in enumerate(genes):
            m = np.ones(pr.shape[1], bool)
            if gg in gidx: m[gidx[gg]] = False                 # KO-excluded mask
            xd, yd = pr[k, m], ob[k, m]                         # drift-removed, KO-excluded
            pnk_dr.append(np.corrcoef(xd, yd)[0, 1] if xd.std() > 1e-9 and yd.std() > 1e-9 else np.nan)
            p, o = pred[k, m], obs[k, m]                        # RAW (drift-in), KO-excluded → your Pnk
            pnk_raw.append(np.corrcoef(p, o)[0, 1] if p.std() > 1e-9 and o.std() > 1e-9 else np.nan)
            deg = (np.abs(obs[k]) > 0.15).astype(int)            # DEG recovery on full gene vector
            degauroc.append(roc_auc_score(deg, np.abs(pred[k])) if 0 < deg.sum() < len(deg) else np.nan)
            effspear.append(spearmanr(np.abs(pred[k]), np.abs(obs[k])).correlation)
        pnk_dr, pnk_raw, degauroc, effspear = map(lambda a: np.array(a, float), (pnk_dr, pnk_raw, degauroc, effspear))
        nm = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else float('nan')
        sem = lambda a: float(np.nanstd(a, ddof=1) / np.sqrt(np.isfinite(a).sum())) if np.isfinite(a).sum() > 1 else 0.0
        return dict(disc=float(disc_ps.mean()), disc_sem=sem(disc_ps),
                    pnk_dr=nm(pnk_dr), pnk_raw=nm(pnk_raw), pnk_raw_sem=sem(pnk_raw),
                    deg_auroc=nm(degauroc), deg_auroc_sem=sem(degauroc), eff_spear=nm(effspear))

    def c1(u, v):
        u = u - u.mean(); v = v - v.mean()
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    rows = []
    for tag, ck in discover():
        try:
            model, gn, ckd = load_shift_model(ck, dev)
        except Exception as e:
            print(f"SKIP {tag}: load failed {e}", flush=True); continue
        cfg = ckd.get('config', {}) or {}; p1 = cfg.get('phase1', {}) or {}
        lp = p1.get('lambda_prior', '?'); fam = family(cfg, tag)
        d = model.latent_dim; BETA = float(p1.get('pert_mutilate_beta', 1.0))
        UMODE = p1.get('pert_mutilate_u_mode', 'shift')
        structural = getattr(model, 'pert_intervene_mode', 'legacy') == 'structural'
        gidx = {g: i for i, g in enumerate(gn)}
        train = [g for g in train0 if g in gidx]; test = [g for g in test0 if g in gidx]
        xc = torch.tensor(xc_np, dtype=torch.float32, device=dev)
        _, _, eps = model.encoder(xc)
        A = model.dag.sample_A(1).squeeze(0); z_prev = model.dag.scm_forward(eps, A)
        pert_dec = getattr(model, 'pert_decoder', None) or model.decoder
        ctrl_recon = pert_dec(z_prev).mean(0).cpu().numpy()

        def predict(genes):
            pred = np.zeros((len(genes), len(gn)), np.float32); sn = []
            for k, g in enumerate(genes):
                S = model.compute_shift(torch.tensor([gidx[g]], device=dev)).sum(0); sn.append(float(S.norm()))
                u = (BETA * S.abs()).clamp(max=1.0); u = model.refine_u(u, S)
                eps_t = eps + S.unsqueeze(0) if structural else model.mix_eps(
                    eps, z_prev + _solve(torch.eye(d, device=dev) - A.T, S.unsqueeze(0)), u.unsqueeze(0))
                z_post, _, _ = model.dag.scm_intervene_u(eps, A, eps_t, u)
                pred[k] = pert_dec(z_post).mean(0).cpu().numpy() - ctrl_recon
            return pred, np.array(sn)

        tr_obs = np.stack([np.asarray(de[g], np.float32) for g in train])
        te_obs = np.stack([np.asarray(de[g], np.float32) for g in test])
        tr_pred, tr_sn = predict(train); te_pred, te_sn = predict(test)
        TR = disc_pnk(tr_pred, tr_obs, train, gidx)
        TE = disc_pnk(te_pred, te_obs, test, gidx)

        # A_gap on held-out TFs (per-TF A sampling, intervene_shift_mutilate; matches agap_probe/figure)
        gaps = []
        for g in test:
            model.dag.disabled = False; Aa = model.dag.sample_A(1).squeeze(0)
            zp = model.dag.scm_forward(eps, Aa)
            zpa, _ = model.intervene_shift_mutilate(eps, [gidx[g]], Aa, zp, u_mode=UMODE, shift_beta=BETA)
            dA = (model.decoder(zpa) - model.decoder(zp)).mean(0).cpu().numpy()
            model.dag.disabled = True; A0 = model.dag.sample_A(1).squeeze(0)
            zp0 = model.dag.scm_forward(eps, A0)
            zp0i, _ = model.intervene_shift_mutilate(eps, [gidx[g]], A0, zp0, u_mode=UMODE, shift_beta=BETA)
            d0 = (model.decoder(zp0i) - model.decoder(zp0)).mean(0).cpu().numpy()
            model.dag.disabled = False
            gaps.append(1.0 - c1(dA, d0))
        agap = float(np.mean(gaps)); agap_sem = float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0

        row = dict(tag=tag, family=fam, dec=cfg.get('model', {}).get('decoder_type', 'linear'),
                   latent=d, prior=lp,
                   train_disc=TR['disc'], test_disc=TE['disc'], test_disc_sem=TE['disc_sem'],
                   disc_gap=TR['disc'] - TE['disc'],
                   test_pnk_driftrm=TE['pnk_dr'],                          # OLD drift-removed KO-excl pearson
                   test_Pnk=TE['pnk_raw'], test_Pnk_sem=TE['pnk_raw_sem'],  # NEW: raw KO-excl pearson (your Pnk)
                   test_deg_auroc=TE['deg_auroc'], test_deg_auroc_sem=TE['deg_auroc_sem'],
                   test_eff_spear=TE['eff_spear'],
                   train_Pnk=TR['pnk_raw'], train_deg_auroc=TR['deg_auroc'],
                   A_gap=agap, A_gap_sem=agap_sem,
                   Sratio=float(te_sn.mean() / (tr_sn.mean() + 1e-9)))
        rows.append(row)
        print(f"  {tag:32s} {fam:10s} L{d:<4d} disc={TE['disc']:.3f} Pnk={TE['pnk_raw']:.3f} "
              f"deg_auroc={TE['deg_auroc']:.3f} eff_spear={TE['eff_spear']:.3f} A_gap={agap:.3f}", flush=True)

    rows.sort(key=lambda r: -r['test_disc'])
    out = 'results/figures/full_eval_table.csv'; os.makedirs('results/figures', exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} arms)", flush=True)


if __name__ == '__main__':
    main()
