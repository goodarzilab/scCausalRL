"""Figure: the A_gap ↔ held-out disc_resid CONFLICT across decoder architectures, with bootstrap CIs.
Reloads each finished v2b model, recomputes per-test-TF held-out disc_resid (drift-removed cosine
discrimination) and per-test-TF A_gap (1−cos(Δ, Δ|A=0)), bootstraps over the 49 test TFs for error
bars, and plots A_gap (y) vs test disc (x). Reference lines: chance / ESM-NN baseline / data ceiling.

Run: python evaluation/make_tradeoff_figure.py --device cpu
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung

VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = os.path.join(VDIR, "joung_singles_proc_v2b.h5ad"); EFF = os.path.join(VDIR, "effective_tf_genes.txt")
# dataviz palette (categorical by decoder family) + status colors for reference lines
FAM = {'learned':'#2a78d6', 'masked':'#1baf7a', 'frozen':'#eda100', 'localized':'#4a3aa7', 'loc+masked':'#c0392b'}
GOOD, CRIT, MUTE = '#0ca30c', '#d03b3b', '#898781'
ARMS = [  # (tag, family, label, is_best)
 ('v2b_sparse_con',                    'learned',   'learned',        False),
 ('v2b_masked_go_l3',                  'masked',    'masked (BEST)',  True),
 ('v2b_masked_go',                     'masked',    'masked',         False),
 ('v2b_masked_tgt_delta_residual',     'masked',    'masked+resid',   False),
 ('v2b_masked_tgt_delta_plus_cellrecon','masked',   'masked+recon',   False),
 ('v2b_masked_tgt_delta_plus_mmd',     'masked',    'masked+mmd',     False),
 ('v2b_masked_tgt_statematch_only',    'masked',    'masked+state',   False),
 ('v2b_frozenRES64_s26',               'frozen',    'frozen-64',      False),
 ('v2b_frozenRES128_s26',              'frozen',    'frozen-128',     False),
 ('v2b_frozenRES256_s26',              'frozen',    'frozen-256',     False),
 ('v2b_sw_frozenRES512_pert30',        'frozen',    'frozen-446',     False),
 ('v2b_J_hardself_k4',                 'localized', 'loc k4',         False),
 ('v2b_J_hardself_k8',                 'localized', 'loc k8',         False),
 ('v2b_sw_J_hardself_L128_k16_pert30', 'localized', 'loc L128/k16',   False),
 ('v2b_sw_J_hardself_L256_k16_pert30', 'localized', 'loc L256/k16',   False),
 ('v2b_J_masked_hardself_k4',          'loc+masked','loc+masked k4',  False),
 ('v2b_J_masked_hardself_k8',          'loc+masked','loc+masked k8',  False),
 ('v2b_J_masked_hardself_k16',         'loc+masked','loc+masked k16', False),
]
CHANCE = 0.500


def zrow(M):
    M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def per_tf(model, gene_names, de, drift, test, x_ctrl, umode, beta):
    # do-op matches evaluation/agap_probe.py exactly (intervene_shift_mutilate + per-condition z_prev)
    gidx = {g: i for i, g in enumerate(gene_names)}
    _, _, eps = model.encoder(x_ctrl)
    def dooop(disabled):
        model.dag.disabled = disabled
        A = model.dag.sample_A(1).squeeze(0)
        zprev = model.dag.scm_forward(eps, A)
        out = []
        for g in test:
            zpost, _ = model.intervene_shift_mutilate(eps, [gidx[g]], A, zprev, u_mode=umode, shift_beta=beta)
            out.append((model.decoder(zpost) - model.decoder(zprev)).mean(0).detach().cpu().numpy())
        model.dag.disabled = False
        return np.stack(out)
    with torch.no_grad():
        predA = dooop(False)                                             # one A sample (for disc, train/test-eval style)
    obs = np.stack([np.asarray(de[g], np.float32) for g in test])
    # per-TF disc_resid (drift-removed): fraction of OTHER test TFs beaten by cosine
    C = zrow(predA - drift) @ zrow(obs - drift).T; dg = np.diag(C)[:, None]; T = len(test)
    disc_tf = (dg > C).sum(1) / (T - 1)                                   # [T]
    # per-TF A_gap: sample A PER TF (matches agap_probe → averages gate noise), 1 − mean-centered cos
    def c1(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    agap_tf = np.zeros(T)
    with torch.no_grad():
        for i, g in enumerate(test):
            model.dag.disabled = False; A = model.dag.sample_A(1).squeeze(0)
            zp = model.dag.scm_forward(eps, A)
            zpost, _ = model.intervene_shift_mutilate(eps, [gidx[g]], A, zp, u_mode=umode, shift_beta=beta)
            dA = (model.decoder(zpost) - model.decoder(zp)).mean(0).cpu().numpy()
            model.dag.disabled = True; A0 = model.dag.sample_A(1).squeeze(0)
            zp0 = model.dag.scm_forward(eps, A0)
            zpost0, _ = model.intervene_shift_mutilate(eps, [gidx[g]], A0, zp0, u_mode=umode, shift_beta=beta)
            d0 = (model.decoder(zpost0) - model.decoder(zp0)).mean(0).cpu().numpy()
            model.dag.disabled = False
            agap_tf[i] = 1.0 - c1(dA, d0)
    return disc_tf.astype(np.float64), agap_tf.astype(np.float64)


def msem(v):
    """mean and ±1 SEM across the held-out TFs (spread over the ~49 test TFs, not a bootstrap CI)."""
    v = np.asarray(v, np.float64); T = len(v)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(T))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu'); a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)
    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
    de = dd['de_single']; gn0 = list(map(str, dd['gene_names']))
    test0 = [g for g in dd['pert_split']['test'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in dd['pert_split']['train'] if g in de]).mean(0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(400, len(ctrl)), replace=False)]], device=dev)

    rows = []
    for tag, fam, lab, best in ARMS:
        ck = None
        for e in (60, 100):
            p = f"results/grid/{tag}/checkpoints/phase1_epoch{e}.pt"
            if os.path.exists(p): ck = p; break
        if ck is None: print(f"skip {tag} (no ckpt)", flush=True); continue
        model, gnm, ckd = load_shift_model(ck, dev)
        _p1 = (ckd.get('config', {}) or {}).get('phase1', {}) or {}
        umode = _p1.get('pert_mutilate_u_mode', 'shift'); beta = float(_p1.get('pert_mutilate_beta', 1.0))
        test = [g for g in test0 if g in gnm]
        dt, at = per_tf(model, gnm, de, drift, test, xc, umode, beta)
        dm, dse = msem(dt); am, ase = msem(at)
        rows.append(dict(tag=tag, fam=fam, lab=lab, best=best, disc=dm, dse=dse, ag=am, ase=ase))
        print(f"{lab:16s} disc={dm:.3f}±{dse:.3f}  A_gap={am:.3f}±{ase:.3f}", flush=True)

    # ---- plot: A_gap (y) vs held-out disc_resid (x), the conflict ----
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    xs = np.array([r['disc'] for r in rows]); ys = np.array([r['ag'] for r in rows])
    # anti-diagonal trend (drawn first, behind points)
    if len(xs) > 2:
        b, m = np.polynomial.polynomial.polyfit(xs, ys, 1)
        xr = np.linspace(xs.min() - 0.005, xs.max() + 0.005, 50)
        ax.plot(xr, b + m*xr, ls=':', color=MUTE, lw=1.6, zorder=1)
        ax.text(xs.max(), b + m*xs.max(), '  conflict: A used ↔ prediction', color=MUTE,
                fontsize=9, style='italic', ha='left', va='center')
    # subtle chance reference on the discrimination axis
    ax.axvline(CHANCE, ls='--', lw=1.1, color=MUTE, alpha=0.6, zorder=0)
    ax.text(CHANCE, ax.get_ylim()[1], ' chance\n 0.50', color=MUTE, va='top', ha='left', fontsize=8)
    for r in rows:
        c = FAM[r['fam']]; best = r['best']
        ax.errorbar(r['disc'], r['ag'], xerr=r['dse'], yerr=r['ase'],
                    fmt='*' if best else 'o', ms=22 if best else 9,
                    mfc=c, mec='k' if best else 'white', mew=1.8 if best else 0.8,
                    ecolor=c, elinewidth=1.2, capsize=2.5, alpha=0.95, zorder=5 if best else 3)
        ax.annotate(r['lab'], (r['disc'], r['ag']), textcoords='offset points',
                    xytext=(8, 5 if not best else -16), fontsize=8, color='#0b0b0b',
                    fontweight='bold' if best else 'normal',
                    ha='left' if not best else 'center')
    ax.set_xlabel('held-out disc_resid  (drift-removed cosine discrimination →)', fontsize=11)
    ax.set_ylabel('A_gap  (↑ latent DAG A is load-bearing)', fontsize=11)
    ax.set_title('The A-load-bearing ↔ generalization conflict  (PRIOR-OFF)\n(v2b, error bars = ±1 SEM across the 49 held-out TFs)',
                 fontsize=12)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0], marker='o', color='w', mfc=c, ms=9, label=f) for f,c in FAM.items()],
              title='decoder', loc='upper right', fontsize=9, framealpha=0.9)
    ax.margins(0.10); ax.grid(True, alpha=0.15); fig.tight_layout()
    out = 'results/figures/tradeoff_agap_vs_disc.png'; os.makedirs('results/figures', exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight'); print(f"\nwrote {out}", flush=True)


if __name__ == '__main__':
    main()
