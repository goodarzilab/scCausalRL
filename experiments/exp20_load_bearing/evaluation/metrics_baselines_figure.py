"""Clean grouped BAR plot (±1 SEM over the 49 held-out TFs): the top model vs the baselines
(ESM-NN, perturb-mean/drift, control-coexpr correlation) on the same metric suite:
  disc_resid     — drift-REMOVED cosine discrimination (chance 0.5)  [primary]
  disc_raw_delta — SAME discrimination but on RAW Δ (no drift removal)
  Pnk            — raw KO-excluded Pearson (drift-in, self-gene dropped)
  eff_spear      — Spearman(|pred|,|obs|) effect-size ranking
(DEG-AUROC deliberately excluded — saturated.)

Run: python evaluation/metrics_baselines_figure.py [--device cpu]
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung

VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = os.path.join(VDIR, "joung_singles_proc_v2b.h5ad"); EFF = os.path.join(VDIR, "effective_tf_genes.txt")
ESM = os.path.join(os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')
TOP = 'results/grid/v2b_retrfix_refresh_k20/checkpoints/phase1_epoch60.pt'


def zrow(M):
    M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def disc_per(pred, obs):                                # per-TF fraction of other TFs beaten by cosine
    C = zrow(pred) @ zrow(obs).T; d = np.diag(C)[:, None]; T = len(pred)
    return (d > C).sum(1) / (T - 1)


def pnk_per(pred, obs, genes, gidx):
    out = []
    for k, g in enumerate(genes):
        m = np.ones(pred.shape[1], bool)
        if g in gidx: m[gidx[g]] = False
        x, y = pred[k, m], obs[k, m]
        out.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
    return np.array(out)


def eff_per(pred, obs):
    return np.array([spearmanr(np.abs(pred[k]), np.abs(obs[k])).correlation for k in range(len(pred))])


def deg_pear_per(pr, ob, genes, gidx, K=20):            # GEARS-style: Pearson on top-K observed DEGs
    out = []
    for k, g in enumerate(genes):
        keep = np.ones(pr.shape[1], bool)
        if g in gidx: keep[gidx[g]] = False             # exclude the perturbed gene (off-target only)
        idx = np.where(keep)[0]
        deg = idx[np.argsort(-np.abs(ob[k, idx]))[:K]]   # top-K movers by |observed drift-removed Δ|
        x, y = pr[k, deg], ob[k, deg]
        out.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu'); a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)
    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
    de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
    tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
    obs = np.stack([np.asarray(de[g], np.float32) for g in te])                 # raw observed Δ [T,G]
    DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr])                # raw train Δ [Ntr,G]

    # ---- top model do-op predictions ----
    model, gnm, ck = load_shift_model(TOP, dev); p1 = (ck.get('config', {}) or {}).get('phase1', {})
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); UMODE = p1.get('pert_mutilate_u_mode', 'shift')
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(400, len(ctrl)), replace=False)]], device=dev)
    _, _, eps = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0); z_prev = model.dag.scm_forward(eps, A)
    cr = model.decoder(z_prev).mean(0).cpu().numpy()
    pred_model = np.zeros_like(obs)
    for k, g in enumerate(te):
        zp, _ = model.intervene_shift_mutilate(eps, [gidx[g]], A, z_prev, u_mode=UMODE, shift_beta=BETA)
        pred_model[k] = model.decoder(zp).mean(0).cpu().numpy() - cr

    # ---- baselines ----
    z = np.load(ESM, allow_pickle=True); esm = {str(gg): e for gg, e in zip(z['genes'], z['emb'])}
    dim = len(next(iter(esm.values())))
    Etr = np.stack([esm.get(g, np.zeros(dim)) for g in tr]); Ete = np.stack([esm.get(g, np.zeros(dim)) for g in te])
    Etr = Etr / (np.linalg.norm(Etr, axis=1, keepdims=True) + 1e-9); Ete = Ete / (np.linalg.norm(Ete, axis=1, keepdims=True) + 1e-9)
    pred_esm = DEtr[(Ete @ Etr.T).argmax(1)]                                    # ESM nearest-train
    pred_mean = np.repeat(drift[None], len(te), 0)                              # perturb-mean (drift)
    # coexpr-NN (COPY): nearest train TF by control co-expression → COPY its real observed residual.
    # (Uses perturbation data; strong on magnitude metrics because it emits a real response shape.)
    Xz = (Xc[ctrl] - Xc[ctrl].mean(0)); Xz = Xz / (np.linalg.norm(Xz, axis=0, keepdims=True) + 1e-9)  # gene z over ctrl
    cc = Xz[:, [gidx[g] for g in tr + te]]; corr = cc.T @ cc; ntr = len(tr)
    pred_coexprNN = DEtr[corr[ntr:, :ntr].argmax(1)]
    # corr-BLIND: predict Δ_g ∝ control co-expression corr(gene_g, TF) — NO observed response copied
    # (perturbation-blind; the exp17 'correlation baseline' that scored ~negative on effect metrics).
    pred_corrblind = np.stack([Xz[:, gidx[g]] @ Xz for g in te]).astype(np.float32)  # [T,G] coexpr profile

    METHODS = [('top model', pred_model, '#1baf7a'), ('ESM-NN(copy)', pred_esm, '#2a78d6'),
               ('coexpr-NN(copy)', pred_coexprNN, '#eda100'),
               ('corr-blind', pred_corrblind, '#9b59b6'), ('perturb-mean', pred_mean, '#888')]
    METRICS = ['disc_resid', 'Pnk_resid', 'deg_pear']   # disc=retrieval; Pnk+DEG-Pearson=profile fit (drift-removed)

    def metrics(pred):
        pr, ob = pred - drift, obs - drift                          # drift-removed everything
        return {'disc_resid': disc_per(pr, ob), 'Pnk_resid': pnk_per(pr, ob, te, gidx),
                'deg_pear': deg_pear_per(pr, ob, te, gidx)}
    res = {name: metrics(pred) for name, pred, _ in METHODS}

    print(f"{'method':14}" + "".join(f"{m:>16}" for m in METRICS))
    for name, _, _ in METHODS:
        print(f"{name:14}" + "".join(f"{np.nanmean(res[name][m]):>10.3f}±{np.nanstd(res[name][m], ddof=1)/np.sqrt(np.isfinite(res[name][m]).sum()):<5.3f}" for m in METRICS), flush=True)

    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(METRICS)); w = 0.16
    for i, (name, _, c) in enumerate(METHODS):
        means = [np.nanmean(res[name][m]) for m in METRICS]
        sems = [np.nanstd(res[name][m], ddof=1) / np.sqrt(np.isfinite(res[name][m]).sum()) for m in METRICS]
        ax.bar(x + (i - 2) * w, means, w, yerr=sems, color=c, label=name, alpha=0.92,
               error_kw=dict(ecolor='#333', elinewidth=1.0, capsize=3))
    CHANCE = {0: 0.5, 1: 0.0, 2: 0.0}   # disc chance=0.5; Pnk & DEG-Pearson (correlations) chance=0
    for xi, ch in CHANCE.items():
        ax.plot([xi - 0.45, xi + 0.45], [ch, ch], ls='--', lw=1.3, color='#c0392b')
        ax.text(xi, ch + 0.01, f'chance={ch:g}', color='#c0392b', fontsize=8, va='bottom', ha='center')
    ax.axhline(0, color='#444', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(['disc_resid\n(retrieval, drift-removed)', 'Pnk_resid\n(whole-profile Pearson)', 'DEG-Pearson\n(top-20 DEGs, GEARS)'], fontsize=9)
    ax.set_ylabel('metric value  (±1 SEM over 49 held-out TFs)', fontsize=11)
    ax.set_title('Held-out singles: top model vs baselines, across metrics', fontsize=12)
    ax.legend(fontsize=8, ncol=5, loc='upper center'); ax.set_ylim(-0.45, 1.0); ax.grid(axis='y', alpha=0.15)
    fig.tight_layout(); out = 'results/figures/metrics_baselines.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
