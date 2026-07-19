"""Diagnose per-held-out-TF discrimination variance for the top v2b model.

Top model: results/grid/v2b_retrfix_refresh_k20/checkpoints/phase1_epoch60.pt
  (masked_go decoder + L0 target gate + contrastive + ESM retrieval, structural do-op).

For each of the 49 held-out (test) TFs:
  - per-TF disc_resid = fraction of OTHER test TFs beaten by drift-removed cosine to own observed residual
  - candidate explanatory features (ESM NN sim, retrieval entropy/max-wt, silence, dmag, resid-norm, ctrl-PCA NN sim)
Then: correlate features with disc, split HIGH/LOW thirds, cluster ESM, make figure.

Run: PYTHONPATH=. python evaluation/tf_variance_diagnosis.py --device cpu
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.heldout_zeroshot_eval import _solve
from data.load_joung import load_joung
from utils.pert_descriptors import _control_pca

VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = os.path.join(VDIR, "joung_singles_proc_v2b.h5ad")
EFF = os.path.join(VDIR, "effective_tf_genes.txt")
ESM = os.path.join(os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')
CKPT = "results/grid/v2b_retrfix_refresh_k20/checkpoints/phase1_epoch60.pt"


def zrow(M):
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu')
    ap.add_argument('--n_ctrl', type=int, default=400); a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)

    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
    de = dd['de_single']
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc_np = Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]]
    gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
    train0 = [g for g in dd['pert_split']['train'] if g in de]
    test0 = [g for g in dd['pert_split']['test'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in train0]).mean(0)

    # ---- model + do-op prediction (structural), reusing full_table.py machinery ----
    model, gn2, ckd = load_shift_model(CKPT, dev)
    assert gn2 == gn
    cfg = ckd.get('config', {}); p1 = cfg.get('phase1', {})
    BETA = float(p1.get('pert_mutilate_beta', 1.0))
    structural = getattr(model, 'pert_intervene_mode', 'legacy') == 'structural'
    d = model.latent_dim
    train = [g for g in train0 if g in gidx]; test = [g for g in test0 if g in gidx]
    print(f"[data] {len(train)} train / {len(test)} test TFs, structural={structural}", flush=True)

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

    te_obs = np.stack([np.asarray(de[g], np.float32) for g in test])
    te_pred, te_sn = predict(test)

    # per-TF disc_resid (drift-removed, fraction of other test TFs beaten)
    pr, ob = te_pred - drift, te_obs - drift
    C = zrow(pr) @ zrow(ob).T; dgn = np.diag(C)[:, None]; T = len(test)
    disc = ((dgn > C).sum(1) / (T - 1)).astype(float)     # [49]

    # ---- candidate features ----
    # ESM raw (1280-d)
    z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}
    dim = len(next(iter(esm.values())))
    def esmv(g):
        v = esm.get(g);
        return v if (v is not None and np.linalg.norm(v) > 0) else None
    Etr_raw = np.stack([esm.get(g, np.zeros(dim)) for g in train])
    Ete_raw = np.stack([esm.get(g, np.zeros(dim)) for g in test])
    Etr_n = Etr_raw / (np.linalg.norm(Etr_raw, axis=1, keepdims=True) + 1e-9)
    Ete_n = Ete_raw / (np.linalg.norm(Ete_raw, axis=1, keepdims=True) + 1e-9)
    Sim = Ete_n @ Etr_n.T                                  # [49, N_tr] cosine
    esm_nn = Sim.max(1)                                    # nearest train sim
    esm_top5 = np.sort(Sim, 1)[:, -5:].mean(1)             # mean top-5
    esm_orphan = np.array([esmv(g) is None for g in test]) # no ESM at all

    # retrieval attention (uses model's own projected keys, self-masked, top-k)
    lab_te = model.pert_label[torch.tensor([gidx[g] for g in test], device=dev)]
    q = model.retr_proj(lab_te); K = model.retr_proj(model.retr_key)
    scores = (q @ K.t()) / (model.retr_proj.out_features ** 0.5 * model.retr_tau)
    gi_te = torch.tensor([gidx[g] for g in test], device=dev)
    selfmask = (gi_te[:, None] == model.retr_gene[None, :])
    scores = scores.masked_fill(selfmask, float('-inf'))
    if model.retr_k < scores.shape[1]:
        kth = scores.topk(model.retr_k, dim=1).values[:, -1:]
        scores = scores.masked_fill(scores < kth, float('-inf'))
    attn = torch.softmax(scores, dim=1).cpu().numpy()      # [49, N_tr]
    retr_max = attn.max(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -(attn * np.where(attn > 0, np.log(attn), 0.0)).sum(1)
    retr_ent = ent                                         # entropy (nats); lower=peaked
    retr_norm = float(model.retr_alpha) * np.linalg.norm(attn @ model.retr_val.cpu().numpy(), axis=1)

    # silence in control (TF's own gene mean control expr < global gene median)
    tei = [gidx[g] for g in test]
    ctrl_expr = Xc[ctrl][:, tei].mean(0)
    med = np.median(Xc[ctrl].mean(0))
    silent = ctrl_expr < med

    # magnitude of observed delta + residual norm
    dmag = np.linalg.norm(te_obs, axis=1)
    resid_norm = np.linalg.norm(te_obs - drift, axis=1)
    pred_sn = te_sn                                        # predicted shift norm |S|
    # derived: how drift-dominated is each observed Δ? (cos to mean-train drift) + residual fraction
    dn = drift / (np.linalg.norm(drift) + 1e-9)
    drift_cos = (te_obs @ dn) / (dmag + 1e-9)              # cos(obs Δ, drift): 1 => pure drift
    resid_frac = resid_norm / (dmag + 1e-9)                # fraction of Δ orthogonal-ish to drift

    # control-PCA nearest-train sim
    Dpca = _control_pca(Xc, ctrl, 50, 2000, 0)
    Ptr = Dpca[[gidx[g] for g in train]]; Pte = Dpca[tei]
    Ptr = Ptr / (np.linalg.norm(Ptr, axis=1, keepdims=True) + 1e-9)
    Pte = Pte / (np.linalg.norm(Pte, axis=1, keepdims=True) + 1e-9)
    pca_nn = (Pte @ Ptr.T).max(1)

    feats = {
        'esm_nn_sim': esm_nn, 'esm_top5_sim': esm_top5,
        'retr_max_wt': retr_max, 'retr_entropy': retr_ent, 'retr_shift_norm': retr_norm,
        'ctrl_expr': ctrl_expr, 'silent(bool)': silent.astype(float),
        'obs_delta_norm': dmag, 'resid_norm': resid_norm, 'pred_shift_norm': pred_sn,
        'ctrlPCA_nn_sim': pca_nn, 'drift_cos': drift_cos, 'resid_frac': resid_frac,
    }

    # ---- correlations ----
    print("\n=== Per-TF disc_resid stats ===", flush=True)
    print(f"  mean={disc.mean():.3f}  std={disc.std():.3f}  min={disc.min():.3f}  "
          f"max={disc.max():.3f}  (chance=0.50)", flush=True)
    print(f"\n=== Feature vs per-TF disc correlations (n={len(test)}) ===", flush=True)
    print(f"  {'feature':<20}{'pearson_r':>11}{'p':>9}{'spearman_r':>12}{'p':>9}", flush=True)
    corr_rows = []
    for nm, v in feats.items():
        v = np.asarray(v, float)
        pr_r, pr_p = pearsonr(v, disc); sp_r, sp_p = spearmanr(v, disc)
        corr_rows.append((nm, pr_r, pr_p, sp_r, sp_p))
        print(f"  {nm:<20}{pr_r:>11.3f}{pr_p:>9.3f}{sp_r:>12.3f}{sp_p:>9.3f}", flush=True)

    # ---- HIGH vs LOW thirds ----
    order = np.argsort(disc); n3 = len(test) // 3
    low_i = order[:n3]; high_i = order[-n3:]
    print(f"\n=== HIGH vs LOW thirds (n={n3} each) ===", flush=True)
    hi_tf = [test[i] for i in high_i]; lo_tf = [test[i] for i in low_i]
    print(f"  HIGH disc TFs ({disc[high_i].min():.2f}-{disc[high_i].max():.2f}): {hi_tf}", flush=True)
    print(f"  LOW  disc TFs ({disc[low_i].min():.2f}-{disc[low_i].max():.2f}): {lo_tf}", flush=True)
    print(f"\n  {'feature':<20}{'HIGH mean':>11}{'LOW mean':>11}{'delta':>9}", flush=True)
    for nm, v in feats.items():
        v = np.asarray(v, float)
        print(f"  {nm:<20}{v[high_i].mean():>11.3f}{v[low_i].mean():>11.3f}{v[high_i].mean()-v[low_i].mean():>+9.3f}", flush=True)
    print(f"\n  silent fraction: HIGH={silent[high_i].mean():.2f}  LOW={silent[low_i].mean():.2f}", flush=True)
    print(f"  ESM-orphan count: HIGH={esm_orphan[high_i].sum()}  LOW={esm_orphan[low_i].sum()}  (total orphans={esm_orphan.sum()})", flush=True)

    # ---- ESM 2D PCA embedding of test TFs ----
    Ecen = Ete_n - Ete_n.mean(0)
    U, Sv, Vt = np.linalg.svd(Ecen, full_matrices=False)
    esm2d = U[:, :2] * Sv[:2]

    # ---- figure ----
    make_figure(test, disc, esm_nn, esm2d, silent, feats, order, n3, corr_rows)

    # save raw table
    import csv
    os.makedirs('results/figures', exist_ok=True)
    out = 'results/figures/tf_variance_perTF.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['TF', 'disc'] + list(feats.keys()) + ['esm_orphan'])
        for i, g in enumerate(test):
            w.writerow([g, f'{disc[i]:.4f}'] + [f'{np.asarray(v,float)[i]:.4f}' for v in feats.values()] + [int(esm_orphan[i])])
    print(f"\nwrote {out}", flush=True)


def make_figure(test, disc, esm_nn, esm2d, silent, feats, order, n3, corr_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    INK = '#1b1b1f'; MUTE = '#8a8a94'; GRID = '#e7e7ec'
    LOW_C = '#4269d0'; MID_C = '#c9cdd6'; HIGH_C = '#e6572e'
    CMAP = plt.cm.get_cmap('RdYlBu_r')

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTE, 'axes.linewidth': 0.8,
                         'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
                         'axes.labelcolor': INK, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                  height_ratios=[1.0, 1.0])

    # (a) sorted per-TF disc bar
    axa = fig.add_subplot(gs[0, :2])
    o = order
    cols = [CMAP((disc[i] - disc.min()) / (disc.max() - disc.min() + 1e-9)) for i in o]
    axa.bar(range(len(test)), disc[o], color=cols, edgecolor='none', width=0.9)
    axa.axhline(0.5, color=MUTE, lw=1, ls='--')
    axa.text(0.5, 0.505, 'chance 0.50', color=MUTE, fontsize=8, va='bottom')
    axa.axhline(0.699, color='#2e8b57', lw=1, ls=':')
    axa.text(0.5, 0.705, 'data ceiling ~0.70', color='#2e8b57', fontsize=8, va='bottom')
    axa.axhline(disc.mean(), color=INK, lw=1, ls='-')
    axa.text(len(test)-0.5, disc.mean()+0.005, f'mean {disc.mean():.2f}', color=INK, fontsize=8, ha='right', va='bottom')
    axa.set_xticks(range(len(test)))
    axa.set_xticklabels([test[i] for i in o], rotation=90, fontsize=5.5)
    axa.set_ylabel('per-TF disc_resid')
    axa.set_title('(a) Held-out TF discrimination, sorted', loc='left', fontweight='bold', fontsize=11)
    axa.set_ylim(0, 1.0)
    for s in ('top', 'right'): axa.spines[s].set_visible(False)

    # (b) disc vs ESM nearest-train sim
    axb = fig.add_subplot(gs[0, 2])
    sc = axb.scatter(esm_nn, disc, c=disc, cmap=CMAP, s=40, edgecolor='white', linewidth=0.5)
    # fit line
    m, b = np.polyfit(esm_nn, disc, 1)
    xs = np.linspace(esm_nn.min(), esm_nn.max(), 50)
    axb.plot(xs, m*xs+b, color=INK, lw=1.2, ls='--')
    r = np.corrcoef(esm_nn, disc)[0, 1]
    axb.axhline(0.5, color=MUTE, lw=0.8, ls='--')
    axb.set_xlabel('ESM2 cosine to nearest TRAIN TF')
    axb.set_ylabel('per-TF disc_resid')
    axb.set_title(f'(b) disc vs ESM neighbor  r={r:.2f}', loc='left', fontweight='bold', fontsize=11)
    for s in ('top', 'right'): axb.spines[s].set_visible(False)

    # (c) ESM 2D embedding colored by disc
    axc = fig.add_subplot(gs[1, 0])
    sc = axc.scatter(esm2d[:, 0], esm2d[:, 1], c=disc, cmap=CMAP, s=55,
                     edgecolor='white', linewidth=0.6, vmin=disc.min(), vmax=disc.max())
    # mark silent with ring
    si = np.where(silent)[0]
    axc.scatter(esm2d[si, 0], esm2d[si, 1], facecolor='none', edgecolor=INK, s=120, linewidth=1.0)
    axc.set_xlabel('ESM PC1'); axc.set_ylabel('ESM PC2')
    axc.set_title('(c) ESM space, colored by disc\n(ring = silent in control)', loc='left', fontweight='bold', fontsize=11)
    cb = fig.colorbar(sc, ax=axc, fraction=0.046, pad=0.04); cb.set_label('disc_resid', fontsize=8)
    for s in ('top', 'right'): axc.spines[s].set_visible(False)

    # (d) correlation bar (|pearson r|)
    axd = fig.add_subplot(gs[1, 1])
    names = [c[0] for c in corr_rows]; rs = [c[1] for c in corr_rows]
    orderr = np.argsort(np.abs(rs))
    cc = [HIGH_C if rs[i] > 0 else LOW_C for i in orderr]
    axd.barh(range(len(names)), [rs[i] for i in orderr], color=cc, edgecolor='none')
    axd.axvline(0, color=INK, lw=0.8)
    axd.set_yticks(range(len(names))); axd.set_yticklabels([names[i] for i in orderr], fontsize=8)
    axd.set_xlabel('Pearson r with per-TF disc')
    axd.set_title('(d) Feature correlations', loc='left', fontweight='bold', fontsize=11)
    for s in ('top', 'right'): axd.spines[s].set_visible(False)

    # (e) disc split silent vs expressed (box/strip)
    axe = fig.add_subplot(gs[1, 2])
    groups = [disc[~silent], disc[silent]]
    labels = [f'expressed\n(n={(~silent).sum()})', f'silent\n(n={silent.sum()})']
    for j, (gd, cl) in enumerate(zip(groups, [HIGH_C, LOW_C])):
        x = np.random.normal(j, 0.06, len(gd))
        axe.scatter(x, gd, color=cl, s=30, alpha=0.7, edgecolor='white', linewidth=0.4)
        axe.hlines(gd.mean(), j-0.2, j+0.2, color=INK, lw=2)
    axe.axhline(0.5, color=MUTE, lw=0.8, ls='--')
    axe.set_xticks([0, 1]); axe.set_xticklabels(labels)
    axe.set_ylabel('per-TF disc_resid')
    axe.set_title('(e) disc by control silence', loc='left', fontweight='bold', fontsize=11)
    for s in ('top', 'right'): axe.spines[s].set_visible(False)

    fig.suptitle('Held-out TF discrimination variance diagnosis  |  v2b_retrfix_refresh_k20 (masked_go+L0+contrastive+retrieval, structural do-op)',
                 fontsize=12, fontweight='bold', x=0.01, ha='left')
    out = 'results/figures/tf_variance_diagnosis.png'
    os.makedirs('results/figures', exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nwrote {out}", flush=True)


if __name__ == '__main__':
    main()
