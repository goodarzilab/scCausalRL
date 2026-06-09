"""Norman combo-perturbation eval for a shift Causal-VAE checkpoint.

Predicts each Norman 2-gene combo (gA,gB) ZERO-SHOT through the shift pathway and
compares three predictors against the observed combo effect:

  1. shift_add   — sum of the two single-gene shift predictions.
       pred = ŝ(gA) + ŝ(gB),  where  ŝ(g) = decode((I−Aᵀ)⁻¹(ε+S_g)) − decode((I−Aᵀ)⁻¹ε)
     NOTE (key identity): for a LINEAR decoder + LINEAR SCM this is *exactly*
     the user's proposed composition  z = (I−Aᵀ)⁻¹(ε + S_A + S_B)  decoded — because
     both the solve and the decode are linear, decode∘solve distributes over the
     sum. We compute the composition form too and assert it matches (max|Δ| logged),
     to make explicit that the linear shift CANNOT produce non-additive epistasis.

  2. additive_obs — sum of the OBSERVED single effects (the standard Norman null
     model). Uses ground-truth singles, so it is an (oracle-ish) reference, not
     zero-shot. Tells us the headroom the additive assumption itself leaves.

  3. shift_uscore — the u-score mutilation do-operator (intervene_combo_u), fed from
     the shift's predicted single-gene gene-space deltas. This is the ONLY model
     predictor that can be non-additive: it solves (I − M·Aᵀ)⁻¹ with
     M=diag(1−u_combo), u_combo=clip(L[gA]+L[gB]); the pair's u differs from either
     single's, so the propagation through an active A adds regulatory epistasis.

Metrics per combo (vs observed combo lfc): downstream AUROC (direction-agnostic —
the meaningful CRISPRi→CRISPRa metric), Pearson-on-DEGs, Spearman(|effect|).
Also logs non-additivity ‖uscore−shift_add‖/‖shift_add‖ per combo.

Baseline: gene-gene correlation from Norman control cells (AUROC).

Usage:
  python evaluation/run_norman_combo_eval.py \
    --checkpoint results/grid/<cell>/checkpoints/phase1_epoch150.pt \
    --config configs/grid/<cell>.yaml --output_dir results/grid/<cell>/norman_combo
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from models.causal_vae import CausalVAE
from utils.metrics import pearson_on_degs
from evaluation.run_perturbation_eval import compute_observed_de, _safe_mean, _extract
from evaluation.run_shift_eval_norman import load_shift_model


def _scm_solve(LHS, rhs):
    return torch.linalg.solve(LHS, rhs.T).T


def _score(pred_de, obs_lfc, obs_effect, obs_sig, gi_mask):
    """AUROC / Pearson / Spearman of a predicted delta vs observed combo effect."""
    pred = pred_de[gi_mask]; obs_l = obs_lfc[gi_mask]
    obs_e = obs_effect[gi_mask]; obs_s = obs_sig[gi_mask]
    r = {}
    n_sig = int(obs_s.sum())
    if 3 <= n_sig < len(obs_s):
        try: r['auroc'] = roc_auc_score(obs_s.astype(int), np.abs(pred))
        except ValueError: pass
    if len(pred) > 10:
        r['spearman'] = spearmanr(np.abs(pred), obs_e)[0]
    r['pearson_deg'] = pearson_on_degs(pred_de, obs_lfc, threshold=0.05)
    r['pearson_deg_flip'] = -r['pearson_deg'] if r['pearson_deg'] == r['pearson_deg'] else np.nan
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--norman_path', default=None)
    ap.add_argument('--output_dir', default='results/norman_combo')
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--min_cells_per_pert', type=int, default=5)
    ap.add_argument('--u_clip_max', type=float, default=None,
                    help="override the model's u_clip_max for the shift_uscore mutilation operator")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import yaml
    config = yaml.safe_load(open(args.config))
    norman_path = args.norman_path or config['data']['norman_path']
    model, gene_names, ckpt = load_shift_model(args.checkpoint, device)
    if args.u_clip_max is not None:
        model.u_clip_max = float(args.u_clip_max)
        print(f"[override] u_clip_max = {model.u_clip_max}")
    # mutilation params from the checkpoint's training config (coherent eval): if the
    # model was trained with the mutilation operator, shift_uscore uses the SAME
    # operator/clamp it was trained with (intervene_shift_mutilate), not the loading
    # bolt-on. Falls back to loading-clamp for additively-trained models.
    _p1 = (ckpt.get('config', {}) or {}).get('phase1', {})
    MUT_TRAINED = _p1.get('pert_loss_mode', 'shift') == 'shift_mutilate'
    U_MODE = _p1.get('pert_mutilate_u_mode', 'loading')
    U_MAX = float(args.u_clip_max) if args.u_clip_max is not None else float(_p1.get('pert_mutilate_u_max', model.u_clip_max))
    BETA = float(_p1.get('pert_mutilate_beta', 1.0))
    print(f"[combo] mutilation-trained={MUT_TRAINED}  u_mode={U_MODE}  u_max={U_MAX}  beta={BETA}")
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names); d = model.latent_dim

    from data.load_norman_aligned import load_norman_aligned
    dd = load_norman_aligned(norman_path, target_gene_names=gene_names,
                             min_cells_per_pert=args.min_cells_per_pert)
    import scipy.sparse
    X = dd['adata'].X
    X = (X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)).astype(np.float32)
    de_single, de_combo, ctrl_mean = compute_observed_de(X, dd, gene_names)
    # combos with BOTH genes present + observed singles available (for additive_obs)
    de_combo = {k: v for k, v in de_combo.items()
                if k[0] in gene_to_idx and k[1] in gene_to_idx
                and k[0] in de_single and k[1] in de_single}
    print(f"Evaluable Norman combos: {len(de_combo)}")
    if not de_combo:
        raise RuntimeError("no evaluable combos")

    # ── control encode + eval DAG (shared) ──
    rng = np.random.default_rng(args.seed)
    ctrl_idx = np.array(dd['control_idx'])
    sel = rng.choice(len(ctrl_idx), min(args.n_ctrl_cells, len(ctrl_idx)), replace=False)
    x_ctrl_t = torch.tensor(X[ctrl_idx[sel]], dtype=torch.float32, device=device)
    corr_ctrl = np.corrcoef(X[ctrl_idx].T); corr_ctrl = np.nan_to_num(np.abs(corr_ctrl))
    np.fill_diagonal(corr_ctrl, 0.0)

    with torch.no_grad():
        _, _, eps_ctrl = model.encoder(x_ctrl_t)               # [Nc, d]
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        I_mat = torch.eye(d, device=device, dtype=A.dtype)
        LHS = I_mat - A.T
        z_prev = torch.clamp(_scm_solve(LHS, eps_ctrl), -20, 20)
        ctrl_recon = model.decoder(z_prev).mean(0).cpu().numpy()

        # cache single-gene shift predictions (latent z_S and gene-space delta)
        zS, pred_single = {}, {}
        need = set([g for k in de_combo for g in k])
        for g in need:
            gi = gene_to_idx[g]
            S = model.compute_shift(torch.tensor([gi], device=device))  # [1,d]
            z_s = torch.clamp(_scm_solve(LHS, S), -20, 20)              # [1,d]
            zS[g] = z_s
            x_hat = model.decoder(torch.clamp(z_prev + z_s, -20, 20)).mean(0).cpu().numpy()
            pred_single[g] = x_hat - ctrl_recon                        # gene-space delta

    rows = []
    max_compose_gap = 0.0
    for (ga, gb), de in de_combo.items():
        obs_lfc, obs_eff, obs_sig = de['lfc'], de['effect'], de['sig']
        gi_a, gi_b = gene_to_idx[ga], gene_to_idx[gb]
        mask = np.ones(n_genes, bool); mask[gi_a] = False; mask[gi_b] = False  # mask both targets

        # 1. shift_add (sum of single shift predictions)
        pred_shift_add = pred_single[ga] + pred_single[gb]

        # ---- composition-form sanity check: z=(I−Aᵀ)⁻¹(ε+S_A+S_B) decoded ----
        with torch.no_grad():
            z_comp = torch.clamp(z_prev + zS[ga] + zS[gb], -20, 20)
            pred_comp = model.decoder(z_comp).mean(0).cpu().numpy() - ctrl_recon
        max_compose_gap = max(max_compose_gap, float(np.max(np.abs(pred_comp - pred_shift_add))))

        # 2. additive_obs (observed singles — standard null)
        pred_add_obs = de_single[ga]['lfc'] + de_single[gb]['lfc']

        # 3. shift_uscore — shift-fed mutilation, COHERENT with training: uses
        #    intervene_shift_mutilate with the trained u_mode/clamp (the shift defines
        #    the target internally, the do-operator propagates it).
        # 4. uscore_obs — observed-fed loading mutilation (reference: operator quality
        #    given perfect singles, independent of the shift's single-prediction).
        with torch.no_grad():
            z_post_u, u_vec = model.intervene_shift_mutilate(
                eps_ctrl, [gi_a, gi_b], A, z_prev,
                u_mode=U_MODE, u_clip_max=U_MAX, shift_beta=BETA)
            pred_uscore = model.decoder(z_post_u).mean(0).cpu().numpy() - ctrl_recon
            u_combo = u_vec
            ox_a = torch.tensor(de_single[ga]['lfc'], dtype=torch.float32, device=device)
            ox_b = torch.tensor(de_single[gb]['lfc'], dtype=torch.float32, device=device)
            z_post_uo, _, _ = model.intervene_combo_u(
                eps_ctrl, [ox_a, ox_b], A, z_prev, [gi_a, gi_b], u_clip_max=U_MAX)
            pred_uscore_obs = model.decoder(z_post_uo).mean(0).cpu().numpy() - ctrl_recon

        # non-additivity of the u-score predictor
        nonadd = float(np.linalg.norm(pred_uscore - pred_shift_add) /
                       (np.linalg.norm(pred_shift_add) + 1e-8))

        # observed epistasis (genetic-interaction strength): how far the OBSERVED
        # combo deviates from observed-additive, on the combo's DEGs (targets masked).
        # 0 = perfectly additive; larger = more epistatic/non-linear. Independent of
        # any model prediction (the tertile axis).
        deg_eval = obs_sig & mask
        epistasis = np.nan; epi_add_pearson = np.nan
        if int(deg_eval.sum()) >= 3:
            resid = (obs_lfc - pred_add_obs)[deg_eval]
            epistasis = float(np.linalg.norm(resid) / (np.linalg.norm(obs_lfc[deg_eval]) + 1e-8))
            epi_add_pearson = float(spearmanr(obs_lfc[deg_eval], pred_add_obs[deg_eval])[0])

        r = {'combo': f'{ga}+{gb}', 'n_cells': de['n_cells'], 'n_de': int(obs_sig[mask].sum()),
             'u_combo_sum': float(u_combo.sum().item()), 'nonadditivity': nonadd,
             'epistasis': epistasis, 'epi_add_spearman': epi_add_pearson}
        for pname, pred in [('shift_add', pred_shift_add), ('additive_obs', pred_add_obs),
                            ('shift_uscore', pred_uscore), ('uscore_obs', pred_uscore_obs)]:
            for mk, mv in _score(pred, obs_lfc, obs_eff, obs_sig, mask).items():
                r[f'{pname}_{mk}'] = mv
        # correlation baseline (avg of the two genes' corr rows as the "combo signature")
        corr_sig = (corr_ctrl[gi_a] + corr_ctrl[gi_b])
        for mk, mv in _score(corr_sig, obs_lfc, obs_eff, obs_sig, mask).items():
            r[f'corr_{mk}'] = mv
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'combo_per_pert.csv'), index=False)

    methods = ['shift_add', 'additive_obs', 'shift_uscore', 'uscore_obs', 'corr']
    col = lambda c: (df[c].tolist() if c in df.columns else [])
    summary = {'n_combos': len(df), 'compose_vs_shiftadd_max_gap': max_compose_gap,
               'mean_nonadditivity': float(df['nonadditivity'].mean()),
               'checkpoint': args.checkpoint}
    print(f"\n{'='*72}\nNORMAN COMBO EVAL — {len(df)} combos  (AUROC = direction-agnostic)\n{'='*72}")
    print(f"  {'method':>14s} {'AUROC':>8s} {'Pear-DEG':>9s} {'Pear(flip)':>11s} {'Spear':>8s}")
    for m in methods:
        au = _safe_mean(col(f'{m}_auroc')); pe = _safe_mean(col(f'{m}_pearson_deg'))
        pf = _safe_mean(col(f'{m}_pearson_deg_flip')); sp = _safe_mean(col(f'{m}_spearman'))
        summary.update({f'{m}_auroc': au, f'{m}_pearson_deg': pe,
                        f'{m}_pearson_deg_flip': pf, f'{m}_spearman': sp})
        print(f"  {m:>14s} {au:>8.4f} {pe:>9.4f} {pf:>11.4f} {sp:>8.4f}")
    print(f"\n  composition-form vs shift_add  max|Δ| = {max_compose_gap:.2e}  "
          f"(≈0 confirms (I−Aᵀ)⁻¹(ε+S_A+S_B) ≡ sum of singles for linear decoder)")
    print(f"  mean u-score non-additivity ‖uscore−shift_add‖/‖shift_add‖ = {summary['mean_nonadditivity']:.3f}")
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    print(f"\nsaved → {args.output_dir}/")


if __name__ == '__main__':
    main()
