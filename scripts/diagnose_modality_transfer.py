"""
Diagnose CRISPRi (Replogle, train) → CRISPRa (Norman, eval) transfer failure.

Companion to CODE_REVIEW.md §5. Tests the hypothesis that the additive baseline
beats the Causal VAE on Norman combos because the model injects CRISPRi-trained
operators (D, A) into CRISPRa predictions, while additive is modality-agnostic.

Four diagnostics
----------------
1. u_clip_max sweep (the key test). In scheme='u_score', u_clip_max controls how
   much the do-operator engages the learned DAG A:
       low  u_clip_max  -> more A involvement (SCM propagates the intervention)
       high u_clip_max  -> A bypassed (program pinned directly)
   If model combo-Pearson IMPROVES as u_clip_max -> 1 (less A), the transferred
   CRISPRi graph is actively hurting on CRISPRa. Additive is u-independent, so it
   is the flat reference line.

2. Direction accuracy on Norman combos. Systematic sign errors ⇒ knockdown→
   activation sign mis-transfer.

3. Magnitude calibration. Slope of observed-vs-predicted log-FC on DEGs. Slope
   far from 1 (shrinkage / wrong sign) quantifies the scale mismatch.

4. Modality-token probe. The eval path never passes modality_id (encoder.py:73-75
   silently defaults to modality 0). If the model is multi-modal, this measures
   how different the epsilon / prediction would be had Norman been encoded under a
   non-zero modality — i.e. the size of the conditioning bug.

Usage
-----
    python scripts/diagnose_modality_transfer.py \
        --checkpoint checkpoints/phase1_best.pt \
        --norman_path /large_storage/goodarzilab/shervin/scCausalRL/Norman/ \
        --u_clip_max_sweep 0.25 0.5 0.75 1.0 \
        --output_dir results/modality_diagnostic

Notes
-----
* Read-only: does not modify the model or any core eval code.
* The model loader mirrors run_norman_validation.py (infers n_modalities /
  n_batches from the state dict).
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

from models.causal_vae import CausalVAE
from data.load_norman_aligned import load_norman_aligned
from evaluation.eval_combinatorial import predict_combo
from evaluation.run_perturbation_eval import compute_observed_de
from utils.metrics import pearson_on_degs

DEG_THRESH = 0.05


# ── model loading (mirrors run_norman_validation.py) ─────────────────────────
def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    target_gene_names = ckpt['gene_names']
    n_genes = len(target_gene_names)
    cfg_m = config['model']
    sd = ckpt['model_state_dict']
    n_batches = int(sd['encoder.batch_emb.weight'].shape[0]) if 'encoder.batch_emb.weight' in sd else 0
    n_modalities = int(sd['encoder.modality_emb.weight'].shape[0]) if 'encoder.modality_emb.weight' in sd else 0
    model = CausalVAE(
        n_genes=n_genes, latent_dim=cfg_m['latent_dim'],
        tau_init=cfg_m['tau_min'], n_dag_samples=cfg_m['n_dag_samples'],
        dagma_s=cfg_m['dagma_s'],
        n_modalities=n_modalities, n_batches=n_batches,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')}); "
          f"n_genes={n_genes}, n_modalities={n_modalities}, n_batches={n_batches}")
    return model, config, target_gene_names, n_modalities


# ── metric helpers ───────────────────────────────────────────────────────────
def _calibration_slope(pred_de, obs_lfc, mask):
    """OLS slope of observed ~ predicted on the masked (DEG) entries.

    slope ≈ 1 ⇒ well-calibrated magnitude; |slope| < 1 ⇒ shrinkage;
    slope < 0 ⇒ sign inverted (the CRISPRi→CRISPRa failure mode)."""
    p, o = pred_de[mask], obs_lfc[mask]
    if len(p) < 3 or np.std(p) < 1e-8:
        return np.nan
    return float(np.polyfit(p, o, 1)[0])


def _combo_metrics(pred_de, obs_lfc, obs_sig, obs_effect):
    """AUROC (DEG detection), Pearson on DEGs, direction acc, calibration slope."""
    out = {}
    n_sig = int(obs_sig.sum())
    if 3 <= n_sig < len(obs_sig):
        try:
            out['auroc'] = roc_auc_score(obs_sig.astype(int), np.abs(pred_de))
        except ValueError:
            pass
    if len(pred_de) > 10:
        rho, _ = spearmanr(np.abs(pred_de), obs_effect)
        out['spearman'] = rho
    out['pearson_deg'] = pearson_on_degs(pred_de, obs_lfc, threshold=DEG_THRESH)
    if n_sig >= 3:
        out['direction_acc'] = float(
            (np.sign(pred_de[obs_sig]) == np.sign(obs_lfc[obs_sig])).mean())
    out['calib_slope'] = _calibration_slope(pred_de, obs_lfc, np.abs(obs_lfc) > DEG_THRESH)
    return out


def _agg(rows, key):
    v = [r[key] for r in rows if key in r and r[key] is not None and not np.isnan(r[key])]
    return float(np.mean(v)) if v else np.nan


# ── diagnostic 1+2+3: u_clip_max sweep with direction + calibration ──────────
def run_u_sweep(model, X, data_dict, de_single, de_combo, gene_names, ctrl_mean,
                device, u_values, n_dag_samples):
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    de_single_adata = data_dict['adata'].uns.get('de_single', {})
    control_idx = np.array(data_dict['control_idx'])
    rng = np.random.default_rng(42)
    ctrl_sample = rng.choice(len(control_idx), min(20, len(control_idx)), replace=False)
    x_ctrl_t = torch.tensor(X[control_idx[ctrl_sample]], dtype=torch.float32, device=device)

    # Additive baseline is u-independent — compute once.
    additive_rows = []
    for (ga, gb), de in de_combo.items():
        if ga not in gene_to_idx or gb not in gene_to_idx:
            continue
        eff_a = np.asarray(de_single_adata.get(ga, de_single.get(ga, {}).get('lfc', np.zeros(len(gene_names)))))
        eff_b = np.asarray(de_single_adata.get(gb, de_single.get(gb, {}).get('lfc', np.zeros(len(gene_names)))))
        add_de = eff_a + eff_b
        additive_rows.append(_combo_metrics(add_de, de['lfc'], de['sig'], de['effect']))
    additive_summary = {f'additive_{k}': _agg(additive_rows, k)
                        for k in ['auroc', 'spearman', 'pearson_deg', 'direction_acc', 'calib_slope']}

    sweep = []
    for u in u_values:
        model_rows = []
        for (ga, gb), de in de_combo.items():
            if ga not in gene_to_idx or gb not in gene_to_idx:
                continue
            dx_a = de_single_adata.get(ga, de_single.get(ga, {}).get('lfc'))
            dx_b = de_single_adata.get(gb, de_single.get(gb, {}).get('lfc'))
            if dx_a is None or dx_b is None:
                continue
            dx_a_t = torch.tensor(np.asarray(dx_a, dtype=np.float32), device=device)
            dx_b_t = torch.tensor(np.asarray(dx_b, dtype=np.float32), device=device)
            with torch.no_grad():
                mean_AB, _, _, _ = predict_combo(
                    model, x_ctrl_t, gene_to_idx[ga], gene_to_idx[gb],
                    dx_a=dx_a_t, dx_b=dx_b_t, n_samples=n_dag_samples, device=device,
                    scheme='u_score', u_clip_max=u, combo_mode='parallel')
            pred_de = mean_AB.mean(axis=0) - ctrl_mean
            model_rows.append(_combo_metrics(pred_de, de['lfc'], de['sig'], de['effect']))
        row = {'u_clip_max': u, 'n_combos': len(model_rows)}
        for k in ['auroc', 'spearman', 'pearson_deg', 'direction_acc', 'calib_slope']:
            row[f'model_{k}'] = _agg(model_rows, k)
        row.update(additive_summary)
        sweep.append(row)
        print(f"  u={u:<4}  model: AUROC={row['model_auroc']:.3f} "
              f"Pearson={row['model_pearson_deg']:.3f} dir={row['model_direction_acc']:.3f} "
              f"slope={row['model_calib_slope']:.3f}   "
              f"| additive: AUROC={additive_summary['additive_auroc']:.3f} "
              f"Pearson={additive_summary['additive_pearson_deg']:.3f}")
    return pd.DataFrame(sweep)


# ── diagnostic 4: modality-token probe ───────────────────────────────────────
def probe_modality(model, X, data_dict, device, n_modalities):
    if n_modalities <= 1:
        print("  Model is single-modality (n_modalities <= 1) — eval path is fine, "
              "no modality-token bug. Skipping probe.")
        return None
    control_idx = np.array(data_dict['control_idx'])
    rng = np.random.default_rng(0)
    idx = control_idx[rng.choice(len(control_idx), min(256, len(control_idx)), replace=False)]
    x = torch.tensor(X[idx], dtype=torch.float32, device=device)
    out = {}
    with torch.no_grad():
        _, _, eps0 = model.encoder(x, modality_id=torch.zeros(x.shape[0], dtype=torch.long, device=device))
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        dec0 = model.decoder(model.dag.scm_forward(eps0, A))
        for m in range(1, n_modalities):
            mid = torch.full((x.shape[0],), m, dtype=torch.long, device=device)
            _, _, epsm = model.encoder(x, modality_id=mid)
            decm = model.decoder(model.dag.scm_forward(epsm, A))
            out[f'eps_l2_mod0_vs_mod{m}'] = float((eps0 - epsm).pow(2).sum(-1).sqrt().mean())
            out[f'decoded_l2_mod0_vs_mod{m}'] = float((dec0 - decm).pow(2).sum(-1).sqrt().mean())
    print("  Modality-token probe (eval encodes Norman as modality 0 by default):")
    for k, v in out.items():
        print(f"    {k} = {v:.4f}")
    print("    ^ nonzero ⇒ the silent default-to-modality-0 materially changes the "
          "encoding; pass the correct modality_id at eval.")
    return out


def plot_sweep(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in [(axes[0], 'pearson_deg', 'Pearson on DEGs'),
                              (axes[1], 'auroc', 'AUROC (DEG detection)')]:
        ax.plot(df['u_clip_max'], df[f'model_{metric}'], 'o-', color='#2171b5',
                lw=2, label='Causal VAE')
        add_val = df[f'additive_{metric}'].iloc[0]
        ax.axhline(add_val, color='#74c476', ls='--', lw=2,
                   label=f'Additive ({add_val:.3f}, u-indep.)')
        ax.set(xlabel='u_clip_max  (← more DAG involvement | less →)',
               ylabel=label, title=label)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('CRISPRi→CRISPRa: does engaging the learned DAG help or hurt on Norman combos?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'u_clip_max_sweep.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def main():
    ap = argparse.ArgumentParser(description="CRISPRi→CRISPRa transfer diagnostic")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--norman_path',
                    default='/large_storage/goodarzilab/shervin/scCausalRL/Norman/')
    ap.add_argument('--u_clip_max_sweep', type=float, nargs='+',
                    default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--n_dag_samples', type=int, default=16)
    ap.add_argument('--output_dir', default='results/modality_diagnostic')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")

    print("\n[1/4] Loading model")
    model, config, gene_names, n_modalities = load_model(args.checkpoint, device)

    print("\n[2/4] Loading Norman (aligned to checkpoint gene axis)")
    data_dict = load_norman_aligned(args.norman_path, gene_names)
    X = data_dict['adata'].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    print("\n[3/4] Observed DE + u_clip_max sweep (diagnostics 1-3)")
    de_single, de_combo, ctrl_mean = compute_observed_de(X, data_dict, gene_names)
    if len(de_combo) == 0:
        print("  No combo perturbations found — cannot run sweep. Exiting.")
        return
    sweep_df = run_u_sweep(model, X, data_dict, de_single, de_combo, gene_names,
                           ctrl_mean, device, args.u_clip_max_sweep, args.n_dag_samples)
    sweep_df.to_csv(os.path.join(args.output_dir, 'u_clip_max_sweep.csv'), index=False)
    plot_sweep(sweep_df, args.output_dir)

    print("\n[4/4] Modality-token probe (diagnostic 4)")
    probe = probe_modality(model, X, data_dict, device, n_modalities)
    if probe is not None:
        pd.DataFrame([probe]).to_csv(
            os.path.join(args.output_dir, 'modality_probe.csv'), index=False)

    # ── interpretation hint ──────────────────────────────────────────────
    best = sweep_df.loc[sweep_df['model_pearson_deg'].idxmax()]
    worst = sweep_df.loc[sweep_df['model_pearson_deg'].idxmin()]
    print("\n" + "=" * 64)
    print("INTERPRETATION")
    print("=" * 64)
    print(f"  Model Pearson is best at u_clip_max={best['u_clip_max']} "
          f"({best['model_pearson_deg']:.3f}) and worst at "
          f"u_clip_max={worst['u_clip_max']} ({worst['model_pearson_deg']:.3f}).")
    if best['u_clip_max'] >= worst['u_clip_max']:
        print("  → Performance rises as the DAG is bypassed (higher u_clip_max). "
              "This is evidence the CRISPRi-trained graph hurts CRISPRa prediction "
              "(CODE_REVIEW.md §5).")
    else:
        print("  → Engaging the DAG (lower u_clip_max) helps — the learned structure "
              "transfers. The additive gap is then NOT primarily a transfer artifact.")
    add_p = sweep_df['additive_pearson_deg'].iloc[0]
    print(f"  Additive Pearson (u-independent) = {add_p:.3f}; "
          f"model best = {best['model_pearson_deg']:.3f}.")
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
