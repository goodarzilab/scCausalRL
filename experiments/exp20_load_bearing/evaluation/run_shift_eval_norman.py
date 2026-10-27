"""External validation of a shift-encoder model on Norman (cross-dataset).

Takes a K562_essential-trained shift checkpoint and evaluates it ZERO-SHOT on
Norman perturbations: Norman is aligned onto the checkpoint's exact gene axis
(data/load_norman_aligned), and every Norman single perturbation whose target
gene exists in that axis is scored through the SAME shift pathway as the in-dist
eval (S^q = c·ShiftEncoder(pert_label[q]) → (I−A^T)⁻¹(ε+S^q) → decode).

IMPORTANT modality caveat: K562_essential is CRISPRi (knockdown); Norman is
CRISPRa (activation). So:
  • downstream-DEG **AUROC** (|pred_de| vs observed DEGs — DIRECTION-AGNOSTIC) is
    the meaningful external metric: does the model identify the right regulatory
    targets of gene q, regardless of sign?
  • Pearson-on-DEGs / direction-accuracy are CONFOUNDED by the CRISPRi→CRISPRa
    sign flip. A strongly NEGATIVE Pearson is actually evidence the model learned
    real directional regulation (knockdown-trained shift, activation observed →
    opposite sign). We report both raw and sign-flipped Pearson for that reason.

Baseline: gene-gene correlation from Norman control cells (same AUROC metric).

Usage:
    python evaluation/run_shift_eval_norman.py \
        --checkpoint results/phase1/checkpoints/phase1_epoch150.pt \
        --config configs/exp2_shift_linear.yaml \
        --output_dir results/phase1/norman_eval --c_mode both
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
from evaluation.run_perturbation_eval import (
    compute_observed_de, compute_correlation_matrix, evaluate_baseline,
    _safe_mean, _extract,
)
from evaluation.run_shift_eval import evaluate_single_perts_shift


def load_shift_model(checkpoint_path, device):
    """Build the shift model from the checkpoint's own saved config + state dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict']
    cfg_m = (ckpt.get('config', {}) or {})['model']
    gene_names = ckpt['gene_names']
    n_genes = len(gene_names)
    n_batches = int(sd['encoder.batch_emb.weight'].shape[0]) if 'encoder.batch_emb.weight' in sd else 0
    n_modalities = int(sd['encoder.modality_emb.weight'].shape[0]) if 'encoder.modality_emb.weight' in sd else 0
    # SENA-δ encoder reload: detect from the saved masked layer, infer K, hand a
    # zero placeholder mask (the real one is a persistent buffer restored by load_state_dict).
    encoder_type = cfg_m.get('encoder_type', 'sena' if 'encoder.W' in sd else 'mlp')
    sena_mask = None
    if encoder_type == 'sena':
        K = int(sd['encoder.W'].shape[1])
        sena_mask = torch.zeros(n_genes, K)
    # SENA-δ masked-learnable decoder reload: detect the saved dec_mask buffer, hand a zero
    # placeholder so the buffer is registered; load_state_dict restores the real mask + weights.
    decoder_mask = None
    if 'decoder.dec_mask' in sd:
        decoder_mask = torch.zeros(n_genes, int(sd['decoder.dec_mask'].shape[1]))
    # Context-conditioned WM heads (exp20 condwm): detect from the state dict so the constructed model has the
    # dataset_clf (GRL adversary) + film_emb (per-dataset FiLM) params → strict load succeeds.
    adv_invariance = any(k.startswith('dataset_clf') for k in sd)
    adv_hidden = int(sd['dataset_clf.0.weight'].shape[0]) if 'dataset_clf.0.weight' in sd else 256
    film_condition = ('film_emb.weight' in sd)
    model = CausalVAE(
        n_genes=n_genes, latent_dim=cfg_m['latent_dim'],
        tau_init=cfg_m.get('tau_min', cfg_m.get('tau_init', 1.0)),
        n_dag_samples=cfg_m.get('n_dag_samples', 16), dagma_s=cfg_m.get('dagma_s', 1.0),
        n_modalities=n_modalities, n_batches=n_batches,
        decoder_type=cfg_m.get('decoder_type', 'linear'),
        decoder_hidden=int(cfg_m.get('decoder_hidden', 1024)),
        decoder_layers=int(cfg_m.get('decoder_layers', 2)),
        use_shift_encoder=bool(cfg_m.get('use_shift_encoder', True)),
        shift_label_dim=int(cfg_m.get('shift_label_dim', 50)),
        shift_hidden=int(cfg_m.get('shift_hidden', 1024)),
        shift_n_layers=int(cfg_m.get('shift_n_layers', 1)),
        shift_output_activation=cfg_m.get('shift_output_activation', 'none'),
        separate_pert_decoder=bool(cfg_m.get('separate_pert_decoder', False)),
        pert_decoder_type=cfg_m.get('pert_decoder_type', 'linear'),
        pert_decoder_hidden=int(cfg_m.get('pert_decoder_hidden', 1024)),
        pert_decoder_layers=int(cfg_m.get('pert_decoder_layers', 2)),
        u_gate_mlp=bool(cfg_m.get('u_gate_mlp', False)),
        nonlinear_eps_mix=bool(cfg_m.get('nonlinear_eps_mix', False)),
        gate_hidden=int(cfg_m.get('gate_hidden', 256)),
        u_gate_pure=bool(cfg_m.get('u_gate_pure', False)),
        dag_disabled=bool(cfg_m.get('dag_disabled', False)),
        encoder_type=encoder_type,
        sena_mask=sena_mask,
        sena_activation=cfg_m.get('sena_activation', 'tanh'),
        pert_intervene_mode=cfg_m.get('pert_intervene_mode', 'legacy'),
        use_target_gate=bool(cfg_m.get('use_target_gate', False)),
        target_gate_tau=float(cfg_m.get('target_gate_tau', 1.0)),
        scm_nonlinear=bool(cfg_m.get('scm_nonlinear', False)),
        n_picard=int(cfg_m.get("scm_picard_iters", 40)),
        scm_solve_ridge=float(cfg_m.get("scm_solve_ridge", 0.0)),
        decoder_mask=decoder_mask,
        decoder_mask_freeze=bool(cfg_m.get('dec_mask_freeze', False)),
        interaction_rank=int(cfg_m.get('interaction_rank', 0)),
        # Exp1 (per-modality shift scale) + Exp3 (wider modality token): infer from the state dict so the
        # constructed model matches the checkpoint shapes exactly (config may be absent/stale otherwise).
        modality_shift_scale=('modality_shift' in sd),
        modality_dim=(int(sd['encoder.modality_emb.weight'].shape[1]) if 'encoder.modality_emb.weight' in sd
                      else int(cfg_m.get('modality_dim', 16))),
        separate_combo_decoder=any(k.startswith('combo_decoder') for k in sd),   # secondary combo head
        dose_scaled_shift=('dose_gain' in sd),                                    # REPROcode dose-scaled shift
        adv_invariance=adv_invariance, adv_hidden=adv_hidden,                     # condwm: GRL dataset adversary
        film_condition=film_condition,                                            # condwm: per-dataset FiLM
    ).to(device)
    # retrieval memory: register placeholders (train-only DB) sized from the state dict so load restores
    if 'retr_val' in sd:
        _p1 = (ckpt.get('config', {}) or {}).get('phase1', {}) or {}
        N_tr, dv = sd['retr_val'].shape
        model.install_retrieval(torch.zeros(N_tr, dtype=torch.long), torch.zeros(N_tr, dv),
                                key_dim=int(sd['retr_proj.weight'].shape[0]),
                                k=int(_p1.get('retrieval_k', 10)), tau=float(_p1.get('retrieval_tau', 1.0)))
    model.load_state_dict(sd)
    if ckpt.get('oracle_gate') is not None:                 # hard self-membership gate (localization arms)
        model.oracle_gate = torch.as_tensor(ckpt['oracle_gate'], dtype=torch.float32, device=device)
    model.eval()
    if not model.use_shift_encoder:
        raise RuntimeError("checkpoint has no shift encoder")
    print(f"Loaded shift model (epoch {ckpt.get('epoch','?')}, decoder={cfg_m.get('decoder_type','linear')}, "
          f"{n_genes} genes)")
    return model, gene_names, ckpt


def main():
    ap = argparse.ArgumentParser(description="External Norman validation (shift pathway)")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True, help="for norman_path (+ fallback model cfg)")
    ap.add_argument('--norman_path', default=None, help="override config's norman_path")
    ap.add_argument('--output_dir', default='results/norman_eval')
    ap.add_argument('--device', default=None)
    ap.add_argument('--c_mode', default='both', choices=['fixed', 'search', 'both'])
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--n_dist_cells', type=int, default=256)
    ap.add_argument('--min_cells_per_pert', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--wandb', action='store_true')
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    norman_path = args.norman_path or config['data']['norman_path']
    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device} | Norman: {norman_path}")

    model, gene_names, ckpt = load_shift_model(args.checkpoint, device)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Align Norman onto the checkpoint's gene axis.
    from data.load_norman_aligned import load_norman_aligned
    data_dict = load_norman_aligned(norman_path, target_gene_names=gene_names,
                                    min_cells_per_pert=args.min_cells_per_pert)
    import scipy.sparse
    X = data_dict['adata'].X
    X = (X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)).astype(np.float32)
    n_ext = len(data_dict['single_perts'])
    if n_ext == 0:
        raise RuntimeError("No evaluable Norman single perts overlap the model's gene axis.")
    print(f"External Norman single perts (gene in axis): {n_ext}")

    de_single, de_combo, ctrl_mean = compute_observed_de(X, data_dict, gene_names)
    X_ctrl = X[data_dict['control_idx']]
    corr_matrix = compute_correlation_matrix(X_ctrl, gene_names)

    c_modes = ['fixed', 'search'] if args.c_mode == 'both' else [args.c_mode]
    summary, all_rows = {}, {}
    rows_for_print = []
    for cm in c_modes:
        res = evaluate_single_perts_shift(
            model, X, data_dict, de_single, gene_names, device,
            n_ctrl_cells=args.n_ctrl_cells, c_mode=cm,
            n_dist_cells=args.n_dist_cells, seed=args.seed)
        # sign-flipped Pearson (CRISPRi-trained vs CRISPRa-observed)
        for r in res:
            r['pearson_deg_flip'] = -r.get('pearson_deg', np.nan) if r.get('pearson_deg') is not None else np.nan
        all_rows[f'model_{cm}'] = res
        rows_for_print.append((f"Shift model (c={cm})", res))
        summary[f'norman_model_{cm}_auroc'] = _safe_mean(_extract(res, 'auroc'))
        summary[f'norman_model_{cm}_pearson_deg'] = _safe_mean(_extract(res, 'pearson_deg'))
        summary[f'norman_model_{cm}_pearson_deg_flip'] = _safe_mean(_extract(res, 'pearson_deg_flip'))
        summary[f'norman_model_{cm}_mmd'] = _safe_mean(_extract(res, 'mmd_model'))
        summary[f'norman_model_{cm}_mmd_ctrl_floor'] = _safe_mean(_extract(res, 'mmd_control'))

    corr_res = evaluate_baseline(corr_matrix, gene_to_idx, de_single, gene_names)
    all_rows['corr'] = corr_res
    rows_for_print.append(("Correlation (Norman control)", corr_res))
    summary['norman_corr_auroc'] = _safe_mean(_extract(corr_res, 'auroc'))
    summary['n_external_perts'] = n_ext

    # ── Report ──
    print(f"\n{'='*64}\nEXTERNAL NORMAN (CRISPRa) — model trained on K562 CRISPRi\n{'='*64}")
    print(f"  {'Method':>30s} {'AUROC':>8s} {'Pear':>8s} {'Pear(flip)':>11s} {'N':>5s}")
    for name, res in rows_for_print:
        print(f"  {name:>30s} {_safe_mean(_extract(res,'auroc')):>8.4f} "
              f"{_safe_mean(_extract(res,'pearson_deg')):>8.4f} "
              f"{_safe_mean(_extract(res,'pearson_deg_flip')):>11.4f} {len(res):>5d}")
    for cm in c_modes:
        mr = all_rows[f'model_{cm}']
        print(f"  [c={cm}] MMD model/ctrl-floor = "
              f"{_safe_mean(_extract(mr,'mmd_model')):.4f} / {_safe_mean(_extract(mr,'mmd_control')):.4f}")

    for key, res in all_rows.items():
        if res:
            pd.DataFrame(res).to_csv(os.path.join(args.output_dir, f'norman_{key}.csv'), index=False)
    summary['checkpoint'] = args.checkpoint
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    print(f"\nSaved to {args.output_dir}/")
    print(f"\nHEADLINE (external Norman, AUROC is the direction-agnostic metric):")
    print(f"  shift model AUROC (c=1)     : {summary.get('norman_model_fixed_auroc', float('nan')):.4f}")
    if 'norman_model_search_auroc' in summary:
        print(f"  shift model AUROC (c-search): {summary['norman_model_search_auroc']:.4f}")
    print(f"  correlation baseline AUROC  : {summary.get('norman_corr_auroc', float('nan')):.4f}")
    print(f"  (Pearson confounded by CRISPRi→CRISPRa; flip sign for true-direction read)")

    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=config.get('wandb', {}).get('project', 'CausalBioRL'),
                             name=f"norman_eval_{os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))}",
                             reinit=True)
            run.log({f'norman/{k}': v for k, v in summary.items() if isinstance(v, (int, float))})
            run.finish()
        except Exception as e:
            print(f"W&B skipped ({e})")


if __name__ == '__main__':
    main()
