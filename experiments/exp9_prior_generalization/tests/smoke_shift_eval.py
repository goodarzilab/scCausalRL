"""Synthetic CPU smoke for the shift-pathway eval core (no data load).

Exercises evaluate_single_perts_shift (fixed + c-search, distributional metrics)
and coexp_neighbour_baseline on a tiny in-memory model + fake data, asserting the
returned metrics are finite and well-formed.

Usage: python tests/smoke_shift_eval.py [--decoder linear|mlp]
"""
import os, sys, argparse, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from models.causal_vae import CausalVAE
from evaluation.run_shift_eval import evaluate_single_perts_shift, coexp_neighbour_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoder', choices=['linear', 'mlp'], default='linear')
    args = ap.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    device = 'cpu'
    G, d, label_dim = 120, 16, 8
    n_ctrl, n_per = 120, 80

    model = CausalVAE(n_genes=G, latent_dim=d, tau_init=0.25, n_dag_samples=8,
                      dagma_s=1.0, decoder_type=args.decoder, decoder_hidden=64,
                      decoder_layers=2, use_shift_encoder=True, shift_label_dim=label_dim,
                      shift_hidden=64).to(device)
    model.set_pert_labels(torch.randn(G, label_dim))
    model.eval()

    # Fake data: control block + 3 perturbations, each shifting a few genes.
    genes = [f'g{i}' for i in range(G)]
    X = rng.normal(0, 1, size=(n_ctrl + 3 * n_per, G)).astype('float32')
    ctrl_idx = list(range(n_ctrl))
    single_perts, de_dict, de_train = {}, {}, {}
    ctrl_mean = X[ctrl_idx].mean(0)
    for k, gname in enumerate(['g3', 'g7', 'g11']):
        idxs = list(range(n_ctrl + k * n_per, n_ctrl + (k + 1) * n_per))
        X[idxs] += rng.normal(0, 0.5, size=G)  # gene-specific shift
        single_perts[gname] = idxs
        lfc = X[idxs].mean(0) - ctrl_mean
        effect = np.abs(lfc) / (X[ctrl_idx].std(0) + 1e-6)
        sig = np.abs(lfc) > np.quantile(np.abs(lfc), 0.7)  # ~30% "DEGs"
        rec = {'lfc': lfc.astype('float32'), 'effect': effect.astype('float32'),
               'sig': sig, 'n_cells': len(idxs)}
        de_dict[gname] = rec
        de_train[gname] = rec  # also serve as the neighbour pool

    adata = types.SimpleNamespace(uns={'de_single': {}})
    data_dict = {'control_idx': ctrl_idx, 'single_perts': single_perts, 'adata': adata}

    for cm in ['fixed', 'search']:
        res = evaluate_single_perts_shift(
            model, X, data_dict, de_dict, genes, device,
            n_ctrl_cells=80, c_mode=cm, n_dist_cells=40, seed=1)
        assert len(res) == 3, f"expected 3 perts, got {len(res)}"
        for r in res:
            for key in ('pearson_deg', 'mse_mean', 'chosen_c', 'mmd_model',
                        'mmd_control', 'energy_model', 'energy_control'):
                assert key in r, f"missing {key} (c_mode={cm})"
                assert np.isfinite(r[key]), f"{key}={r[key]} not finite (c_mode={cm})"
        print(f"  c_mode={cm}: "
              f"AUROC={np.nanmean([r.get('auroc', np.nan) for r in res]):.3f} "
              f"pear_deg={np.nanmean([r['pearson_deg'] for r in res]):.3f} "
              f"chosen_c={[r['chosen_c'] for r in res]} "
              f"mmd(model/ctrl)={np.mean([r['mmd_model'] for r in res]):.4f}/"
              f"{np.mean([r['mmd_control'] for r in res]):.4f}")

    coexp = coexp_neighbour_baseline(model, de_train, de_dict, genes, k=2)
    assert len(coexp) == 3
    print(f"  coexp baseline: {len(coexp)} rows, "
          f"pear_deg={np.nanmean([r['pearson_deg'] for r in coexp]):.3f}")
    print(f"SMOKE OK (eval core, decoder={args.decoder})")


if __name__ == '__main__':
    main()
