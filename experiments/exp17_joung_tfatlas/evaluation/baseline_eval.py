"""Non-learned baselines for the Joung TF-atlas eval, scored on the SAME metrics as the
model evals (heldout_zeroshot_eval.py / combo_triple_eval.py) so the model's value-add is
readable directly. Two canonical perturbation-prediction baselines, both in Δ-space:

  context_mean : predict the (unperturbed) control profile  ->  Δ_pred = 0          (trivial floor)
  perturb_mean : predict the mean TRAINING perturbation     ->  Δ_pred = drift       (shared-direction floor)
                 drift = mean over training single-TF Δ (leak-free: train split only)

Why these matter for THIS eval:
  - perturb_mean usually scores HIGH on raw pearson_delta (all TF over-expressions share a
    transcriptional direction) — that's exactly why the *_resid (drift-removed) and
    discrimination metrics are the load-bearing ones. By construction perturb_mean has ZERO
    residual signal and CANNOT discriminate (identical prediction for every TF -> chance).
  - context_mean is the Δ=0 floor (raw pearson undefined; meaningful via mse_delta).
A useful model must beat perturb_mean on *_resid + discrimination, not just raw pearson.

Run (high-mem node):  TAG=baselines sbatch scripts/baseline_eval.sbatch
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from evaluation.heldout_zeroshot_eval import gene_space, fate_space, zscore_rows
from evaluation.combo_triple_eval import parse_tfs, COMBO


def mse_delta(P, O):
    return float(np.mean(((P - O) ** 2).mean(1)))


def score(pred, obs, train_obs, gene_names, do_fate=True):
    """gene-space + drift-removed residual + (optional) developmental-fate, like the model eval."""
    drift = train_obs.mean(0, keepdims=True)
    gs, *_ = gene_space(pred, obs)
    gr, *_ = gene_space(pred - drift, obs - drift)
    out = {**{k: v for k, v in gs.items()},
           **{f'{k}_resid': v for k, v in gr.items() if k != 'n'},
           'mse_delta': mse_delta(pred, obs)}
    if do_fate and len(pred) >= 3:
        try:
            fs, *_ = fate_space(pred, obs, train_obs, gene_names)
            out.update(fs)
        except Exception as e:
            out['fate_error'] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='baselines')
    ap.add_argument('--min_cells', type=int, default=20)
    args = ap.parse_args()
    out = os.path.join(ROOT, 'results', 'baseline_eval', args.tag); os.makedirs(out, exist_ok=True)

    from data.load_joung import load_joung, _lognorm
    import anndata as ad, scipy.sparse as sp, pandas as pd
    dd = load_joung(n_hvg=None)
    gene_names = dd['gene_names']; G = len(gene_names)
    de = dd['de_single']
    test = [g for g in dd['pert_split']['test'] if g in de]
    train = [g for g in dd['pert_split']['train'] if g in de]
    obs_s = np.stack([np.asarray(de[g], np.float32) for g in test])
    train_obs = np.stack([np.asarray(de[g], np.float32) for g in train])
    drift = train_obs.mean(0)
    print(f"[baseline] singles: test={len(test)} train={len(train)} genes={G}", flush=True)

    summary = {'tag': args.tag, 'n_test_singles': len(test)}
    gidx = {g: i for i, g in enumerate(gene_names)}

    # ---- control-correlation features (for corr + coexp-neighbour baselines) ----
    Xc = dd['adata'].X; Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
    ctrl = np.array(dd['control_idx'])
    rng = np.random.default_rng(0)
    csel = ctrl[rng.choice(len(ctrl), min(5000, len(ctrl)), replace=False)]
    Zc = Xc[csel].astype(np.float32)
    Zc = (Zc - Zc.mean(0)) / (Zc.std(0) + 1e-8)             # z-scored genes over control cells
    nC = Zc.shape[0]
    train_ti = np.array([gidx[g] for g in train])

    def corr_pred(ti):
        """correlation baseline: Δ_pred[g] = corr_control(g, TF_gene); self masked."""
        p = (Zc.T @ Zc[:, ti]) / nC
        p[ti] = 0.0
        return p.astype(np.float32)

    def coexp_pred(ti, k=10):
        """coexp-neighbour baseline: mean OBSERVED Δ of the k training TFs whose gene is most
        co-expressed (control corr) with the held-out TF gene (zero-shot prior-knowledge comparator)."""
        sims = (Zc[:, train_ti].T @ Zc[:, ti]) / nC
        nbrs = [j for j in np.argsort(-sims) if train_ti[j] != ti][:k]
        return train_obs[nbrs].mean(0).astype(np.float32)

    # ---- SINGLES ----
    summary['singles'] = {
        'context_mean': score(np.zeros((len(test), G), np.float32), obs_s, train_obs, gene_names),
        'perturb_mean': score(np.tile(drift, (len(test), 1)).astype(np.float32), obs_s, train_obs, gene_names),
        'corr':         score(np.stack([corr_pred(gidx[g]) for g in test]), obs_s, train_obs, gene_names),
        'coexpression': score(np.stack([coexp_pred(gidx[g]) for g in test]), obs_s, train_obs, gene_names),
    }

    # ---- COMBOS ---- (observed Δ exactly as combo_triple_eval.py: same lognorm frame + control baseline)
    ctrl_mean_combo = Xc[ctrl].mean(0)
    A2 = ad.read_h5ad(COMBO); A2.X = _lognorm(A2.X)
    Xk = A2.X.toarray() if sp.issparse(A2.X) else np.asarray(A2.X)
    tf_set = A2.obs['TF_set'].astype(str).values
    rows = []
    for setname in sorted(set(tf_set)):
        tfs = parse_tfs(setname)
        gis = [gidx[t] for t in tfs if t in gidx]
        m = tf_set == setname
        if len(gis) >= 2 and m.sum() >= args.min_cells:
            rows.append({'tfs': tfs, 'gis': gis, 'k': len(gis), 'obs': Xk[m].mean(0) - ctrl_mean_combo})
    ks = np.array([r['k'] for r in rows])
    print(f"[baseline] combos: usable sets={len(rows)} (pairs={int((ks==2).sum())}, triples={int((ks>=3).sum())})", flush=True)

    def additive_obs(tfs):
        """Additive null: sum of the OBSERVED single-TF Δ for the TFs in the set (no epistasis)."""
        comp = [np.asarray(de[t], np.float32) for t in tfs if t in de]
        return np.sum(comp, axis=0) if comp else np.zeros(G, np.float32)
    def corr_combo(gis):  return np.sum([corr_pred(t) for t in gis], axis=0).astype(np.float32)
    def coexp_combo(gis): return np.sum([coexp_pred(t) for t in gis], axis=0).astype(np.float32)

    summary['n_combos'] = int(len(rows))
    summary['combos'] = {}
    for kind, mask in [('all', np.ones(len(rows), bool)), ('pairs', ks == 2), ('triples', ks >= 3)]:
        if mask.sum() < 2:
            continue
        sel = [r for r, mk in zip(rows, mask) if mk]
        O = np.stack([r['obs'] for r in sel])
        df = bool(mask.sum() >= 3)
        summary['combos'][kind] = {
            'n': int(mask.sum()),
            'context_mean': score(np.zeros_like(O), O, train_obs, gene_names, do_fate=df),
            'perturb_mean': score(np.tile(drift, (len(O), 1)).astype(np.float32), O, train_obs, gene_names, do_fate=df),
            'additive':     score(np.stack([additive_obs(r['tfs']) for r in sel]).astype(np.float32), O, train_obs, gene_names, do_fate=df),
            'corr':         score(np.stack([corr_combo(r['gis']) for r in sel]), O, train_obs, gene_names, do_fate=df),
            'coexpression': score(np.stack([coexp_combo(r['gis']) for r in sel]), O, train_obs, gene_names, do_fate=df),
        }

    json.dump(summary, open(f"{out}/summary.json", 'w'), indent=2)
    print("\n===== BASELINES (" + args.tag + ") =====")
    for split, d in [('SINGLES', summary['singles']), ('COMBOS/all', summary['combos'].get('all', {}))]:
        print(f"\n[{split}]")
        for name, m in d.items():
            if not isinstance(m, dict):
                continue
            print(f"  {name:13s} pearsonΔ={m.get('pearson_delta', float('nan')):.3f}  "
                  f"pΔ_resid={m.get('pearson_delta_resid', float('nan')):.3f}  "
                  f"discrim={m.get('discrimination_cos', float('nan')):.3f}  "
                  f"disc_resid={m.get('discrimination_cos_resid', float('nan')):.3f}  "
                  f"mse={m.get('mse_delta', float('nan')):.4f}  "
                  f"fate_top1={m.get('fate_top1', float('nan')):.3f}")
    print(f"\nsaved → {out}/")


if __name__ == '__main__':
    main()
