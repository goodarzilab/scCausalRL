"""Is the latent DAG load-bearing for PREDICTION, or does the decoder + shift do the work?

Recompute the world-model perturbation metrics under three adjacency conditions, holding the
decoder/encoder/shift-encoder fixed:
  real      A = learned adjacency
  shuffled  A with columns permuted (destroys learned regulatory structure, keeps magnitude/sparsity)
  zero      A = 0  (no SCM propagation → Δ = decode(u⊙S); pure decoder+shift, no causal graph)

Metrics on held-out singles (zero-shot) and combos, drift-removed:
  disc   = retrieval accuracy (is pred_i nearest its OWN obs_i by cosine among the set)
  spear  = mean per-item Spearman(|pred|, |obs|) on the item's DEGs (effect-size recovery)
  pear   = mean per-item Pearson(pred, obs) on DEGs
real − shuffled and real − zero quantify how much the DAG actually contributes.
Run via scripts/dag_ablation.sbatch.
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import add_stack_args, build_stack
from scipy.stats import spearmanr
CHUNK = 256


def metrics(P, O, drift, thr):
    """P,O [K,G] predicted/observed Δ. Returns disc, spearman, pearson on resid DEGs."""
    Pr, Or = P - drift, O - drift
    Prn = Pr / (np.linalg.norm(Pr, axis=1, keepdims=True) + 1e-9)
    Orn = Or / (np.linalg.norm(Or, axis=1, keepdims=True) + 1e-9)
    disc = float(np.mean((Prn @ Orn.T).argmax(1) == np.arange(len(P))))
    sp, pe = [], []
    for i in range(len(P)):
        sel = np.abs(Or[i]) > thr
        if sel.sum() < 3: continue
        p, o = Pr[i, sel], Or[i, sel]
        if p.std() < 1e-8 or o.std() < 1e-8: continue
        sp.append(spearmanr(np.abs(p), np.abs(o)).correlation)
        pe.append(np.corrcoef(p, o)[0, 1])
    return disc, float(np.nanmean(sp)), float(np.nanmean(pe))


def predict(op, data, keys, S_list, A, ctrl_recon):
    """Δ predictions [K,G] for gene-index sets under adjacency A."""
    z = op.apply_many(data.eps_ctrl_mean, torch.stack(S_list), A)          # [K,d]
    out = []
    for c in z.split(CHUNK):
        out.append((op.decode(c) - ctrl_recon).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--n_perm', type=int, default=5); ap.add_argument('--device', default=None)
    a = ap.parse_args(); dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, meta, op, data = build_stack(a, dev, n_dag=8); data.set_pool('curated')
    A = op.mean_A(); d = A.shape[0]; ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]
    drift = np.mean([data.de_single[g]['lfc'] for g in data.single_split['train']], 0).astype(np.float64)
    thr = data.thresh; rng = np.random.default_rng(0)

    # held-out singles + combos
    sg = [g for g in data.single_split['test'] if g in data.de_single]
    S_s = [op.shift_for([data.gene_to_idx[g]]).squeeze(0) for g in sg]
    O_s = np.stack([data.de_single[g]['lfc'] for g in sg]).astype(np.float64)
    ck = list(data.combo_split['test']) + list(data.combo_split['train'])      # all 40 pairs
    S_c = [op.shift_for([data.gene_to_idx[x] for x in k]).sum(0) for k in ck]
    O_c = np.stack([data.de_combo[k]['lfc'] for k in ck]).astype(np.float64)

    def run(A_variant):
        Ps = predict(op, data, sg, S_s, A_variant, ctrl_recon).astype(np.float64)
        Pc = predict(op, data, ck, S_c, A_variant, ctrl_recon).astype(np.float64)
        return metrics(Ps, O_s, drift, thr), metrics(Pc, O_c, drift, thr)

    rows = []
    (sd, ss, spe), (cd, cs, cpe) = run(A)
    rows.append(['real', sd, ss, spe, cd, cs, cpe])
    Z = torch.zeros_like(A)
    (sd, ss, spe), (cd, cs, cpe) = run(Z)
    rows.append(['zero (A=0)', sd, ss, spe, cd, cs, cpe])
    sh = []
    for _ in range(a.n_perm):
        perm = torch.as_tensor(rng.permutation(d), device=dev)
        sh.append(run(A[:, perm]))
    ss_s = np.array([[*m[0], *m[1]] for m in sh])
    rows.append(['shuffled-A (mean)'] + list(ss_s.mean(0)))
    rows.append(['shuffled-A (std)'] + list(ss_s.std(0)))

    df = pd.DataFrame(rows, columns=['A', 'singles_disc', 'singles_spear', 'singles_pear',
                                     'combos_disc', 'combos_spear', 'combos_pear'])
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    print(f"\n{'='*100}\nDAG LOAD-BEARING ABLATION — WM={os.path.basename(os.path.dirname(os.path.dirname(meta['checkpoint'])))}")
    print(f"held-out singles n={len(sg)} | combos n={len(ck)} | latent d={d}\n")
    print(df.to_string(index=False))
    real = df.iloc[0]; zero = df.iloc[1]; shuf = df.iloc[2]
    print("\nreal − shuffled:", {c: round(float(real[c] - shuf[c]), 3) for c in df.columns[1:]})
    print("real − zero    :", {c: round(float(real[c] - zero[c]), 3) for c in df.columns[1:]})
    print("\nIf real ≈ shuffled ≈ zero → the DAG is NOT load-bearing for prediction (decoder+shift do the work).")
    df.to_csv(f'{HERE}/dag_ablation.csv', index=False); print(f"saved -> dag_ablation.csv")


if __name__ == '__main__':
    main()
