"""Was the latent DAG degenerate / decoder-memorized in the EARLIER Replogle/Norman experiments,
or is it a Joung/exp19 regression? Model-agnostic: read decoder.D + dag.{theta,weights,mask} straight
from each checkpoint's state_dict (no per-experiment imports), reconstruct A, and run the same
diagnostics as the Joung GRN figures:
  degeneracy : gate-frac-ON, |edge weights|, soft-A density(>1e-3), spectral radius, |A|
  causality  : CollecTRI AUROC of G=D·Aprob·Dᵀ  vs  shuffled-A  vs  random-D  vs  corr-floor
Aprob = sigmoid(θ/τ)·mask (the gate-prob matrix the CollecTRI prior shapes, = get_edge_probs).
Run via scripts/grn_legacy_sweep.sbatch (CPU).
"""
import os, torch, numpy as np, numpy.linalg as la, pandas as pd
from sklearn.metrics import roc_auc_score
import decoupler as dc
R = '/home/sgoudarzi/scCausalRL/experiments'
# (tag, dataset, checkpoint) — spread of Replogle/Norman across sparsity settings
WMS = [
    ('exp2_shiftlin',   'replogle', f'{R}/exp2_shift_linear/results/phase1/checkpoints/phase1_epoch150.pt'),
    ('exp4_repl_energy', 'replogle', f'{R}/exp4_shift_linear_energy/results/phase1/checkpoints/phase1_epoch150.pt'),
    ('exp6_repl_mmdmse', 'replogle', f'{R}/exp6_shift_linear_mmd_mse/results/phase1/checkpoints/phase1_epoch150.pt'),
    ('exp8_norman_ls10', 'norman',  f'{R}/exp8_dag_revival_grid/results/grid/exp8_sl0p5_ls10p0/checkpoints/phase1_epoch150.pt'),
    ('exp9_norman',      'norman',  f'{R}/exp9_prior_generalization/results/grid/exp9_c2_heldtf/checkpoints/phase1_epoch150.pt'),
    ('exp10_norman_ls30','norman',  f'{R}/exp10_sparsity_disentangle/results/grid/exp10_g_ls30/checkpoints/phase1_epoch150.pt'),
    ('exp11_norman',     'norman',  f'{R}/exp11_norman_shift/results/phase1/checkpoints/phase1_epoch150.pt'),
    ('exp12_norman_mut', 'norman',  f'{R}/exp12_shift_mutilate/results/grid/exp12_repl_loadclamp/checkpoints/phase1_epoch150.pt'),
    ('exp14_repl_combo', 'replogle', f'{R}/exp14_norman_combotrain/results/grid/exp14_decorr300/checkpoints/phase1_epoch150.pt'),
    ('exp16_repl_multi', 'replogle', f'{R}/exp16_multicellline/results/grid/exp16_multicellline/checkpoints/phase1_epoch80.pt'),
]


def main():
    ct_net = dc.op.collectri(organism='human')
    rng = np.random.default_rng(0); rows = []
    for tag, ds, ck in WMS:
        if not os.path.exists(ck):
            print(f"[skip] {tag}: no ckpt"); continue
        c = torch.load(ck, map_location='cpu', weights_only=False); sd = c.get('model_state_dict', c)
        if 'decoder.D' not in sd or 'dag.theta' not in sd:
            print(f"[skip] {tag}: not LDVAE state_dict"); continue
        p1 = (c.get('config', {}) or {}).get('phase1', {}) or {}
        gn = [str(g).upper() for g in (c.get('gene_names') or [])]
        D = sd['decoder.D'].numpy().astype(np.float64); th = sd['dag.theta'].numpy().astype(np.float64)
        w = sd['dag.weights'].numpy().astype(np.float64)
        mask = sd['dag.mask'].numpy() if 'dag.mask' in sd else (1 - np.eye(th.shape[0]))
        tau = float(sd['dag.tau']) if 'dag.tau' in sd else 1.0
        d = th.shape[0]; off = ~np.eye(d, dtype=bool)
        gprob = (1 / (1 + np.exp(-th / tau))) * mask                  # gate probs (= get_edge_probs)
        Aw = gprob * w * mask                                         # soft weighted A
        # degeneracy stats
        gate_on = float(np.mean(gprob[off] > 0.9)); wmean = float(np.abs(w[off]).mean())
        dens = float(np.mean(np.abs(Aw[off]) > 1e-3)); Amean = float(np.abs(Aw[off]).mean())
        spec = float(np.abs(la.eigvals(Aw)).max())
        # CollecTRI causality (TF rows)
        au = sh = rd = None
        if gn:
            up2i = {g: i for i, g in enumerate(gn)}
            ct = {(up2i[s], up2i[t]) for s, t in zip(ct_net['source'].astype(str).str.upper(), ct_net['target'].astype(str).str.upper())
                  if s in up2i and t in up2i and up2i[s] != up2i[t]}
            if len(ct) > 50:
                tf_rows = sorted({a for a, _ in ct}); rp = {r: k for k, r in enumerate(tf_rows)}
                n = len(gn); Dtf = D[tf_rows]
                y = np.zeros((len(tf_rows), n), np.int8)
                for a, b in ct: y[rp[a], b] = 1
                pos = np.where(y.ravel())[0]; neg = rng.choice(np.where(y.ravel() == 0)[0], min(300000, (y == 0).sum()), replace=False)
                ev = np.concatenate([pos, neg]); yy = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
                def A(score): return float(roc_auc_score(yy, score.ravel()[ev]))
                au = A(np.abs(Dtf @ gprob @ D.T))
                sh = float(np.mean([A(np.abs(Dtf @ gprob[:, rng.permutation(d)] @ D.T)) for _ in range(3)]))
                Q, _ = la.qr(rng.standard_normal((n, d))); rd = A(np.abs(Q[tf_rows] @ gprob @ D.T))
        rows.append(dict(WM=tag, dataset=ds, lam_sparse=p1.get('lambda_sparse'), mode=p1.get('lambda_sparse_mode'),
                         lam_prior=p1.get('lambda_prior'), gate_on=round(gate_on, 3), absW=round(wmean, 3),
                         softA_dens=round(dens, 3), softA_spec=round(spec, 1), absA=round(Amean, 4),
                         collectri_auroc=round(au, 3) if au else None, shuffA=round(sh, 3) if sh else None,
                         randomD=round(rd, 3) if rd else None))
        print(f"  [{tag:18s}] {ds:8s} ls={p1.get('lambda_sparse')} gate_on={gate_on:.2f} |w|={wmean:.3f} "
              f"dens={dens:.2f} spec={spec:.1f} | AUROC={au and round(au,3)} shufA={sh and round(sh,3)} randD={rd and round(rd,3)}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f'{os.path.dirname(os.path.abspath(__file__))}/grn_legacy_sweep.csv', index=False)
    print(f"\n{'='*110}\nLEGACY GRN DIAGNOSTIC (Replogle/Norman; degenerate if gate_on high + |w| collapsed; causal if AUROC>>shufA & >>randD)")
    pd.set_option('display.width', 200); print(df.to_string(index=False))
    print("saved -> grn_legacy_sweep.csv")


if __name__ == '__main__':
    main()
