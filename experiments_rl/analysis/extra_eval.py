"""Two extra exp20 diagnostics per arm:

(1) RAW A_gap — A-load-bearing on the NON-drift-removed prediction (spearman/pearson of |Δ| vs
    observed |Δ|, real vs zero vs shuf A). The main sweep drift-removes; this asks whether A helps
    predict the raw effect (which is dominated by the shared program).

(2) Fate-hierarchy GRN validation — does the projected GRN G=|D·M·Dᵀ| (M=A linear, or the nonlinear
    Jacobian (I−diag(tanh')Aᵀ)⁻¹) connect TFs that drive the SAME developmental fate? Ground truth =
    the human-fetal-atlas TF annotation (main_fate). AUROC(same-fate TF pair | G score), with shuffle-A
    and random-D controls. This is a prior-FREE biological test (independent of CollecTRI).

Run: python analysis/extra_eval.py --grid x20po|x20arch --device cpu
"""
import os, sys, argparse, gc
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import build_stack
from analysis.grn_dag_sweep import (predict, wm_metrics, X20_WMS, X20PO_WMS, X20ARCH_WMS, _A, EXP20)
from analysis.component_ablation import build_eval
from sklearn.metrics import roc_auc_score

FATE_CSV = ('/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/'
            'analysis/mouse_dev_annotation/joung_tf_developmental_annotation.csv')


def fate_auroc(G_tf, tf_idx, fates):
    """G_tf: [T,T] |GRN| among annotated TFs; fates: [T] main_fate labels. AUROC same-fate vs diff."""
    T = len(tf_idx)
    same, score = [], []
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            same.append(1 if fates[i] == fates[j] else 0)
            score.append(G_tf[i, j])
    same = np.array(same); score = np.nan_to_num(np.array(score))
    if same.sum() == 0 or same.sum() == len(same):
        return None
    return float(roc_auc_score(same, score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='x20po', choices=['x20', 'x20po', 'x20arch'])
    ap.add_argument('--device', default=None); ap.add_argument('--n_perm', type=int, default=3)
    a = ap.parse_args()
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu')); torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)
    wm_list = {'x20': X20_WMS, 'x20po': X20PO_WMS, 'x20arch': X20ARCH_WMS}[a.grid]

    fate_df = pd.read_csv(FATE_CSV)
    tf2fate = {str(t).upper(): str(f) for t, f in zip(fate_df['TF'], fate_df['main_fate']) if str(f) != 'nan'}

    rows = []
    for tag in wm_list:
        ckpt = f'{EXP20}/results/grid/{tag}/checkpoints/phase1_epoch60.pt'
        if not os.path.exists(ckpt):
            print(f"[skip] {tag}"); continue
        args = _A(); args.checkpoint = ckpt; args.device = dev; args.wm_root = EXP20
        model, meta, op, data = build_stack(args, dev, 8); data.set_pool('curated')
        A, ctrl, drift, Ss, Os, Sc, Oc = build_eval(op, data)
        d = A.shape[0]; zero_drift = np.zeros_like(drift)
        A0 = torch.zeros_like(A); Ashuf = A[:, torch.as_tensor(rng.permutation(d), device=dev)]

        # (1) RAW A_gap: wm_metrics with drift=0
        def raw(Av, S, O):
            return wm_metrics(predict(op, data, S, Av, ctrl), O, zero_drift, data.thresh)
        (rsd, rss, rsp) = raw(A, Ss, Os); (zsd, zss, zsp) = raw(A0, Ss, Os); (hsd, hss, hsp) = raw(Ashuf, Ss, Os)
        row = dict(WM=tag, s_spear_raw_real=rss, s_spear_raw_zero=zss, s_spear_raw_shuf=hss,
                   s_pear_raw_real=rsp, s_pear_raw_zero=zsp,
                   s_Agap_raw=rss - max(zss, hss))
        if Oc is not None and len(Sc):
            (_, rcs, _) = raw(A, Sc, Oc); (_, zcs, _) = raw(A0, Sc, Oc); (_, hcs, _) = raw(Ashuf, Sc, Oc)
            row.update(c_spear_raw_real=rcs, c_Agap_raw=rcs - max(zcs, hcs))

        # (2) Fate-hierarchy GRN: G = |D · M · Dᵀ| among annotated TFs
        D = model.decoder.D.detach().cpu().numpy().astype(np.float64)
        gn = [str(g).upper() for g in data.gene_names]
        tf_idx = [i for i, g in enumerate(gn) if g in tf2fate]
        fates = [tf2fate[gn[i]] for i in tf_idx]
        Araw = model.dag.get_edge_probs().detach().cpu().numpy().astype(np.float64)
        if np.isfinite(Araw).all() and np.isfinite(D).all():
            nonlin = getattr(model.dag, 'scm_nonlinear', False)
            if nonlin:                       # Jacobian operator at z*≈0 (tanh'(0)=1) → (I−Aᵀ)⁻¹
                M = np.linalg.inv(np.eye(d) - Araw.T)
            else:
                M = Araw
            Dtf = D[tf_idx]
            G = np.abs(Dtf @ M @ Dtf.T)                                   # [T,T]
            row['fate_auroc'] = fate_auroc(G, tf_idx, fates)
            sh = []
            for _ in range(a.n_perm):
                Msh = M[:, rng.permutation(d)]
                sh.append(fate_auroc(np.abs(Dtf @ Msh @ Dtf.T), tf_idx, fates))
            row['fate_shuffA'] = float(np.mean([s for s in sh if s is not None])) if any(sh) else None
            Q, _ = np.linalg.qr(rng.standard_normal((len(gn), d)))
            row['fate_randomD'] = fate_auroc(np.abs(Q[tf_idx] @ M @ Q[tf_idx].T), tf_idx, fates)
            row['n_tf_annot'] = len(tf_idx)
        else:
            print(f"  [SKIP-NaN fate] {tag}")
        rows.append(row)
        print(f"  [{tag}] s_Agap_raw={row.get('s_Agap_raw'):+.3f} (real {row.get('s_spear_raw_real'):.3f} "
              f"vs zero {row.get('s_spear_raw_zero'):.3f}/shuf {row.get('s_spear_raw_shuf'):.3f}) | "
              f"fate_auroc={row.get('fate_auroc')} shuffA={row.get('fate_shuffA')} randD={row.get('fate_randomD')}", flush=True)
        del model, op, data; gc.collect()

    df = pd.DataFrame(rows); out = f'{HERE}/extra_eval_{a.grid}.csv'; df.to_csv(out, index=False)
    pd.set_option('display.width', 220)
    print("\n===== RAW A_gap + FATE-HIERARCHY GRN (" + a.grid + ") =====")
    print(df.round(3).to_string(index=False))
    print(f"\nfate_auroc > shuffA AND > 0.5 ⇒ G encodes developmental hierarchy via A; "
          f"fate_auroc≈shuffA ⇒ decoder-coexpression only. saved -> {out}")


if __name__ == '__main__':
    main()
