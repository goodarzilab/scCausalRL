"""GRN-causality + DAG-load-bearing sweep across ALL singles-only exp19 world models.

Per WM: (1) GRN G=D·A·Dᵀ vs CollecTRI AUROC + shuffled-A control + random-D control + density +
spectral radius (raw edge-probs that the prior shapes, and the spectrally-clamped solve-time A) +
DoRothEA held-out AUROC (edges the prior never saw); (2) DAG load-bearing: real vs col-shuffled vs
A=0 on held-out singles + combos (disc / effect-spearman / pearson, drift-removed). Decoupler nets +
control-correlation GRN computed ONCE (WM-independent). Run via scripts/grn_dag_sweep.sbatch.
"""
import os, sys, argparse, gc
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import build_stack
EXP19 = '/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose'
SINGLES = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
WMS = [  # singles-only exp19 models (load_combos=False)
    'v2_esm_con_ugate', 'v2_esm_con_epsmix', 'v2_esm_con_ugate_epsmix', 'v2_esm_con_ident',
    'v2_combo_gate512', 'v2_combo_resid', 'v2_combo_resid_hicon_g512']
GRN_WMS = ['v2_esm_con_ugate'] + ['grn_' + t for t in (  # baseline + the 14 GRN-fix ablations
    'spA1', 'spA5', 'spA10', 'spA30', 'shiftl1', 'shiftl1hi', 'resid_spA', 'lowD_spA',
    'ortho_spA', 'prior0_spA', 'fix', 'fix_epsmix', 'ugatepure', 'ugatepure_fix')]
GRNFIX_WMS = ['grnfix_' + t for t in (   # the 12 completed grn-fix runs (gates-mode / dagutil / shift-l1)
    'gatesp5', 'gatesp10', 'gatesp20', 'gates10_noprior', 'gates10_shiftl1',
    'dagutil', 'dagutil_hi', 'dagutil_hiD', 'dagutil_noprior',
    'dagutil_shiftl1', 'dagutil_shiftl1_noprior', 'noDAG')]
SENA_WMS = ['grnfix_' + t for t in (     # the 9 SENA + sparsity-ablation runs (+ 2 baselines for contrast)
    'sena', 'sena_leak0', 'sena_shiftl1', 'sena_shiftl1_nodag', 'sena_shiftl1_hi',
    'sena_shiftl1_splo', 'sena_shiftl1_sp0', 'dagutil_splo', 'dagutil_sp0',
    'dagutil_noprior', 'noDAG')]
EPSD_WMS = ['grnfix_' + t for t in (     # the 5 ε-decorrelation arms (+ baselines)
    'epsdecorr', 'epsdecorr_ortho', 'epsdecorr_ortho_hi', 'sena_epsdecorr', 'sena_epsdecorr_ortho',
    'dagutil_noprior', 'noDAG')]
EXP20 = '/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing'
X20_WMS = ['struct_min_esm', 'struct_sparse_esm', 'struct_sparse_con', 'struct_sparse_resid',
           'struct_sparse_pertonly', 'struct_sparse_residrecon', 'struct_sparse_all']   # exp20 structural fix
X20PO_WMS = [t + '_po' for t in X20_WMS]                                                 # prior-OFF reruns
X20ARCH_WMS = ['struct_frozenD', 'struct_whiten', 'struct_frozenD_whiten',               # Phase-2 architecture arms
               'staged_esm_s26', 'staged_esm_s36', 'staged_con_s26',
               'staged_frozenD_s26', 'staged_nonlinSCM_s26']
CHUNK = 256


class _A:
    dataset = 'joung'; wm_root = EXP19; combo_path = None; singles_path = None; curated_topk = 256; n_ctrl = 500


def auroc(score_tf, pairset, tf_rows, n):
    from sklearn.metrics import roc_auc_score
    rp = {r: k for k, r in enumerate(tf_rows)}; y = np.zeros(score_tf.shape, np.int8); npos = 0
    for ti, gi in pairset:
        if ti in rp and gi < n: y[rp[ti], gi] = 1; npos += 1
    if npos == 0: return None
    s = np.nan_to_num(score_tf.ravel(), nan=0.0, posinf=0.0, neginf=0.0)   # robust to NaN in D·A·Dᵀ
    return float(roc_auc_score(y.ravel(), s))


def wm_metrics(P, O, drift, thr):
    Pr, Or = P - drift, O - drift
    Prn = Pr / (np.linalg.norm(Pr, 1, keepdims=True) + 1e-9); Orn = Or / (np.linalg.norm(Or, 1, keepdims=True) + 1e-9)
    disc = float(np.mean((Prn @ Orn.T).argmax(1) == np.arange(len(P))))
    sp, pe = [], []
    for i in range(len(P)):
        sel = np.abs(Or[i]) > thr
        if sel.sum() < 3 or Pr[i, sel].std() < 1e-8 or Or[i, sel].std() < 1e-8: continue
        sp.append(spearmanr(np.abs(Pr[i, sel]), np.abs(Or[i, sel])).correlation); pe.append(np.corrcoef(Pr[i, sel], Or[i, sel])[0, 1])
    return disc, float(np.nanmean(sp)), float(np.nanmean(pe))


def predict(op, data, S_list, A, ctrl_recon):
    z = op.apply_many(data.eps_ctrl_mean, torch.stack(S_list), A); out = []
    for c in z.split(CHUNK): out.append((op.decode(c) - ctrl_recon).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n_perm', type=int, default=3); ap.add_argument('--device', default=None)
    ap.add_argument('--grid', default='base', choices=['base', 'grn', 'grnfix', 'sena', 'epsd', 'x20', 'x20po', 'x20arch'],
                    help="grnfix=12 grn-fix; sena=9 SENA/sparsity; epsd=5 ε-decorr; x20=exp20 structural (+baselines)")
    a = ap.parse_args(); dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu')); torch.set_grad_enabled(False)
    import decoupler as dc
    rng = np.random.default_rng(0)
    wm_list = {'grn': GRN_WMS, 'grnfix': GRNFIX_WMS, 'sena': SENA_WMS, 'epsd': EPSD_WMS, 'x20': X20_WMS, 'x20po': X20PO_WMS, 'x20arch': X20ARCH_WMS}.get(a.grid, WMS)
    ROOT = EXP20 if a.grid in ('x20', 'x20po', 'x20arch') else EXP19
    NETS = {}; CORR = {}; rows = []
    for tag in wm_list:
        ckpt = f'{ROOT}/results/grid/{tag}/checkpoints/phase1_epoch60.pt'
        if not os.path.exists(ckpt): print(f"[skip] {tag}"); continue
        args = _A(); args.checkpoint = ckpt; args.device = dev; args.wm_root = ROOT
        model, meta, op, data = build_stack(args, dev, 8); data.set_pool('curated')
        gn = [str(g) for g in data.gene_names]; n = len(gn); up2i = {g.upper(): i for i, g in enumerate(gn)}
        # --- one-time: nets + corr GRN ---
        if not NETS:
            def pairs(net):
                s = set()
                for x, y in zip(net['source'].astype(str).str.upper(), net['target'].astype(str).str.upper()):
                    if x in up2i and y in up2i and up2i[x] != up2i[y]: s.add((up2i[x], up2i[y]))
                return s
            ct = pairs(dc.op.collectri(organism='human')); dor = pairs(dc.op.dorothea(organism='human', levels=['A', 'B']))
            held = dor - ct; tf_rows = sorted({t for t, _ in ct} | {t for t, _ in held})
            NETS.update(ct=ct, held=held, tf_rows=tf_rows)
            import anndata as ad; sys.path.insert(0, '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas')
            from data.load_joung import _lognorm
            S = ad.read_h5ad(SINGLES); ci = np.where(S.obs['is_control'].values.astype(bool))[0]
            sel = rng.choice(ci, 3000, replace=False); Xc = np.asarray(_lognorm(S.X)[sel].todense(), np.float64)
            Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
            corr_tf = np.abs((Xc[:, tf_rows].T @ Xc) / Xc.shape[0])
            CORR['ct'] = auroc(corr_tf, ct, tf_rows, n); CORR['held'] = auroc(corr_tf, held, tf_rows, n)
            del S, Xc; gc.collect()
        ct, held, tf_rows = NETS['ct'], NETS['held'], NETS['tf_rows']
        # --- GRN stats ---
        D = model.decoder.D.detach().cpu().numpy().astype(np.float64); Araw = model.dag.get_edge_probs().detach().cpu().numpy().astype(np.float64)
        if not (np.isfinite(D).all() and np.isfinite(Araw).all()):
            print(f"  [SKIP-NaN] {tag}: NaN/inf in D({(~np.isfinite(D)).sum()}) or Araw({(~np.isfinite(Araw)).sum()}) "
                  f"— arm diverged in training", flush=True)
            del model, op, data; gc.collect()
            continue
        d = Araw.shape[0]; Dtf = D[tf_rows]
        Gtf = np.abs(Dtf @ Araw @ D.T); au = auroc(Gtf, ct, tf_rows, n); auh = auroc(Gtf, held, tf_rows, n)
        sh = [auroc(np.abs(Dtf @ Araw[:, rng.permutation(d)] @ D.T), ct, tf_rows, n) for _ in range(a.n_perm)]
        Q, _ = np.linalg.qr(rng.standard_normal((n, d))); aurd = auroc(np.abs(Q[tf_rows] @ Araw @ Q.T), ct, tf_rows, n)
        Asolve = op.mean_A().detach().cpu().numpy()
        dens_raw = float((np.abs(Araw) > 1e-3).mean()); spec_raw = float(np.abs(np.linalg.eigvals(Araw)).max())
        spec_solve = float(np.abs(np.linalg.eigvals(Asolve)).max()); dens_solve = float((np.abs(Asolve) > 1e-3).mean())
        # --- DAG ablation ---
        A_t = op.mean_A(); ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]
        drift = np.mean([data.de_single[g]['lfc'] for g in data.single_split['train']], 0).astype(np.float64)
        sg = [g for g in data.single_split['test'] if g in data.de_single]
        Ss = [op.shift_for([data.gene_to_idx[g]]).squeeze(0) for g in sg]; Os = np.stack([data.de_single[g]['lfc'] for g in sg]).astype(np.float64)
        ck = list(data.combo_split['test']) + list(data.combo_split['train'])
        Sc = [op.shift_for([data.gene_to_idx[x] for x in k]).sum(0) for k in ck]; Oc = np.stack([data.de_combo[k]['lfc'] for k in ck]).astype(np.float64)
        def ab(Av):
            return wm_metrics(predict(op, data, Ss, Av, ctrl_recon), Os, drift, data.thresh), \
                   wm_metrics(predict(op, data, Sc, Av, ctrl_recon), Oc, drift, data.thresh)
        (rsd, rss, rsp), (rcd, rcs, rcp) = ab(A_t)
        (zsd, zss, zsp), (zcd, zcs, zcp) = ab(torch.zeros_like(A_t))
        shf = [ab(A_t[:, torch.as_tensor(rng.permutation(d), device=dev)]) for _ in range(a.n_perm)]
        shS = np.array([[*m[0]] for m in shf]).mean(0); shC = np.array([[*m[1]] for m in shf]).mean(0)
        rows.append(dict(WM=tag, collectri_auroc=au, shuffA=float(np.mean(sh)), randomD=aurd, corrGRN=CORR['ct'],
                         dorothea_held=auh, dor_corr=CORR['held'], dens_raw=dens_raw, spec_raw=spec_raw,
                         dens_solve=dens_solve, spec_solve=spec_solve,
                         # DAG load-bearing: real vs zero vs col-shuffled A on ALL prediction metrics
                         s_disc_real=rsd, s_disc_zero=zsd, s_disc_shuf=shS[0],
                         s_spear_real=rss, s_spear_zero=zss, s_spear_shuf=shS[1],
                         s_pear_real=rsp, s_pear_zero=zsp, s_pear_shuf=shS[2],
                         c_disc_real=rcd, c_disc_zero=zcd, c_disc_shuf=shC[0],
                         c_spear_real=rcs, c_spear_zero=zcs, c_spear_shuf=shC[1],
                         c_pear_real=rcp, c_pear_zero=zcp, c_pear_shuf=shC[2]))
        print(f"  [{tag}] AUROC={au:.3f} shufA={np.mean(sh):.3f} | dens_raw={dens_raw:.2f} spec_solve={spec_solve:.2f} "
              f"| s_pear r/0/shf={rsp:.2f}/{zsp:.2f}/{shS[2]:.2f} "
              f"s_spear r/0/shf={rss:.2f}/{zss:.2f}/{shS[1]:.2f} "
              f"c_pear r/0/shf={rcp:.2f}/{zcp:.2f}/{shC[2]:.2f}", flush=True)
        del model, op, data; gc.collect(); torch.cuda.empty_cache() if dev.type == 'cuda' else None

    out_csv = f'{HERE}/grn_dag_sweep{"" if a.grid in ("base","grn") else "_"+a.grid}.csv'
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    pd.set_option('display.width', 240); pd.set_option('display.max_columns', 40)
    print("\n================ GRN + DAG SWEEP (singles-only exp19 WMs) ================")
    print(df.round(3).to_string(index=False))
    print(f"\ncorr-GRN baseline AUROC: CollecTRI={CORR['ct']:.3f} DoRothEA-held={CORR['held']:.3f}")
    print("Causal if: AUROC >> shuffA AND >> randD; DAG load-bearing if real >> {zero, shuf} on disc/spear/pear.")
    print(f"saved -> {out_csv}")


if __name__ == '__main__':
    main()
