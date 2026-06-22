"""Singles -> combos/triples generalization eval for the Joung TF-atlas WM (the core
exp17 question: the model trains on SINGLE TF overexpressions; can the do-operator compose
single-TF shifts into correct COMBO / TRIPLE predictions, zero-shot?).

For each held-out multi-TF set (real TFs only; GFP/mCherry fillers dropped):
  observed  Δ_obs = mean(combo cells) − mean(hESC control)
  do-op     Δ_do  = decode( mutilation-solve( Σ shift(TF_i) ) ) − ctrl_recon   (set-dependent u)
  additive  Δ_add = Σ_i Δ_single_pred(TF_i)                                      (no epistasis)
Both predictors are zero-shot on the SET (model never saw combos). Scored with the same
families as the singles eval + a do-vs-additive comparison.

Metrics (saved results/combo_eval/<tag>/, split by pairs vs triples):
  gene-space : pearson_delta, de_direction, discrimination_cos, retrieval_top1
  identity   : fate_top1, fate_betacos, fate_entropy_r  (Cao-mouse NNLS, nuisance fit on
               TRAINING singles)
  epistasis  : do-op vs additive pearson_delta + discrimination (does coherent composition
               beat additive?)

Run (high-mem node):  CKPT=... TAG=epoch80 sbatch scripts/combo_eval.sbatch
"""
import os, sys, json, argparse, re
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from evaluation.heldout_zeroshot_eval import gene_space, fate_space, zscore_rows, _solve

COMBO = os.path.join(ROOT, 'data', 'joung_combo_proc.h5ad')
FILLERS = {'GFP', 'mCherry', 'mcherry', 'EGFP'}


def parse_tfs(tf_set):
    """'TFORF0098-MSGN1,TFORF3549-GFP' -> ['MSGN1'] (real TFs, fillers dropped)."""
    out = []
    for tok in str(tf_set).split(','):
        sym = tok.split('-')[-1].strip()
        if sym and sym not in FILLERS:
            out.append(sym)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--min_cells', type=int, default=20)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    out = os.path.join(ROOT, 'results', 'combo_eval', args.tag); os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from evaluation.run_shift_eval_norman import load_shift_model
    from data.load_joung import load_joung
    import anndata as ad, scipy.sparse as sp, pandas as pd
    torch.set_grad_enabled(False)                      # eval only — no autograd
    model, gene_names, ck = load_shift_model(args.checkpoint, dev)
    gidx = {g: i for i, g in enumerate(gene_names)}; d = model.latent_dim; G = len(gene_names)
    p1 = ck.get('config', {}).get('phase1', {})
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); U_MAX = float(p1.get('pert_mutilate_u_max', 0.5))
    U_MODE = p1.get('pert_mutilate_u_mode', 'shift')

    # control basal from the singles atlas (same hESC control), + training-single Δ for nuisance fit
    dd = load_joung(n_hvg=None)
    Xc = dd['adata'].X; Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
    de_single = dd['de_single']
    train_obs = np.stack([np.asarray(de_single[g], np.float32) for g in dd['pert_split']['train'] if g in de_single])
    rng = np.random.default_rng(0); ctrl = np.array(dd['control_idx'])
    sel = ctrl[rng.choice(len(ctrl), min(args.n_ctrl_cells, len(ctrl)), replace=False)]
    x_ctrl = torch.tensor(Xc[sel], dtype=torch.float32, device=dev)
    with torch.no_grad():
        _, _, eps_ctrl = model.encoder(x_ctrl)
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        LHS = torch.eye(d, device=dev, dtype=A.dtype) - A.T
        z_prev = torch.clamp(_solve(LHS, eps_ctrl), -20, 20)
        pert_dec = getattr(model, "pert_decoder", None) or model.decoder  # sep-head aware (no-op otherwise)
        ctrl_recon = pert_dec(z_prev).mean(0).cpu().numpy()

    def single_shift(gi):
        return model.compute_shift(torch.tensor([gi], device=dev)).sum(0)

    def do_op(gis):                       # coherent mutilation composition of a TF set
        S = torch.stack([single_shift(g) for g in gis]).sum(0)
        z_S = torch.clamp(_solve(LHS, S.unsqueeze(0)), -20, 20)
        z_target = z_prev + z_S
        u = (BETA * S.abs()).clamp(max=1.0) if U_MODE == 'shift' else \
            model.get_loadings()[torch.tensor(gis, device=dev)].sum(0).clamp(0, U_MAX)
        u = model.refine_u(u, S)                                    # nonlinear gate (no-op unless u_gate_mlp)
        eps_tilde = model.mix_eps(eps_ctrl, z_target, u.unsqueeze(0))
        z_post, _, _ = model.dag.scm_intervene_u(eps_ctrl, A, eps_tilde, u)
        return pert_dec(z_post).mean(0).cpu().numpy() - ctrl_recon

    def additive(gis):                    # Σ single-TF predicted Δ (no epistasis)
        tot = np.zeros(G, np.float32)
        for g in gis:
            z_S = torch.clamp(_solve(LHS, single_shift(g).unsqueeze(0)), -20, 20)
            tot += (pert_dec(z_prev + z_S).mean(0).cpu().numpy() - ctrl_recon)
        return tot

    # observed combo Δ — lognorm exactly as load_joung does for the singles (same frame)
    from data.load_joung import _lognorm
    A2 = ad.read_h5ad(COMBO); A2.X = _lognorm(A2.X)
    Xk = A2.X.toarray() if sp.issparse(A2.X) else np.asarray(A2.X)
    tf_set = A2.obs['TF_set'].astype(str).values
    ctrl_mean_combo = Xc[ctrl].mean(0)    # use the singles-atlas control baseline (consistent frame)

    rows = []
    for setname in sorted(set(tf_set)):
        tfs = parse_tfs(setname)
        gis = [gidx[t] for t in tfs if t in gidx]
        if len(gis) < 2:                  # need >=2 real on-axis TFs (true combo)
            continue
        m = tf_set == setname
        if m.sum() < args.min_cells:
            continue
        obs = Xk[m].mean(0) - ctrl_mean_combo
        rows.append({'set': '+'.join(tfs), 'k': len(gis), 'gis': gis, 'obs': obs,
                     'do': do_op(gis), 'add': additive(gis)})
    print(f"[combo-eval] usable multi-TF sets: {len(rows)} "
          f"(pairs={sum(r['k']==2 for r in rows)}, triples={sum(r['k']>=3 for r in rows)})", flush=True)

    res = {}
    for kind, sel_rows in [('pairs', [r for r in rows if r['k'] == 2]),
                           ('triples', [r for r in rows if r['k'] >= 3]),
                           ('all', rows)]:
        if len(sel_rows) < 2:
            continue
        Obs = np.stack([r['obs'] for r in sel_rows]); Do = np.stack([r['do'] for r in sel_rows])
        Add = np.stack([r['add'] for r in sel_rows])
        gdo, *_ = gene_space(Do, Obs); gadd, *_ = gene_space(Add, Obs)
        # carry ALL gene-space metrics (incl. effect_spearman, deg_auroc) under do_/add_ prefixes
        entry = {'n': len(sel_rows)}
        entry.update({f'do_{k}': v for k, v in gdo.items() if k != 'n'})
        entry.update({f'add_{k}': v for k, v in gadd.items() if k != 'n'})
        if len(sel_rows) >= 3:
            try:
                fdo, *_ = fate_space(Do, Obs, train_obs, gene_names)
                entry.update({'do_fate_top1': fdo['fate_top1'], 'do_fate_betacos': fdo['fate_betacos'],
                              'do_fate_entropy_r': fdo['fate_entropy_r']})
            except Exception as e:
                entry['fate_error'] = str(e)
        res[kind] = entry
        print(f"  [{kind}] n={entry['n']} do: pearson={entry['do_pearson_delta']:.3f} "
              f"discrim={entry['do_discrimination_cos']:.3f} | add: pearson={entry['add_pearson_delta']:.3f} "
              f"discrim={entry['add_discrimination_cos']:.3f}" +
              (f" | fate_top1={entry.get('do_fate_top1', float('nan')):.3f}" if 'do_fate_top1' in entry else ""), flush=True)

    json.dump({'tag': args.tag, 'epoch': ck.get('epoch'), **res}, open(f"{out}/summary.json", 'w'), indent=2)
    pd.DataFrame([{'set': r['set'], 'k': r['k']} for r in rows]).to_csv(f"{out}/sets.csv", index=False)
    print(f"\nsaved → {out}/")


if __name__ == '__main__':
    main()
