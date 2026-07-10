"""Overfitting / generalization diagnostic: compute drift-removed disc_resid + pear_resid on the
TRAIN gene split vs the HELD-OUT (test) gene split, and compare shift-encoder magnitudes across the
two splits. A large train≫test gap ⇒ the model fits seen TFs but the ESM→shift map does not
generalize to unseen TFs (zero-shot failure); systematically smaller held-out ‖S‖ ⇒ the shift
encoder extrapolates to a shrunken/mean shift for unseen descriptors.

Run: python evaluation/train_vs_heldout.py --checkpoint <ckpt>
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.heldout_zeroshot_eval import gene_space, zscore_rows, _solve
from data.load_joung import load_joung


def disc_resid(pred, obs, drift):
    g, _, _, _ = gene_space(pred - drift, obs - drift)
    return g['discrimination_cos'], g['pearson_delta'], g['retrieval_top1'], g['deg_auroc']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='cpu'); ap.add_argument('--n_ctrl', type=int, default=400)
    ap.add_argument('--match_train_split', action='store_true',
                    help='Reproduce the checkpoint data split exactly (min_cells_per_pert / '
                         'train_frac / val_frac / split_seed) so train/test cover the SAME TFs '
                         'seen in training. Needed to score memorization on a tiny overfit set.')
    a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)
    model, gene_names, ck = load_shift_model(a.checkpoint, dev)
    d = model.latent_dim; p1 = ck.get('config', {}).get('phase1', {}) or {}
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); U_MODE = p1.get('pert_mutilate_u_mode', 'shift')
    structural = getattr(model, 'pert_intervene_mode', 'legacy') == 'structural'

    # follow the CHECKPOINT's data config so v2 arms are scored on their own 4019-gene axis + the
    # effective-TF split (else a v2 model would be evaluated against the old 9923 proc = axis mismatch)
    _dc = (ck.get('config') or {}).get('data', {}) or {}
    _kw = {'n_hvg': None}
    if _dc.get('joung_proc_path'): _kw['proc_path'] = _dc['joung_proc_path']
    if _dc.get('effective_tfs'): _kw['effective_tfs'] = _dc['effective_tfs']
    if a.match_train_split:
        # reproduce the EXACT train/test split the checkpoint was trained on (tiny overfit set)
        _kw['min_cells_per_pert'] = int(_dc.get('min_cells_per_pert', 5))
        _kw['train_frac'] = float(_dc.get('single_pert_train_frac', 0.8))
        _kw['val_frac'] = float(_dc.get('single_pert_val_frac', 0.1))
        _kw['split_seed'] = int(_dc.get('split_seed', 42))
    dd = load_joung(**_kw); de = dd['de_single']; gidx = {g: i for i, g in enumerate(gene_names)}
    Xc = dd['adata'].X; Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
    rng = np.random.default_rng(0); ctrl = np.array(dd['control_idx'])
    sel = rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)
    x_ctrl = torch.tensor(Xc[ctrl[sel]], dtype=torch.float32, device=dev)
    _, _, eps_ctrl = model.encoder(x_ctrl)
    A = model.dag.sample_A(n_samples=1).squeeze(0)
    z_prev = model.dag.scm_forward(eps_ctrl, A)
    pert_dec = getattr(model, 'pert_decoder', None) or model.decoder
    ctrl_recon = pert_dec(z_prev).mean(0).cpu().numpy()

    train = [g for g in dd['pert_split']['train'] if g in de and g in gidx]
    test = [g for g in dd['pert_split']['test'] if g in de and g in gidx]
    drift = np.stack([np.asarray(de[g], np.float32) for g in train]).mean(0)   # leak-free (train)

    def predict_split(genes):
        pred = np.zeros((len(genes), len(gene_names)), np.float32); snorm = []
        for k, g in enumerate(genes):
            gi = gidx[g]
            S = model.compute_shift(torch.tensor([gi], device=dev)).sum(0)
            snorm.append(float(S.norm()))
            u = (BETA * S.abs()).clamp(max=1.0)
            u = model.refine_u(u, S)
            eps_tilde = eps_ctrl + S.unsqueeze(0) if structural else model.mix_eps(
                eps_ctrl, z_prev + _solve(torch.eye(d, device=dev) - A.T, S.unsqueeze(0)), u.unsqueeze(0))
            z_post, _, _ = model.dag.scm_intervene_u(eps_ctrl, A, eps_tilde, u)
            pred[k] = pert_dec(z_post).mean(0).cpu().numpy() - ctrl_recon
        return pred, np.array(snorm)

    def pear_noKO(pred, obs, genes):
        # per-TF Pearson of the drift-removed Δ (pred vs obs) EXCLUDING the perturbed gene itself
        # (the OE/KO gene dominates its own Δ; this measures the OFF-TARGET response). Mean over TFs.
        pr, ob = pred - drift, obs - drift; vals = []
        for k, g in enumerate(genes):
            m = np.ones(pr.shape[1], bool)
            if g in gidx:
                m[gidx[g]] = False                      # drop the perturbed gene
            a2, b2 = pr[k, m], ob[k, m]
            if a2.std() > 1e-9 and b2.std() > 1e-9:
                vals.append(float(np.corrcoef(a2, b2)[0, 1]))
        return float(np.mean(vals)) if vals else float('nan')

    print(f"=== TRAIN vs HELD-OUT | {a.tag or os.path.basename(os.path.dirname(os.path.dirname(a.checkpoint)))} ===")
    print(f"  {'split':6s} {'n':>4s} {'disc_resid':>11s} {'pear_resid':>11s} {'pear_noKO':>10s} {'retr':>6s} {'deg':>6s} {'|S|_mean':>9s} {'|S|_std':>8s}")
    stats = {}
    for name, genes in [('train', train), ('test', test)]:
        obs = np.stack([np.asarray(de[g], np.float32) for g in genes])
        pred, snorm = predict_split(genes)
        dc, pe, rt, dg = disc_resid(pred, obs, drift)
        pnk = pear_noKO(pred, obs, genes)
        stats[name] = dict(disc=dc, pear=pe, pnk=pnk, sn=snorm.mean())
        print(f"  {name:6s} {len(genes):4d} {dc:11.3f} {pe:11.3f} {pnk:10.3f} {rt:6.3f} {dg:6.3f} {snorm.mean():9.3f} {snorm.std():8.3f}")
    print(f"  --> disc GAP (train-test) = {stats['train']['disc']-stats['test']['disc']:+.3f} ; "
          f"pear GAP = {stats['train']['pear']-stats['test']['pear']:+.3f} ; "
          f"|S| ratio test/train = {stats['test']['sn']/(stats['train']['sn']+1e-9):.3f}")


if __name__ == '__main__':
    main()
