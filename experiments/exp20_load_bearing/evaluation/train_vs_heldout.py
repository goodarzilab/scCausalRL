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
    a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)
    model, gene_names, ck = load_shift_model(a.checkpoint, dev)
    d = model.latent_dim; p1 = ck.get('config', {}).get('phase1', {}) or {}
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); U_MODE = p1.get('pert_mutilate_u_mode', 'shift')
    structural = getattr(model, 'pert_intervene_mode', 'legacy') == 'structural'

    dd = load_joung(n_hvg=None); de = dd['de_single']; gidx = {g: i for i, g in enumerate(gene_names)}
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

    print(f"=== TRAIN vs HELD-OUT | {a.tag or os.path.basename(os.path.dirname(os.path.dirname(a.checkpoint)))} ===")
    print(f"  {'split':6s} {'n':>4s} {'disc_resid':>11s} {'pear_resid':>11s} {'retr':>6s} {'deg':>6s} {'|S|_mean':>9s} {'|S|_std':>8s}")
    stats = {}
    for name, genes in [('train', train), ('test', test)]:
        obs = np.stack([np.asarray(de[g], np.float32) for g in genes])
        pred, snorm = predict_split(genes)
        dc, pe, rt, dg = disc_resid(pred, obs, drift)
        stats[name] = dict(disc=dc, pear=pe, sn=snorm.mean())
        print(f"  {name:6s} {len(genes):4d} {dc:11.3f} {pe:11.3f} {rt:6.3f} {dg:6.3f} {snorm.mean():9.3f} {snorm.std():8.3f}")
    print(f"  --> disc GAP (train-test) = {stats['train']['disc']-stats['test']['disc']:+.3f} ; "
          f"pear GAP = {stats['train']['pear']-stats['test']['pear']:+.3f} ; "
          f"|S| ratio test/train = {stats['test']['sn']/(stats['train']['sn']+1e-9):.3f}")


if __name__ == '__main__':
    main()
