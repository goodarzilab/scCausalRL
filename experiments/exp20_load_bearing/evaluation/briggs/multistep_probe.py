"""C — Multi-step fate-dynamics probe. The do-op predicts the Joung 7-day snapshot; Briggs is a 14-28-day
selected endpoint. Question: if we ITERATE the do-op (apply combo -> decode -> re-encode -> apply again),
simulating more differentiation "time", does the fate signal rise toward the endpoint AND correlate BETTER
with the observed endpoint rho_obs? If yes, a multi-step reward closes the assay-gap that caps one-step at ~0
for high k. For a sample of observed combos, roll T steps and report per-step: mean readout-fate, and
Spearman(fate_t, rho_obs). (WM-dynamics experiment, no RL training.)
Run: python evaluation/briggs/multistep_probe.py --ckpt <prod> --readout <prod joung_ctrl briggs readout>
"""
import os, sys, argparse, csv, numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.briggs.hspc_sim import build_cao_hsc_signature
from data.load_joung import load_joung
from scipy.stats import spearmanr
torch.set_grad_enabled(False)
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
E = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{E}/results/grid/v2b_prod_deg8_all40/checkpoints/phase1_epoch40.pt')
    ap.add_argument('--readout', default=f'{E}/results/briggs/readout_ridge_v2b_prod_deg8_all40_ans_joungctrl_rho_briggs.npz')
    ap.add_argument('--answer', default=f'{E}/results/briggs/observed_combo_targets_joungctrl.csv')
    ap.add_argument('--T', type=int, default=5); ap.add_argument('--n_ctrl', type=int, default=64)
    ap.add_argument('--n_combos', type=int, default=1500); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    model, gene_names, _ = load_shift_model(a.ckpt, a.device); gn = list(map(str, gene_names)); g2i = {g: i for i, g in enumerate(gn)}
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt",
                    split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    _, _, eps0 = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0)
    rd = np.load(a.readout); coef = torch.tensor(rd['coef'], dtype=torch.float32); icpt = float(rd['intercept'])
    rmean = torch.tensor(rd['mean'], dtype=torch.float32); rscale = torch.tensor(rd['scale'], dtype=torch.float32)
    def fate(z): return float((((z - rmean) / rscale) @ coef) + icpt)      # deployed readout scorer

    obs = [r for r in csv.DictReader(open(a.answer))]
    rng.shuffle(obs); rows = []
    for r in obs:
        tfs = r['combo'].split('|'); gi = [g2i[g] for g in tfs if g in g2i]
        if len(gi) != len(tfs): continue
        gi_t = torch.tensor(gi, dtype=torch.long)
        eps = eps0.clone(); traj = []
        for t in range(a.T):
            z_prev = model.dag.scm_forward(eps, A)
            z_post, _ = model.intervene_shift_mutilate(eps, gi_t, A, z_prev, u_mode='shift', shift_beta=1.0)
            traj.append(fate(z_post.mean(0)))
            x = model.decoder(z_post)                                       # [n_ctrl, G]
            _, _, eps = model.encoder(x)                                    # re-encode -> next-step exogenous noise
        rows.append((float(r['rho_briggs']), int(r['k']), traj))
        if len(rows) >= a.n_combos: break

    rho = np.array([x[0] for x in rows]); ks = np.array([x[1] for x in rows]); F = np.array([x[2] for x in rows])  # [N, T]
    print(f"[multistep] {len(rows)} combos, T={a.T}, n_ctrl={len(xc)}", flush=True)
    print(f"{'step':>5} {'mean_fate':>10} {'Spearman(fate_t, rho_obs)':>26} {'k>=5 Spearman':>15}", flush=True)
    for t in range(a.T):
        sp_all = spearmanr(F[:, t], rho).correlation
        m = ks >= 5; sp_hi = spearmanr(F[m, t], rho[m]).correlation if m.sum() > 20 else float('nan')
        print(f"{t+1:>5} {F[:, t].mean():>10.3f} {sp_all:>26.3f} {sp_hi:>15.3f}", flush=True)
    # does iterating help? compare t=1 vs best-t
    best_t = int(np.argmax([spearmanr(F[:, t], rho).correlation for t in range(a.T)]))
    print(f"\n[verdict] 1-step Spearman {spearmanr(F[:,0],rho).correlation:+.3f} -> best at step {best_t+1} "
          f"{spearmanr(F[:,best_t],rho).correlation:+.3f} | mean fate step1 {F[:,0].mean():.3f} -> stepT {F[:,-1].mean():.3f} "
          f"({'RISES (multi-step helps)' if F[:,-1].mean()>F[:,0].mean()+0.02 else 'flat/declines'})", flush=True)


if __name__ == '__main__':
    main()
