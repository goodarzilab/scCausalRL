"""RL-free sanity for the merged Briggs fate reward (before any policy training):
  (1) rank all candidate SINGLE TFs by fate_score -> do known HSPC drivers land near the top?
  (2) score known cocktails (P1, P2, Sugimura-7, Lange-4) vs random k-sets -> are they high-percentile?
If the reward already ranks real HSPC biology high, it's a valid RL objective; if not, RL can't help.
Run: python analysis/briggs_reward_sanity.py --ckpt <deg3_t25 ckpt> --readout <readout_ridge npz>
"""
import os, sys, argparse, numpy as np, scipy.sparse as sp, torch
RL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, RL)
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.briggs_fate_reward import FateReward
torch.set_grad_enabled(False)
EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"

# canonical HSPC / hemato-endothelial master regulators (positive controls)
HSPC_DRIVERS = ['ETV2', 'GATA2', 'RUNX1', 'LMO2', 'TAL1', 'LYL1', 'GFI1B', 'SPI1', 'FLI1', 'ERG',
                'MEIS1', 'HOXA9', 'GATA1', 'MYB', 'HHEX', 'RUNX3', 'IKZF1', 'CEBPA', 'KLF1', 'MECOM']
COCKTAILS = {'P1': ['TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2'],
             'P2': ['DDIT3', 'MZF1', 'HOXB3', 'MIER1', 'SIRT6', 'ZNF85'],
             'Sugimura7': ['ERG', 'HOXA5', 'HOXA9', 'HOXA10', 'LCOR', 'RUNX1', 'SPI1'],
             'Lange4': ['ETV2', 'GATA2', 'LMO2', 'TAL1']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{EXP20}/results/grid/v2b_deg3_t25_frozen_r32_s1/checkpoints/phase1_epoch40.pt')
    ap.add_argument('--readout', default=f'{EXP20}/results/briggs/readout_ridge_v2b_deg3_t25_frozen_r32_s1.npz')
    ap.add_argument('--n_ctrl', type=int, default=256); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    model, meta = load_frozen_wm(a.ckpt, EXP20, a.device)
    op = WorldModelOperator(model, meta, n_dag_samples=8)
    gn = meta['gene_names']; g2i = meta['gene_to_idx']
    # control eps
    sys.path.insert(0, EXP20); from data.load_joung import load_joung
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None,
                    effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    _, _, eps = model.encoder(xc); eps0 = eps.mean(0)                          # [d] control

    R = FateReward(op, a.readout, eps0, lam=0.0)                               # pure fate for ranking
    reliable = [g for g in open(V + "/effective_tf_genes.txt").read().split() if g in g2i]
    print(f"[reward] candidates(reliable∩axis)={len(reliable)} | HSPC drivers on axis: "
          f"{[d for d in HSPC_DRIVERS if d in g2i]}", flush=True)

    # (1) rank singles
    sets = [[g2i[g]] for g in reliable]
    fs = R.score(sets).cpu().numpy()
    order = np.argsort(-fs); rank = {reliable[order[r]]: r for r in range(len(order))}
    n = len(reliable)
    print(f"\n[singles ranking] top-15 fate TFs: {[reliable[order[r]] for r in range(15)]}", flush=True)
    print(f"[HSPC driver percentiles] (1.0=top):", flush=True)
    pcts = []
    for d in HSPC_DRIVERS:
        if d in rank:
            pct = 1 - rank[d] / n; pcts.append(pct)
            print(f"    {d:8s} rank {rank[d]+1:4d}/{n}  pctile {pct*100:.0f}", flush=True)
    if pcts:
        from scipy.stats import mannwhitneyu
        driver_ranks = [rank[d] for d in HSPC_DRIVERS if d in rank]
        other = [r for g, r in rank.items() if g not in HSPC_DRIVERS]
        U = mannwhitneyu(driver_ranks, other, alternative='less').pvalue
        print(f"  --> HSPC drivers median pctile {np.median(pcts)*100:.0f}  (enrichment vs rest p={U:.1e})", flush=True)

    # (2) known cocktails vs random k-sets
    print(f"\n[cocktail fate scores vs random k-sets]", flush=True)
    for nm, tfs in COCKTAILS.items():
        on = [t for t in tfs if t in g2i]
        if len(on) < 2:
            print(f"    {nm}: only {len(on)}/{len(tfs)} TFs on axis — skip", flush=True); continue
        s = [g2i[t] for t in on]; fscore = float(R.score([s])[0])
        rand = R.score([[g2i[reliable[j]] for j in rng.choice(len(reliable), len(on), replace=False)] for _ in range(2000)]).cpu().numpy()
        pct = float((rand < fscore).mean() * 100)
        print(f"    {nm:10s} ({len(on)}/{len(tfs)} on-axis): fate {fscore:+.3f}  pctile {pct:.0f} vs random-{len(on)}", flush=True)
    print("done", flush=True)


if __name__ == '__main__':
    main()
