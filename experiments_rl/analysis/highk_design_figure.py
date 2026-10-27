"""High-k (k<=6) HSPC cocktail-design comparison FIGURE + tables.

Uniform leak-free scoring of every method under the SAME frozen Joung+Briggs winner WM
(jb_ft_replay_co8) and the SAME leave-HIT-out Ridge fate readout (rho_briggs, holdhits — the
14 answer-key TFs were held out of the readout fit, so scoring never saw the answer key).

Bars   : MPC-beam@k6, set-PPO@k6, greedy@k6  (learned RL vs search vs greedy)
Refs   : Lange-4, Sugimura-7, P1, P2         (literature / Briggs answer-key cocktails, dashed lines)
Score  : fate = Ridge_fate(z_post(set)) under the eval-mode DAG (mean_A); higher = more HSPC-like.
Overlap: |set ∩ answer_key| where answer_key = P1 ∪ P2 ∪ Lange-4 (the 14 held-out TFs).

Greedy + literature are computed here directly (no training). MPC / set-PPO picks are read from the
CSVs their SLURM jobs wrote (top row by pessimistic r~), then RE-SCORED here with the identical readout
so every bar is apples-to-apples. Run via scripts/rl_cpu.sbatch (loads the WM + Joung control cells).
"""
import os, sys, csv, argparse, numpy as np, scipy.sparse as sp, torch
RL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, RL)
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.briggs_fate_reward import FateReward
torch.set_grad_enabled(False)
EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"

P1 = {'TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2'}
P2 = {'DDIT3', 'MZF1', 'HOXB3', 'MIER1', 'SIRT6', 'ZNF85'}
LANGE4 = {'ETV2', 'GATA2', 'LMO2', 'TAL1'}
SUGIMURA7 = ['ERG', 'HOXA5', 'HOXA9', 'HOXA10', 'LCOR', 'RUNX1', 'SPI1']
ANSWER_KEY = P1 | P2 | LANGE4                                        # the 14 held-out TFs

# Okabe-Ito (colorblind-safe)
OI = dict(black='#000000', orange='#E69F00', skyblue='#56B4E9', green='#009E73',
          yellow='#F0E442', blue='#0072B2', vermillion='#D55E00', purple='#CC79A7')


def read_top_combo(csv_path):
    """Top row (best pessimistic r~) of an MPC/set-PPO candidate CSV -> (tf_list, reported_r_pess)."""
    if not os.path.exists(csv_path):
        return None, None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None, None
    r0 = rows[0]                                                     # scripts write sorted by -r~
    return r0['combo'].split('|'), float(r0.get('r_pess', 'nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{EXP20}/results/grid/jb_ft_replay_co8/checkpoints/ckpt.pt')
    ap.add_argument('--readout', default=f'{EXP20}/results/briggs/readout_ridge_jb_ft_replay_co8_rho_briggs_holdhits.npz')
    ap.add_argument('--support_feats', default=f'{EXP20}/results/briggs/readout_feats_jb_ft_replay_co8_ans.npz')
    ap.add_argument('--exclude_tfs', default='ETV2,GATA2,LMO2,TAL1,RPA3,ZFP90,RNF138,KLF2,DDIT3,MZF1,HOXB3,MIER1,SIRT6,ZNF85')
    ap.add_argument('--mpc_csv', default=f'{EXP20}/results/briggs/highk_mpc_k6.csv')
    ap.add_argument('--ppo_csv', default=f'{EXP20}/results/briggs/highk_setppo_k6.csv')
    ap.add_argument('--n_ctrl', type=int, default=256); ap.add_argument('--n_dag', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=f'{EXP20}/results/briggs/figures/rl_goal3b_highk.png')
    a = ap.parse_args()

    model, meta = load_frozen_wm(a.ckpt, EXP20, a.device)
    op = WorldModelOperator(model, meta, n_dag_samples=a.n_dag)
    g2i = meta['gene_to_idx']
    sys.path.insert(0, EXP20); from data.load_joung import load_joung
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None,
                    effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    _, _, eps = model.encoder(xc); eps0 = eps.mean(0)

    # single leak-free readout -> R.score == R._fate_all[0,0] == the Ridge fate score
    R = FateReward(op, [a.readout], eps0, lam=1.0, n_dag=a.n_dag)
    excl = [t for t in a.exclude_tfs.split(',') if t]
    R.set_support(a.support_feats, 0.0, exclude_tfs=excl or None)     # maha for reporting only (gamma=0)
    A_eval = op.mean_A()

    # pool (identical to MPC/PPO): reliable-495 UNION Briggs on-axis
    reliable = set(open(V + "/effective_tf_genes.txt").read().split())
    briggs_tfs = set()
    for r in csv.DictReader(open(f'{EXP20}/results/briggs/observed_combo_targets.csv')):
        briggs_tfs.update(r['combo'].split('|'))
    pool = sorted({g for g in (reliable | briggs_tfs) if g in g2i})
    print(f"[pool] {len(pool)} TFs on-axis", flush=True)

    def score_set(tfs):
        gi = [g2i[t] for t in tfs if t in g2i]
        if not gi:
            return float('nan'), float('nan'), gi
        z = op.apply_sets(eps0, [gi], A_eval)
        fate = float(R._fate_all(z)[0, 0]); maha = float(R.mahalanobis(z)[0])
        return fate, maha, gi

    def overlap(tfs):
        return sorted(set(tfs) & ANSWER_KEY)

    # ---- GREEDY: top-6 single-TF do-ops by leak-free readout ----
    singles = [[g2i[g]] for g in pool]
    z_s = op.apply_sets(eps0, singles, A_eval)
    fate_s = R._fate_all(z_s)[0].cpu().numpy()
    order = np.argsort(-fate_s)
    greedy_tfs = [pool[i] for i in order[:6]]
    print("\n[greedy] top-6 single-TF readout scores:", flush=True)
    for i in order[:6]:
        print(f"    {pool[i]:10s} single-fate {fate_s[i]:+.4f}", flush=True)
    g_fate, g_maha, _ = score_set(greedy_tfs)

    # ---- LEARNED / SEARCH picks (re-scored under the same readout) ----
    mpc_tfs, mpc_r = read_top_combo(a.mpc_csv)
    ppo_tfs, ppo_r = read_top_combo(a.ppo_csv)

    results = {}
    if mpc_tfs is not None:
        f, m, _ = score_set(mpc_tfs); results['MPC-beam@k6'] = dict(tfs=mpc_tfs, fate=f, maha=m, r_pess=mpc_r, color=OI['blue'])
    if ppo_tfs is not None:
        f, m, _ = score_set(ppo_tfs); results['set-PPO@k6'] = dict(tfs=ppo_tfs, fate=f, maha=m, r_pess=ppo_r, color=OI['vermillion'])
    results['greedy@k6'] = dict(tfs=greedy_tfs, fate=g_fate, maha=g_maha, r_pess=float('nan'), color=OI['green'])

    # ---- LITERATURE references (dashed lines) ----
    lit = {}
    for nm, tfs, col in [('Lange-4', sorted(LANGE4), OI['orange']),
                         ('Sugimura-7', SUGIMURA7, OI['skyblue']),
                         ('P1', sorted(P1), OI['purple']),
                         ('P2', sorted(P2), OI['black'])]:
        f, m, gi = score_set(tfs)
        lit[nm] = dict(tfs=tfs, fate=f, maha=m, color=col, present=len(gi), k=len(tfs))

    # ---- print tables ----
    print(f"\n{'='*92}\nHIGH-k (k<=6) HSPC DESIGN — uniform leak-free readout (rho_briggs, holdhits)\n{'='*92}", flush=True)
    print(f"{'method':16s} {'k':>2s} {'fate':>8s} {'maha':>6s} {'ans_overlap':>12s}  TFs", flush=True)
    for nm, d in results.items():
        ov = overlap(d['tfs'])
        print(f"{nm:16s} {len(d['tfs']):2d} {d['fate']:+8.4f} {d['maha']:6.2f} {len(ov):>7d}/{len(ANSWER_KEY):<4d}  {sorted(d['tfs'])}  ∩ans={ov}", flush=True)
    print(f"\n{'literature':16s} {'k':>2s} {'fate':>8s} {'maha':>6s}", flush=True)
    for nm, d in lit.items():
        print(f"{nm:16s} {d['k']:2d} {d['fate']:+8.4f} {d['maha']:6.2f}   {d['tfs']} (present {d['present']}/{d['k']})", flush=True)

    # ---- FIGURE ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 6.2))
    names = list(results.keys()); vals = [results[n]['fate'] for n in names]
    cols = [results[n]['color'] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, vals, width=0.62, color=cols, edgecolor='white', linewidth=1.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('HSPC fate-readout score  (Ridge $\\rho_{Briggs}$, held-out)', fontsize=12)
    ax.axhline(0, color='#888888', lw=1, zorder=1)
    ax.grid(axis='y', color='#e6e6e6', lw=0.8, zorder=0)

    ymax = max(vals + [lit[n]['fate'] for n in lit])
    ymin = min(vals + [lit[n]['fate'] for n in lit] + [0.0])
    pad = 0.14 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad * 3.2)

    # bar annotations: value + picked TFs + answer-key overlap
    for xi, n in zip(x, names):
        d = results[n]; ov = overlap(d['tfs'])
        top = d['fate'] + (0.02 if d['fate'] >= 0 else -0.02) * (ymax - ymin + 1e-6) / 0.14
        ax.text(xi, d['fate'] + pad * 0.18, f"{d['fate']:+.3f}", ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#222222', zorder=4)
        tf_str = '\n'.join(sorted(d['tfs']))
        ax.text(xi, ymin - pad * 0.55, f"k={len(d['tfs'])}  ∩ans={len(ov)}/{len(ANSWER_KEY)}\n{tf_str}",
                ha='center', va='top', fontsize=7.4, color='#444444', zorder=4)

    # literature dashed reference lines
    for nm, d in lit.items():
        ax.axhline(d['fate'], ls='--', lw=1.8, color=d['color'], zorder=2,
                   label=f"{nm} (k={d['k']}): {d['fate']:+.3f}")
    ax.legend(loc='upper right', fontsize=9, frameon=True, title='literature cocktails', title_fontsize=9)

    ax.set_title("High-k (k$\\leq$6) HSPC design — search vs learned RL vs literature", fontsize=13, fontweight='bold', pad=26)
    n_note = (f"WINNER Joung+Briggs WM (jb_ft_replay_co8). HELD-OUT / leak-free readout: fit on 3847 non-hit combos "
              f"(14 answer-key TFs held out).\nUniform scoring: fate = Ridge$_{{fate}}$(z_post) under eval DAG; "
              f"pool n={len(pool)} on-axis TFs; n_ctrl={a.n_ctrl}; n_dag={a.n_dag}. answer key = P1$\\cup$P2$\\cup$Lange-4 (14 TFs).")
    fig.text(0.5, 0.005, n_note, ha='center', va='bottom', fontsize=7.6, color='#555555')
    fig.subplots_adjust(bottom=0.30, top=0.88)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches='tight')
    print(f"\nwrote {a.out}", flush=True)

    # summary CSV next to the figure
    sc = a.out.replace('.png', '_scores.csv')
    with open(sc, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'k', 'fate', 'maha', 'ans_overlap', 'answer_key_hits', 'tfs'])
        for nm, d in results.items():
            ov = overlap(d['tfs']); w.writerow([nm, len(d['tfs']), round(d['fate'], 4), round(d['maha'], 3), len(ov), '|'.join(ov), '|'.join(sorted(d['tfs']))])
        for nm, d in lit.items():
            ov = overlap(d['tfs']); w.writerow([nm, d['k'], round(d['fate'], 4), round(d['maha'], 3), len(ov), '|'.join(ov), '|'.join(d['tfs'])])
    print(f"wrote {sc}", flush=True)
    print("done", flush=True)


if __name__ == '__main__':
    main()
