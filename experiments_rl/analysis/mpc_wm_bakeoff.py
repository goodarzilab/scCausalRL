"""Which frozen WM gives the best RL eval? MPC (the proven planner; needs no per-WM training)
on the 12 held-out Joung combo pairs, across the WMs trained alongside the combo model. Reports
both bases so residual-trained and raw-trained WMs are judged fairly:
  pΔ_raw / pΔ_resid   signed-Pearson(MPC-set Δ, obs Δ) on DEGs, raw and drift-removed
  disc_resid          12-way retrieval: is each combo's pred Δ (resid) nearest its OWN obs Δ
Run via scripts/mpc_wm_bakeoff.sbatch.
"""
import os, sys, argparse, gc
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import build_stack
from rl0_mpc.mpc_planner import MPCPlanner
EXP19 = '/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose'
WMS = [  # (tag, checkpoint) — combo model + the variants trained alongside it
    ('ugate (current RL WM)', 'v2_esm_con_ugate/checkpoints/phase1_epoch60.pt'),
    ('ugate_combo (combo-trained)', 'v2_esm_con_ugate_combo/checkpoints/phase1_epoch20.pt'),
    ('combo_resid', 'v2_combo_resid/checkpoints/phase1_epoch60.pt'),
    ('combo_gate512', 'v2_combo_gate512/checkpoints/phase1_epoch60.pt'),
    ('combo_resid_hicon_g512', 'v2_combo_resid_hicon_g512/checkpoints/phase1_epoch60.pt'),
    ('esm_con_ident', 'v2_esm_con_ident/checkpoints/phase1_epoch60.pt'),
]


class _Args:
    dataset = 'joung'; wm_root = EXP19; combo_path = None; singles_path = None
    curated_topk = 256; n_ctrl = 500


def pear(p, o, sel):
    p, o = p[sel], o[sel]
    if len(p) < 3 or p.std() < 1e-8 or o.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(p, o)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default=None); a = ap.parse_args()
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_grad_enabled(False)
    rows = []
    for tag, rel in WMS:
        ckpt = f'{EXP19}/results/grid/{rel}'
        if not os.path.exists(ckpt):
            print(f"[skip] {tag}: no checkpoint"); continue
        args = _Args(); args.checkpoint = ckpt; args.device = dev
        try:
            model, meta, op, data = build_stack(args, dev, 8); data.set_pool('curated')
        except Exception as e:
            print(f"[skip] {tag}: {e}"); continue
        A = op.mean_A(); ctrl = op.decode(op.scm_forward(data.eps_ctrl_mean.unsqueeze(0), A)).squeeze(0).cpu().numpy()
        eps0 = data.eps_ctrl_mean.unsqueeze(0)
        drift = np.stack([data.de_single[g]['lfc'] for g in data.single_split['train']]).mean(0)
        held = data.combo_split['test']
        mpc = MPCPlanner(op, data, max_set=2, beam_width=16, n_dag_samples=8, pessimism_alpha=1.0)
        preds, obss = [], []
        for k in held:
            genes = mpc(k)                                    # MPCPlanner returns gene INDICES
            z, _, _ = op.apply(eps0, list(genes), A)
            preds.append(op.decode(z).squeeze(0).cpu().numpy() - ctrl)
            obss.append(data.de_combo[k]['lfc'])
        P, O = np.stack(preds), np.stack(obss)
        Pr, Or = P - drift, O - drift
        praw = np.mean([pear(P[i], O[i], np.abs(O[i]) > data.thresh) for i in range(len(held))])
        pres = np.mean([pear(Pr[i], Or[i], np.abs(Or[i]) > data.thresh) for i in range(len(held))])
        # 12-way resid retrieval (discrimination)
        Prn = Pr / (np.linalg.norm(Pr, 1, keepdims=True) + 1e-9); Orn = Or / (np.linalg.norm(Or, 1, keepdims=True) + 1e-9)
        Sim = Prn @ Orn.T; disc = float(np.mean(Sim.argmax(1) == np.arange(len(held))))
        rows.append({'WM': tag, 'gate_aware': op.gate_aware, 'pΔ_raw': round(praw, 3),
                     'pΔ_resid': round(pres, 3), 'disc_resid': round(disc, 3)})
        print(f"  {tag:32s} pΔ_raw={praw:.3f} pΔ_resid={pres:.3f} disc_resid={disc:.3f}", flush=True)
        del model, op, data; gc.collect(); torch.cuda.empty_cache() if dev == 'cuda' else None

    df = pd.DataFrame(rows)
    print(f"\n{'='*80}\nMPC held-out combo eval across WMs (12 pairs, curated pool)")
    print(df.to_string(index=False))
    print("\nchance disc_resid = 1/12 = 0.083 | ugate reference: pΔ_resid≈0.49 disc≈0.50 (full-metrics)")
    df.to_csv(f'{HERE}/mpc_wm_bakeoff.csv', index=False); print("saved -> mpc_wm_bakeoff.csv")


if __name__ == '__main__':
    main()
