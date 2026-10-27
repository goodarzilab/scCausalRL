"""Score the RL PLANNERS on the SAME metric suite as the world-model eval — so the RL
methods are comparable to the WM table (discrimination, pearson_delta, effect-spearman,
DEG-auroc, fate identifiability) rather than just the planner `match`.

For each method, on the held-out combo pairs:
  - planner(target) -> chosen gene set
  - do-op predict Δ for that set (gate-aware, from hESC control)
  - assemble Pred[n_combos, n_genes] vs Obs (observed combo Δ)
  - gene_space(Pred, Obs)  -> pearson_delta, discrimination_cos, retrieval_top1,
                              effect_spearman, deg_auroc, de_direction
  - fate_space(Pred, Obs)  -> fate_top1, fate_betacos  (Cao-mouse NNLS identifiability)
Reference rows: 'true-combo do-op' (oracle: do-op of the ACTUAL combo genes = the WM's
combo ceiling) and 'observed-singles additive'.

Run (cpu_high_mem):  python eval_rl_full_metrics.py --device cpu
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.setup_stack import build_stack
from rl0_mpc.mpc_planner import MPCPlanner

EXP19 = '/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose'
CKPT = f'{EXP19}/results/grid/v2_esm_con_ugate/checkpoints/phase1_epoch60.pt'


class _Args:
    dataset = 'joung'; wm_root = EXP19; checkpoint = CKPT
    combo_path = None; singles_path = None; curated_topk = 256; n_ctrl = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None)
    ap.add_argument('--output', default='rl0_mpc/results/rl_full_metrics')
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    args = _Args(); args.device = dev
    model, meta, op, data = build_stack(args, dev, 8)
    from evaluation.heldout_zeroshot_eval import gene_space, fate_space   # exp19 on path via build_stack

    A = op.mean_A()
    ctrl_recon = op.decode(op.scm_forward(data.eps_ctrl_mean.unsqueeze(0), A)).squeeze(0).cpu().numpy()
    eps0 = data.eps_ctrl_mean.unsqueeze(0)
    G = len(data.gene_names)

    def do_pred(genes):
        if not genes:
            return np.zeros(G, np.float32)
        z, _, _ = op.apply(eps0, list(genes), A)
        return op.decode(z).squeeze(0).cpu().numpy() - ctrl_recon

    held = data.combo_split['test']
    Obs = np.stack([data.de_combo[k]['lfc'] for k in held])
    train_obs = np.stack([data.de_single[g]['lfc'] for g in data.single_split['train']])
    drift = train_obs.mean(0)                                  # shared differentiation program

    # ---- planners ----
    from rl1_seq_ppo.train import make_planner as mk_seq
    from rl2_set_ppo.train import make_planner as mk_set
    from rl1_seq_ppo.ppo import PPO
    from rl2_set_ppo.set_ppo import SetPPO
    from core.env import CausalLatentEnv
    env = CausalLatentEnv(op, data, max_steps=2, n_dag_samples=8)

    def load_seq(ckpt, pool):
        data.set_pool(pool)
        p = PPO(state_dim=op.d, n_actions=len(data.candidate_idx), device=dev)
        p.load(ckpt); return mk_seq(p, op, data, env)

    def load_set(ckpt, pool):
        data.set_pool(pool)
        ag = SetPPO(state_dim=op.d, n_candidates=len(data.candidate_idx), k_max=2, device=dev)
        ag.policy.load_state_dict(torch.load(ckpt, map_location=dev)['policy'])
        return mk_set(ag, op, data, env)

    def mpc_planner(pool, beam=16):
        data.set_pool(pool)
        mpc = MPCPlanner(op, data, max_set=2, beam_width=beam, n_dag_samples=8, pessimism_alpha=1.0)
        return lambda key: mpc(key)

    R = 'rl1_seq_ppo/results'; R2 = 'rl2_set_ppo/results'; R3 = 'rl3_mpc_distill/results'
    specs = [
        ('true-combo do-op (oracle)', 'true', None, 'screened'),
        ('MPC · curated',             'mpc',  None, 'curated'),
        ('distill_BC+PPO · curated',  'set',  f'{R3}/joung_distill_curated/set_ppo_distill.pt', 'curated'),
        ('set-PPO · RESID · curated', 'set',  f'{R2}/joung_set_resid_curated/set_ppo.pt', 'curated'),
        ('seq-PPO · RESID · curated', 'seq',  f'{R}/joung_seq_resid_curated/seq_ppo.pt', 'curated'),
        ('set-PPO · raw · curated',   'set',  f'{R2}/joung_set_raw_curated/set_ppo.pt', 'curated'),
        ('seq-PPO · raw · curated',   'seq',  f'{R}/joung_seq_raw_curated/seq_ppo.pt', 'curated'),
        ('set-PPO · raw · full(orig)','set',  f'{R2}/joung_ugate/set_ppo.pt', 'screened'),
    ]

    def gs_of(P, O):
        r = gene_space(P, O)
        return r[0] if isinstance(r, tuple) else r

    def fs_of(P, O):
        r = fate_space(P, O, train_obs, data.gene_names)
        return r[0] if isinstance(r, tuple) else r

    rows, chosen_sets, mpc_sets = [], {}, None
    for name, kind, ckpt, pool in specs:
        try:
            genes_per = None
            if kind == 'true':
                data.set_pool(pool)
                genes_per = [[data.gene_to_idx[g] for g in k] for k in held]
            elif kind == 'mpc':
                planner = mpc_planner(pool); genes_per = [list(planner(k)) for k in held]
            else:
                if not os.path.exists(ckpt):
                    print(f"skip {name}: no ckpt {ckpt}"); continue
                planner = load_seq(ckpt, pool) if kind == 'seq' else load_set(ckpt, pool)
                genes_per = [list(planner(k)) for k in held]
            chosen_sets[name] = [frozenset(g) for g in genes_per]
            if name.startswith('MPC'):
                mpc_sets = chosen_sets[name]
            Pred = np.stack([do_pred(g) for g in genes_per])
            gs = gs_of(Pred, Obs); fs = fs_of(Pred, Obs)
            # DRIFT-REMOVED (residual) metrics — same convention as the WM table
            gsr = gs_of(Pred - drift, Obs - drift)
            row = {'method': name,
                   'pearson_delta': gs.get('pearson_delta'), 'pearson_delta_resid': gsr.get('pearson_delta'),
                   'discrimination_cos': gs.get('discrimination_cos'),
                   'discrimination_resid': gsr.get('discrimination_cos'),
                   'effect_spearman_resid': gsr.get('effect_spearman'),
                   'deg_auroc_resid': gsr.get('deg_auroc'),
                   'fate_top1': fs.get('fate_top1'), 'fate_betacos': fs.get('fate_betacos')}
            rows.append(row)
            print(f"{name:28s} pear_d_res={row['pearson_delta_resid']:.3f} disc_res={row['discrimination_resid']:.3f} "
                  f"effSpear_res={row['effect_spearman_resid']:.3f} | pear_d_raw={row['pearson_delta']:.3f} "
                  f"fate_t1={row['fate_top1']:.3f}", flush=True)
        except Exception as e:
            import traceback; print(f"FAIL {name}: {e}"); traceback.print_exc()

    # MPC-agreement: how much each planner's chosen gene SET matches MPC's (Jaccard) →
    # answers "is distill just MPC?" and "does PPO change the picks?"
    agree = {}
    if mpc_sets is not None:
        for name, sets in chosen_sets.items():
            j = np.mean([len(a & b) / max(1, len(a | b)) for a, b in zip(sets, mpc_sets)])
            ident = np.mean([float(a == b) for a, b in zip(sets, mpc_sets)])
            agree[name] = {'jaccard_vs_mpc': round(float(j), 3), 'identical_set_frac': round(float(ident), 3)}
            print(f"  [agree-vs-MPC] {name:28s} jaccard={agree[name]['jaccard_vs_mpc']:.3f} "
                  f"identical={agree[name]['identical_set_frac']:.3f}")

    json.dump({'n_held_combos': len(held), 'rows': rows, 'mpc_agreement': agree},
              open(f'{a.output}/summary.json', 'w'), indent=2)
    print(f"\nsaved → {a.output}/summary.json  (n held-out combos={len(held)})")


if __name__ == '__main__':
    main()
