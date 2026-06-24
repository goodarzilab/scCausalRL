"""One-shot core verification + value-add ladder (single Norman load).

Loads the frozen WM + Norman once, then:
  1) env reward sanity (true-gene action rewarded under the eval-A mean / ensemble-std fix),
  2) MPC sanity (exact-target recovery; reaches closer than control),
  3) the value-add ladder on the held-out combos: random → greedy-1 → additive → MPC.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.wm_data import WMData
from core.env import CausalLatentEnv
from core.baselines import Planners
from core.value_add_ladder import run_ladder
from rl0_mpc.mpc_planner import MPCPlanner

N_DAG = 8


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, meta = load_frozen_wm(device=dev)
    op = WorldModelOperator(model, meta, n_dag_samples=N_DAG)
    data = WMData(model, meta, op, n_ctrl_cells=300)
    rng = np.random.default_rng(0)

    print("\n========== (1) ENV REWARD SANITY ==========")
    env = CausalLatentEnv(op, data, max_steps=2, alpha_pess=1.0, n_dag_samples=N_DAG)
    combo = data.combo_split['test'][0]
    true_gi = [data.gene_to_idx[g] for g in combo]
    pos = 0; rs = []
    for _ in range(10):
        env.reset(combo, rng=rng); _, r, _, info = env.step(true_gi); pos += (r > 0); rs.append(r)
    print(f"  true-gene action reward: {pos}/10 positive  mean={np.mean(rs):.3f}  (match={info['match']:.3f})")
    env.reset(combo); zc = env.z.clone(); env.step([])
    print(f"  no-op keeps state: {bool(torch.allclose(zc, env.z, atol=1e-4))}")

    print("\n========== (2) MPC SANITY (gene-space) ==========")
    # the TRUE genes of a held-out single should be a strong (often top) plan for that
    # single's own DE target — sanity that gene-space MPC tracks real perturbations.
    g_str = data.single_split['test'][0]
    mpc1 = MPCPlanner(op, data, max_set=1, beam_width=4, n_dag_samples=4, pessimism_alpha=0.0)
    chosen, info = mpc1.plan(g_str, return_info=True)
    print(f"  single target '{g_str}': MPC chose {[data.gene_names[i] for i in chosen]}  "
          f"match={info['match']:.3f}  recovered_true={data.gene_to_idx[g_str] in chosen}")

    print("\n========== (3) VALUE-ADD LADDER (held-out combos) ==========")
    df = run_ladder(op, data, split='combo', max_set=2, beam=6, alpha=1.0, n_dag=N_DAG)
    import pandas as pd
    pd.set_option('display.width', 160)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    os.makedirs('rl0_mpc/results', exist_ok=True)
    df.to_csv('rl0_mpc/results/ladder_combo.csv', index=False)
    print("\nsaved → rl0_mpc/results/ladder_combo.csv")


if __name__ == "__main__":
    main()
