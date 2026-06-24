"""Tests for the MPC planner (M0).

Run directly: python rl0_mpc/test_mpc_planner.py
Checks: beam=1/max_set=1 == greedy-1 argmin; an EXACT target (the operator's own
z_post for a known gene) is recovered at ~0 cost; MPC reaches closer than the control
baseline on held-out combos; pessimism raises cost monotonically in α.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.wm_data import WMData
from core.baselines import Planners
from rl0_mpc.mpc_planner import MPCPlanner

_FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILS.append(name)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, meta = load_frozen_wm(device=dev)
    op = WorldModelOperator(model, meta, n_dag_samples=8)
    data = WMData(model, meta, op, n_ctrl_cells=200)
    planners = Planners(op, data)

    combo = data.combo_split['test'][0]

    # 1) beam=1, max_set=1, alpha=0 == greedy-1 argmin over singles (under eval-A,
    #    the cost's mean frame; α=0 ⇒ the ensemble std term is inert)
    mpc_g = MPCPlanner(op, data, max_set=1, beam_width=1, n_dag_samples=8, pessimism_alpha=0.0)
    A_fixed = op.sample_dags(8)
    g_mpc = mpc_g.plan(combo, A_batch=A_fixed)
    zt = data.target_for(combo)
    A_eval = op.mean_A()
    eps0 = data.eps_ctrl_mean.unsqueeze(0)
    costs = []
    for gi in data.candidate_idx:
        z, _, _ = op.apply(eps0, [gi], A_eval)
        costs.append(float(((z.squeeze(0) - zt) ** 2).sum()))
    g_arg = [data.candidate_idx[int(np.argmin(costs))]]
    check("beam1/max_set1 == exhaustive argmin", g_mpc == g_arg, f"{g_mpc} vs {g_arg}")

    # 2) exact target: operator's own z_post for a held-out single gene → MPC recovers it
    g_true = data.gene_to_idx[data.single_split['test'][0]]
    z_exact, _, _ = op.apply(data.eps_ctrl_mean.unsqueeze(0), [g_true], data.A_ref)
    # temporarily inject this as a target
    data.z_target_single['__exact__'] = z_exact.squeeze(0)
    mpc1 = MPCPlanner(op, data, max_set=1, beam_width=4, n_dag_samples=4, pessimism_alpha=0.0)
    chosen, info = mpc1.plan('__exact__', return_info=True)
    check("exact single target recovered", chosen == [g_true], f"chose {chosen}, true {[g_true]}")
    check("exact target near-zero cost", info['dist'] < 0.25, f"dist={info['dist']:.3f}")

    # 3) MPC (max_set=2) reaches closer than control on held-out combos
    mpc2 = MPCPlanner(op, data, max_set=2, beam_width=6, n_dag_samples=8, pessimism_alpha=1.0)
    ratios = []
    for k in data.combo_split['test'][:12]:
        chosen, info = mpc2.plan(k, return_info=True)
        ratios.append(info['dist'] / (float((data.z0 - data.target_for(k)).norm()) + 1e-8))
    ratios = np.array(ratios)
    check("MPC reaches closer than control baseline", (ratios < 1).mean() >= 0.8,
          f"frac<1={float((ratios<1).mean()):.2f} median={np.median(ratios):.3f}")

    # 4) pessimism raises cost monotonically in α (same A_batch)
    c_lo, _ = MPCPlanner(op, data, max_set=2, beam_width=6, n_dag_samples=8,
                         pessimism_alpha=0.0).plan(combo, A_batch=A_fixed, return_info=True)
    info_lo = MPCPlanner(op, data, max_set=2, beam_width=6, n_dag_samples=8,
                         pessimism_alpha=0.0).plan(combo, A_batch=A_fixed, return_info=True)[1]
    info_hi = MPCPlanner(op, data, max_set=2, beam_width=6, n_dag_samples=8,
                         pessimism_alpha=3.0).plan(combo, A_batch=A_fixed, return_info=True)[1]
    check("pessimism raises plan cost (α↑ ⇒ cost↑)", info_hi['cost'] >= info_lo['cost'] - 1e-6,
          f"cost(α=0)={info_lo['cost']:.3f} cost(α=3)={info_hi['cost']:.3f}")

    print(f"\n{'ALL MPC TESTS PASSED' if not _FAILS else 'FAILURES: ' + ', '.join(_FAILS)}")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
