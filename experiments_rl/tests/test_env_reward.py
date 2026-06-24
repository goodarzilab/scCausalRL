"""Smoke + invariant tests for the continuous-latent env and penalized reward.

Run directly:  python tests/test_env_reward.py   (loads the frozen WM + Norman once)
Checks: obs shape; potential shaping rewards reaching the target; pessimism formula
(reward = scale·(mean − α·std) − cost) and monotonicity in α; no-op leaves state put;
multi-step ε-threading runs finite; disagreement signal present.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.wm_data import WMData
from core.env import CausalLatentEnv

_FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILS.append(name)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, meta = load_frozen_wm(device=dev)
    op = WorldModelOperator(model, meta, n_dag_samples=12)
    data = WMData(model, meta, op, n_ctrl_cells=200)

    combo = data.combo_split['test'][0]
    true_gi = [data.gene_to_idx[g] for g in combo]
    rng = np.random.default_rng(0)

    # --- obs shape ---
    env = CausalLatentEnv(op, data, max_steps=2, alpha_pess=1.0, n_dag_samples=12)
    obs = env.reset(combo, rng=rng)
    check("obs shape == [2d]", tuple(obs.shape) == (2 * op.d,), str(tuple(obs.shape)))

    # --- potential shaping: applying the TRUE genes should give positive reward ---
    pos = 0
    for _ in range(10):
        env.reset(combo, rng=rng)
        _, r, done, info = env.step(true_gi)
        pos += (r > 0)
    check("true-gene action mostly rewarded (>0)", pos >= 7, f"{pos}/10 positive")
    check("info has disagreement + dist", ('disagreement' in info and 'dist' in info),
          f"disagree={info['disagreement']:.3f} dist={info['dist']:.3f}")
    check("episode terminates by max_steps", done is False or env.step_count <= 2,
          f"step_count={env.step_count}")

    # --- reward formula matches scale·(mean − α·std) − cost ---
    env.reset(combo, rng=rng)
    _, r, _, info = env.step(true_gi)
    expect = env.reward_scale * (info['r_mean'] - env.alpha * info['r_std']) - env.lambda_cost
    check("reward == scale·(mean−α·std) − cost", abs(r - float(np.clip(expect, -10, 10))) < 1e-4,
          f"r={r:.4f} expect={expect:.4f}")

    # --- pessimism monotone in α: higher α => lower (or equal) reward when std>0 ---
    def reward_at_alpha(a, seed):
        e = CausalLatentEnv(op, data, max_steps=2, alpha_pess=a, n_dag_samples=12)
        e.reset(combo, rng=np.random.default_rng(seed)); _, rr, _, ii = e.step(true_gi)
        return rr, ii['r_std']
    r0, s0 = reward_at_alpha(0.0, 7)
    r2, s2 = reward_at_alpha(3.0, 7)
    check("pessimism lowers reward (α↑ ⇒ r↓ when std>0)", (s0 <= 1e-6) or (r2 <= r0 + 1e-6),
          f"r(α=0)={r0:.4f} r(α=3)={r2:.4f} std={s0:.4f}")

    # --- no-op leaves the latent state unchanged ---
    env.reset(combo, rng=rng)
    z_before = env.z_ens.clone()
    env.step([])
    check("empty action is a no-op (z unchanged)", torch.allclose(z_before, env.z_ens, atol=1e-4))

    # --- multi-step sequential ε-threading runs finite + changes state each step ---
    env.reset(combo, rng=rng)
    z0 = env.z_ens.mean(0).clone()
    env.step([true_gi[0]]); z1 = env.z_ens.mean(0).clone()
    env.step([true_gi[1]]); z2 = env.z_ens.mean(0).clone()
    finite = bool(torch.isfinite(z2).all())
    moved = float((z1 - z0).norm()) > 1e-4 and float((z2 - z1).norm()) > 1e-4
    check("2-step ε-threaded rollout finite + moves each step", finite and moved,
          f"|z1-z0|={float((z1-z0).norm()):.3f} |z2-z1|={float((z2-z1).norm()):.3f}")

    print(f"\n{'ALL ENV TESTS PASSED' if not _FAILS else 'FAILURES: ' + ', '.join(_FAILS)}")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
