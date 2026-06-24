"""Local smoke for the large-pool task before SLURM submission (one Norman load):
  (1) batched apply_many == single apply (operator correctness),
  (2) full-pool MPC returns a 2-gene plan + timing on a couple held-out combos,
  (3) seq-PPO and set-PPO run a few iters at |actions|=8000 without shape errors.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.wm_data import WMData
from core.env import CausalLatentEnv

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model, meta = load_frozen_wm(device=dev)
op = WorldModelOperator(model, meta, n_dag_samples=4)
data = WMData(model, meta, op, n_ctrl_cells=200)
data.set_pool('full')
eps0 = data.eps_ctrl_mean; A = op.mean_A()

print("\n(1) batched operator correctness")
gi = data.combo_split['test'][0]
test_genes = [data.gene_to_idx[g] for g in gi] + [0, 4000, 7999]
S = op.shift_for(test_genes)
z_many = op.apply_many(eps0, S, A)
err = 0.0
for r, g in enumerate(test_genes):
    z_single, _, _ = op.apply(eps0.unsqueeze(0), [g], A)
    err = max(err, float((z_many[r] - z_single.squeeze(0)).abs().max()))
print(f"  max|apply_many - apply| over 5 genes = {err:.2e}  ({'OK' if err < 1e-3 else 'MISMATCH'})")

print("\n(2) full-pool MPC (|pool|=%d)" % len(data.candidate_idx))
from rl0_mpc.mpc_planner import MPCPlanner
mpc = MPCPlanner(op, data, max_set=2, beam_width=8, n_dag_samples=4)
for key in data.combo_split['test'][:2]:
    t = time.time(); chosen, info = mpc.plan(key, return_info=True)
    print(f"  {'+'.join(key)}: chose {[data.gene_names[i] for i in chosen]}  match={info['match']:.3f}  "
          f"({time.time()-t:.1f}s, {info['n_candidates']} cand)")

print("\n(3) PPO few-iter shape checks at |actions|=8000")
from rl1_seq_ppo.ppo import PPO
from rl1_seq_ppo.train import train as seq_train
env = CausalLatentEnv(op, data, max_steps=2, alpha_pess=1.0, n_dag_samples=4)
tk = list(data.single_split['train'])[:10] + list(data.combo_split['train'])[:10]
ppo = PPO(state_dim=op.d, n_actions=len(data.candidate_idx), device=dev)
seq_train(op, data, env, ppo, tk, n_iters=60, n_steps_per_update=32, rng=np.random.default_rng(0), log_every=60)
print("  seq-PPO 60 iters OK")
from rl2_set_ppo.set_ppo import SetPPO
from rl2_set_ppo.train import train as set_train
env1 = CausalLatentEnv(op, data, max_steps=1, alpha_pess=1.0, n_dag_samples=4)
agent = SetPPO(state_dim=op.d, n_candidates=len(data.candidate_idx), k_max=2, device=dev)
set_train(op, data, env1, agent, tk, n_iters=60, n_steps_per_update=32, rng=np.random.default_rng(0), log_every=60)
print("  set-PPO 60 iters OK")
print("\nSMOKE OK")
