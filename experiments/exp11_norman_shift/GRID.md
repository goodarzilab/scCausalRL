# exp8 — DAG-revival grid (make A do causal work, not get bypassed)

**Motivation.** On exp2 the learned shift `Sᵖ` fully **bypasses the SCM**: A's
contribution to the prediction is ~1e-5, `dag.weights ~ 1e-5`, prediction is
`delta ≈ D·Sᵖ`. Diagnosis (`evaluation/dag_grad_diagnostic.py`): `Sᵖ` is a free,
unregularized path while A is shrunk by `lambda_sparse` (mean|A|, λ=10). The
optimizer takes the free path and lets A decay. (`lambda_spectral` is a
relu(σ−0.95) ceiling — inactive at small A, so NOT the shrink force.)

**Hypothesis.** Penalizing the shift (so it can't be the whole answer) + not
shrinking A forces the perturbation to propagate **through A**, reviving the
causal graph — the prerequisite for the RL stage.

**Grid (6 cells; base = exp2/exp6 recipe: shift encoder + linear decoder + MSE+MMD):**

| knob | values | role |
|---|---|---|
| `phase1.pert_shift_l1` | **0.0, 0.1, 0.5** | L1 penalty on `‖Sᵖ‖` (in the pert loss) — forces A to be used |
| `phase1.lambda_sparse` | **10.0, 0.0** | 10 = exp2 (shrinks A) · 0 = let A grow |

`lambda_spectral=0.1` kept (stability ceiling). Control = `exp8_sl0p0_ls10p0`
(≈ exp2); most aggressive A-revival = `exp8_sl0p5_ls0p0`.

**Code vs exp6:** `perturbation_set_loss_shift` gains `shift_l1_weight`
(adds `shift_l1_weight · mean|Sᵖ|`); `train_vae` reads `phase1.pert_shift_l1`;
W&B logs `pert_diag/shift_l1_term`.

**Runs.** Each cell: `run_phase1_experiment.sbatch` → chained held-out
`run_shift_eval.sbatch` (does forcing A hurt prediction?) + chained
`run_dag_grad_diag_generic.sbatch` (did A revive? bypass ratio, dag.weights).
Outputs under `results/grid/<cell>/`.

**Watch (W&B CausalBioRL):** `grad_norm/dag`, `pert_diag/shift_l1_term`,
`pert_diag/mean_shift_norm`, `dag/edge_density`, `loss/pert`, `loss/recon`.
**Success =** held-out AUROC stays ≥ exp2 (~0.70) **and** the A-bypass ratio
rises well above 1e-5 (A actually propagates). If shrinking the shift tanks
AUROC, the linear SCM can't carry propagation → architectural limit.
