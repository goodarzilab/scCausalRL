# Recon variants — attacking the shared-program cheat in the loss

The per-cell reconstruction `MSE(x_hat, x)` is the "trivial path": encoder+decoder can fit x
without A, and the target is ~99% shared differentiation program (QC: raw pairwise cos 0.987).
Three knobs (all in `compute_total_loss`, threaded from `phase1`):

- **`pert_residual_target: true`** (pre-existing) — the *perturbation* target becomes `Δ − drift`
  (TF-specific residual). Bias-cancelled, genuinely meaningful. `drift` = mean train-TF Δ.
- **`lambda_recon` (<1.0)** — down-weight the per-cell recon so representational load moves onto the
  SCM/pert path. `total = lambda_recon·l_recon + beta_eff·l_kl`, with `beta_eff = beta·lambda_recon`
  (couples KL to recon so a small recon weight can't let KL crush the encoder). This is the
  meaningful **"not reconstructing cells"** knob (arm `struct_sparse_pertonly`, `lambda_recon=0.1`).
- **`kl_free_bits`** — floor on KL (nats/dim) to guard posterior collapse when `lambda_recon` is small.
- **`residual_recon: true`** — reconstruct `x − drift`. HONEST CAVEAT: near-no-op for a linear
  decoder (the constant drift is absorbed by the decoder bias `b`); only bites when paired with
  `separate_pert_decoder: true` or a frozen decoder bias (arm `struct_sparse_residrecon`).
