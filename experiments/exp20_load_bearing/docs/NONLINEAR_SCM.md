# Phase 2 (staged) — nonlinear structural equations

`z_i = φ(Σ_j A_ji z_j) + ε_i` (saturating φ, e.g. tanh) so combo epistasis flows THROUGH A
instead of the side MLPs (`u_gate`/`eps_mix`), which currently carry non-additivity (why A_gap≈0
even on the fully-nonlinear exp19 runs). Flag `scm_nonlinear: true`.

Invasive: replaces the closed-form linear solve `z=(I−Aᵀ)⁻¹ε` with a damped iterative fixed-point
(implicit-diff or unrolled) at ~15 inlined `(I−MAᵀ)` sites; breaks the `D·A·Dᵀ` GRN readout and D†
shortcuts (fine — we judge by A_gap/prediction). RUN ONLY after Phase-1 shows A_gap>0. Not yet implemented.
