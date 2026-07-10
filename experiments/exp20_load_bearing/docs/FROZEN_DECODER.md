# Phase 2 (staged) — frozen GO-decoder

Init D from `build_go_bp_mask` (gene×pathway binary, K≈886), `requires_grad=False`, and
`lambda_ortho=lambda_D_sparse=0`. Latents = named GO:BP programs; a frozen D cannot morph to absorb
the shared program, breaking the encoder/decoder rotational symmetry that lets A be absorbed.
Flags `decoder_type: frozen_go`, `latent_dim: K`. Larger scale (A is K×K → O(K³) solves).
RUN ONLY after Phase-1 shows A_gap>0. Not yet implemented.
