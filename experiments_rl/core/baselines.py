"""Planning baselines for the value-add ladder.

Every method here (and MPC, and the PPO policies) is a uniform planner:
    planner(target_key) -> list[int]   # the chosen gene SET to reach z_target
so the ladder random → greedy-1 → additive → MPC → PPO is an apples-to-apples
comparison. The ladder's primary metric is GENE RECOVERY (did the planner find the
true perturbation's genes from its induced state?) + latent distance; gene-space DE
fidelity is scored separately by decoding the do-operator-applied state.

- random:   k random candidate genes (k = target cardinality).
- greedy-1: the single candidate gene whose do-operator state is closest to z_target.
- additive: greedily add candidate single-effect latent deltas (ADDITIVE latent
            composition, NOT the mutilation operator) to best match z_target — the
            order-free additive planner MPC's mutilation planning must beat.

All precomputation is one do-operator pass per candidate gene (cheap).
"""
import numpy as np
import torch


class Planners:
    def __init__(self, op, data):
        self.op = op
        self.data = data
        self.device = op.device
        self.cand = list(data.candidate_idx)             # gene indices (action vocabulary)
        self.A = data.A_ref
        self.eps0_vec = data.eps_ctrl_mean               # [d]
        self.eps0 = data.eps_ctrl_mean.unsqueeze(0)      # [1, d] control representative
        self.z0 = data.z0                                # [d] control latent centroid
        self.ctrl_recon = self.op.decode(self.z0.unsqueeze(0)).squeeze(0)   # [n_genes]
        # per-candidate single-gene do-operator latent delta (z_post − z0), BATCHED so it
        # scales to the full ~8000-gene pool (the per-gene loop would be far too slow).
        S_all = op.shift_for(self.cand)                  # [C, d]
        z_post_all = op.apply_many(self.eps0_vec, S_all, self.A)            # [C, d]
        self.delta_mat = z_post_all - self.z0.unsqueeze(0)                  # [C, d]

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def cardinality(target_key):
        return 2 if isinstance(target_key, tuple) else 1

    def true_genes(self, target_key):
        keys = target_key if isinstance(target_key, tuple) else (target_key,)
        return [self.data.gene_to_idx[g] for g in keys]

    @torch.no_grad()
    def predict_de(self, gene_set):
        """Apply the (validated mutilation) do-operator for a chosen set and decode to
        a gene-space DE prediction (x̂ − control recon). Used to score reached-state
        fidelity uniformly across all planners."""
        zp, _, _ = self.op.apply(self.eps0, list(gene_set), self.A)
        x_hat = self.op.decode(zp).mean(0)
        return (x_hat - self.ctrl_recon).cpu().numpy()

    @torch.no_grad()
    def reached_latent(self, gene_set):
        zp, _, _ = self.op.apply(self.eps0, list(gene_set), self.A)
        return zp.squeeze(0)

    # --- planners --------------------------------------------------------
    def random(self, target_key, rng=None):
        rng = rng or np.random.default_rng()
        k = self.cardinality(target_key)
        return [int(x) for x in rng.choice(self.cand, size=k, replace=False)]

    @torch.no_grad()
    def greedy1(self, target_key):
        zt = self.data.target_for(target_key)
        cand_z = self.z0.unsqueeze(0) + self.delta_mat          # [C, d]
        d = (cand_z - zt.unsqueeze(0)).norm(dim=1)
        return [self.cand[int(d.argmin())]]

    @torch.no_grad()
    def additive(self, target_key):
        """Greedy additive latent composition: repeatedly add the candidate single
        delta that most reduces ‖(z0 + Σδ) − z_target‖. Order-free set, no operator
        epistasis — the additive bar MPC must beat."""
        k = self.cardinality(target_key)
        zt = self.data.target_for(target_key)
        cur = self.z0.clone()
        avail = list(range(len(self.cand)))
        chosen = []
        for _ in range(k):
            cand = cur.unsqueeze(0) + self.delta_mat[avail]      # [|avail|, d]
            d = (cand - zt.unsqueeze(0)).norm(dim=1)
            j = int(d.argmin())
            pick = avail.pop(j)
            chosen.append(pick)
            cur = cur + self.delta_mat[pick]
        return [self.cand[j] for j in chosen]
