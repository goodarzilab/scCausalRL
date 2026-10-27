"""Multi-WM ensemble for CONSERVATIVE cocktail scoring.

Loads K frozen world models (same `jb_prod_wm` config, different split seeds) and scores a cocktail's
fate-reward Φ under each, so a deploy selector or a training reward can use the CROSS-WM conservative
score `min_k Φ^(k)` — robust to which WM you happen to trust. This is the multi-WM analogue of the
single-WM DAG-ensemble pessimism (`−α·σ` over sampled DAGs); it captures a *different* uncertainty
(learned-parameter/split disagreement) that a single WM's DAG samples cannot.

Why no latent alignment is needed: every WM shares the SAME gene-space goal (drift, cent_g, DEG
support), and Φ = signed-Pearson(decode(z)−ctrl_recon−drift, cent_g−drift) is computed in gene space.
So Φ^(k) ∈ [−1,1] are directly comparable across WMs even though each has its own latent A / shift
head / decoder. Only the WM-specific control-latent quantities (eps_ctrl_mean, mean_A, ctrl_recon)
are recomputed per WM, by re-encoding the SHARED control cells `data._ctrl_X` — no h5ad reload.
"""
import torch
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.scoring import torch_match


class WMEnsemble:
    def __init__(self, base_op, data, extra_checkpoints, wm_root, dev, n_dag=8):
        self.dev = dev
        self.members = []                                    # list of dict(op, eps0, A, ctrl_recon)
        self._add(base_op, data)                             # member 0 = the base WM (already loaded)
        for ck in extra_checkpoints:
            model_k, meta_k = load_frozen_wm(checkpoint_path=ck, wm_root=wm_root, device=dev)
            op_k = WorldModelOperator(model_k, meta_k, n_dag_samples=n_dag)
            self._add(op_k, data)
        print(f"[WMEnsemble] loaded {len(self.members)} WMs (base + {len(extra_checkpoints)} seed(s)) "
              f"for cross-WM conservative scoring", flush=True)

    @torch.no_grad()
    def _add(self, op, data):
        _, _, eps = op.model.encoder(data._ctrl_X)           # re-encode shared control cells through THIS WM
        A = op.mean_A()
        z0 = op.scm_forward(eps, A).mean(0)
        self.members.append(dict(op=op, eps0=eps.mean(0), A=A,
                                 ctrl_recon=op.decode(z0.unsqueeze(0))[0]))     # [n_genes]

    @property
    def K(self):
        return len(self.members)

    @torch.no_grad()
    def phi_all(self, gene_set, o_res, sel, drift):
        """Per-WM drift-removed Φ for a cocktail (list[int]) -> tensor [K] in [-1,1]."""
        out = []
        for m in self.members:
            z, _, _ = m['op'].apply(m['eps0'].unsqueeze(0), gene_set, m['A'])   # [1,d]
            pred = m['op'].decode(z) - m['ctrl_recon']                          # [1,n_genes]
            out.append(torch_match(pred - drift, o_res, sel)[0])
        return torch.stack(out)                                                 # [K]

    @torch.no_grad()
    def score(self, gene_set, o_res, sel, drift, lam=0.5):
        """Conservative summaries of Φ over the WM ensemble: mean / WCO(min_k) / UWO(mean−lam·std)."""
        pk = self.phi_all(gene_set, o_res, sel, drift)
        return dict(mean=float(pk.mean()), wco=float(pk.min()),
                    uwo=float(pk.mean() - lam * pk.std()), std=float(pk.std()))
