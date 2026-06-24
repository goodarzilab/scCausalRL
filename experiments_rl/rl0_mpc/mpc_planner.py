"""M0 — Model-Predictive Planner: batched, large-pool beam search (gene-space objective).

Searches the candidate gene pool (87 screened OR the full ~8000-gene zero-shot axis) for
the set |G|≤k_max that maximizes the gene-space DE-match to a target perturbation, using
the batched do-operator so thousands of candidates are scored in a few GPU calls:

    score(G) = Φ(decode(do(G; A*)))  −  α·std_n Φ(decode(do(G; A^(n)))),   maximize.

Beam search with pruning (the large pool is not exhaustively enumerable —
C(8000,2)≈3.2e7): depth-1 scores all C singles and keeps the top-B; depth-2 expands each
of the B beams against all C partners (B·C pairs) and keeps the best; the B+1 finalists
are then re-ranked by the pessimistic score (ensemble std). `beam=1,k_max=1` ⇒ greedy-1.

Uniform planner: `planner(target_key) -> gene_set`. Reads the active pool from
`data.candidate_idx` (call `data.set_pool('full')` first for the large-pool task).
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.scoring import observed_de, torch_match

CHUNK = 4096


class MPCPlanner:
    def __init__(self, op, data, max_set=2, beam_width=8, n_dag_samples=16,
                 pessimism_alpha=1.0, deg_thresh=0.05):
        self.op = op
        self.data = data
        self.cand = list(data.candidate_idx)
        self.C = len(self.cand)
        self.max_set = int(max_set)
        self.beam = int(beam_width)
        self.N = int(n_dag_samples)
        self.alpha = float(pessimism_alpha)
        self.thresh = float(deg_thresh)
        self.eps0 = data.eps_ctrl_mean
        self.A_eval = op.mean_A()
        self.ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]           # [G]
        self.S_all = op.shift_for(self.cand)                           # [C, d] (cached)

    def _match_batch(self, z_post, o_t, sel_idx):
        """z_post [K,d] -> match [K] (decode in chunks; signed Pearson on DEGs)."""
        outs = []
        for c in z_post.split(CHUNK):
            pred = self.op.decode(c) - self.ctrl_recon                 # [c, G]
            outs.append(torch_match(pred, o_t, sel_idx))
        return torch.cat(outs)

    def _ensemble_std(self, local_set, o_t, sel_idx, A_batch):
        S = self.S_all[list(local_set)].sum(0, keepdim=True)           # [1, d]
        ms = [float(self._match_batch(self.op.apply_many(self.eps0, S, A_batch[n]), o_t, sel_idx)[0])
              for n in range(A_batch.shape[0])]
        return float(np.std(ms))

    @torch.no_grad()
    def plan(self, target_key, A_batch=None, return_info=False):
        lfc, mask = observed_de(self.data, target_key)
        sel = mask & (np.abs(lfc) > self.thresh)
        dev = self.op.device
        sel_idx = torch.tensor(sel.nonzero()[0], dtype=torch.long, device=dev)
        o_t = torch.tensor(lfc[sel], dtype=torch.float32, device=dev)
        if A_batch is None:
            A_batch = self.op.sample_dags(self.N)

        # depth 1: score every candidate single under eval-A
        z1 = self.op.apply_many(self.eps0, self.S_all, self.A_eval)    # [C, d]
        m1 = self._match_batch(z1, o_t, sel_idx)                       # [C]
        B = min(self.beam, self.C)
        top = torch.topk(m1, B).indices.tolist()
        finalists = [([top[0]], float(m1[top[0]]))]                    # best single

        if self.max_set >= 2 and self.C >= 2:
            for bi in top:
                S_pair = self.S_all[bi].unsqueeze(0) + self.S_all      # [C, d]
                z2 = self.op.apply_many(self.eps0, S_pair, self.A_eval)
                m2 = self._match_batch(z2, o_t, sel_idx)
                m2[bi] = -1e9                                          # no self-pair
                j = int(torch.argmax(m2))
                finalists.append(([bi, j], float(m2[j])))

        # pessimistic re-rank of the finalists
        best_local, best_score = None, -1e18
        for local_set, m_eval in finalists:
            score = m_eval - self.alpha * self._ensemble_std(local_set, o_t, sel_idx, A_batch)
            if score > best_score:
                best_score, best_local, best_meval = score, local_set, m_eval
        chosen = [self.cand[i] for i in best_local]
        if return_info:
            return chosen, {'score': best_score, 'match': best_meval, 'n_candidates': self.C}
        return chosen

    @torch.no_grad()
    def top_singles(self, target_key, k=2):
        """The k best INDEPENDENT single genes by gene-space match (a marginal/top-k
        baseline) — does joint beam search (plan) beat picking the k best marginals?"""
        lfc, mask = observed_de(self.data, target_key)
        sel = mask & (np.abs(lfc) > self.thresh)
        dev = self.op.device
        sel_idx = torch.tensor(sel.nonzero()[0], dtype=torch.long, device=dev)
        o_t = torch.tensor(lfc[sel], dtype=torch.float32, device=dev)
        z1 = self.op.apply_many(self.eps0, self.S_all, self.A_eval)
        m1 = self._match_batch(z1, o_t, sel_idx)
        idx = torch.topk(m1, min(k, self.C)).indices.tolist()
        return [self.cand[i] for i in idx]

    def __call__(self, target_key):
        return self.plan(target_key)
