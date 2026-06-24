"""Perturbation-effect soft recall — credit a planner for picking genes with SIMILAR
effects to the true ones, not only the exact genes.

Motivation: exact gene recall is ~0 because the perturbation→state map is many-to-one
(the planner reaches the target via different but functionally-equivalent drivers). That
makes recall conflate "picked a gene with the same downstream effect" with "picked junk".
This module relaxes recall using the genes' *perturbation-effect profiles* themselves:

  - REFERENCE EFFECTS. Each screened gene g has an OBSERVED differential-expression
    profile e_obs(g) ∈ R^G (data.de_single[g]['lfc']). We cluster the screened genes by
    effect-profile correlation (1−corr distance, average linkage) into K posthoc
    "effect modules" — genes that, when perturbed, move the cell the same way.

  - QUERY EFFECTS. A chosen gene may be unscreened (zero-shot, 8000-pool), so it has no
    observed profile. We embed it by the MODEL's PREDICTED single-gene DE profile
    e_pred(g) = decode(do(ε_ctrl, {g})) − decode(z_ctrl) ∈ R^G — defined for any gene.
    Predicted and observed profiles live in the SAME gene-expression frame, so a chosen
    gene's predicted profile can be compared directly to a true gene's observed profile
    and assigned to the nearest observed-effect cluster centroid.

Two soft-recall measures (both reported, NEITHER penalizes — "no harsh penalty"):
  - soft_recall_corr  (graded): for each true gene, max correlation of its OBSERVED
    profile with any chosen gene's PREDICTED profile, averaged. Continuous in [-1,1];
    exact recovery → 1. This is the "similar perturbation effect" credit.
  - soft_recall_clust (cluster-hit): fraction of true genes whose effect cluster is hit
    by some chosen gene (chosen gene assigned to nearest observed-effect cluster). 1 iff
    every true driver's module is represented in the plan.

`effect_bonus()` exposes the graded measure as a small additive reward term (>=0 after a
relu), so RL/MPC can be *rewarded* for effect-similar picks without ever being penalised.

Dataset-agnostic: works off `data.de_single` + the operator, so the same metric serves
the Norman ladder and (later) the Joung TF-atlas planner.
"""
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering


class EffectClusters:
    def __init__(self, op, data, planners=None, n_clusters=12, ref_genes=None,
                 min_ref=8, verbose=True):
        """op: WorldModelOperator; data: WMData; planners: optional Planners (reuses its
        ctrl_recon / decode path for predicted profiles). n_clusters: posthoc modules.
        ref_genes: gene-name list to cluster (default = all screened singles with DE)."""
        self.op = op
        self.data = data
        self.planners = planners
        self.device = op.device
        self.gene_to_idx = data.gene_to_idx
        self.n_genes = len(data.gene_names)
        # control reconstruction baseline (predicted profiles are deltas from it)
        if planners is not None:
            self.eps0 = planners.eps0
            self.ctrl_recon = planners.ctrl_recon
        else:
            self.eps0 = data.eps_ctrl_mean.unsqueeze(0)
            self.ctrl_recon = op.decode(data.z0.unsqueeze(0)).squeeze(0)

        # --- reference: observed-DE profiles of screened genes -----------------
        if ref_genes is None:
            ref_genes = sorted(data.de_single.keys())
        self.ref_genes = [g for g in ref_genes if g in data.de_single]
        E = np.stack([np.asarray(data.de_single[g]['lfc'], np.float32)
                      for g in self.ref_genes])              # [R, G] observed profiles
        self.ref_idx = np.array([self.gene_to_idx[g] for g in self.ref_genes])
        self.E_obs = E
        self.E_obs_z = self._zscore_rows(E)                  # centered+unit for corr

        # --- posthoc clustering by effect-profile correlation ------------------
        # Ward linkage on the unit-normalised profiles (Euclidean here ∝ √(2(1−corr)))
        # gives BALANCED, correlation-based modules — average-linkage on the raw
        # correlation distance chains into one mega-cluster + singletons.
        K = int(min(n_clusters, max(2, len(self.ref_genes) // min_ref)))
        if len(self.ref_genes) >= 3:
            cl = AgglomerativeClustering(n_clusters=K, linkage='ward')
            self.labels = cl.fit_predict(self.E_obs_z)
        else:
            self.labels = np.zeros(len(self.ref_genes), int)
        self.K = int(self.labels.max()) + 1
        # cluster centroids in observed-effect space (mean profile per module)
        self.centroids_z = np.stack([self.E_obs_z[self.labels == c].mean(0)
                                     for c in range(self.K)])
        self.gene_cluster = {g: int(c) for g, c in zip(self.ref_genes, self.labels)}
        self._pred_cache = {}
        if verbose:
            sizes = np.bincount(self.labels, minlength=self.K)
            print(f"[effect_clusters] {len(self.ref_genes)} screened genes → {self.K} "
                  f"effect modules (sizes {sizes.tolist()})")

    # ---------------------------------------------------------------------------
    @staticmethod
    def _zscore_rows(M):
        M = np.asarray(M, np.float32)
        M = M - M.mean(1, keepdims=True)
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / (n + 1e-8)

    @torch.no_grad()
    def predicted_profile(self, gene_idx):
        """Model-predicted single-gene DE profile (decode(do(ε,{g})) − ctrl_recon)."""
        gi = int(gene_idx)
        if gi not in self._pred_cache:
            zp, _, _ = self.op.apply(self.eps0, [gi], self.data.A_ref)
            x_hat = self.op.decode(zp).mean(0)
            self._pred_cache[gi] = (x_hat - self.ctrl_recon).float().cpu().numpy()
        return self._pred_cache[gi]

    def assign_cluster(self, gene_idx):
        """Assign any gene to the nearest observed-effect module via its predicted
        profile (correlation to module centroids). Screened genes keep their true module."""
        g = self.data.gene_names[int(gene_idx)]
        if g in self.gene_cluster:
            return self.gene_cluster[g]
        p = self._zscore_rows(self.predicted_profile(gene_idx)[None])[0]
        return int(np.argmax(self.centroids_z @ p))

    # --- the two soft-recall measures ------------------------------------------
    def soft_recall(self, chosen_idx, true_idx):
        """Return dict(soft_recall_corr, soft_recall_clust, n_true). chosen_idx/true_idx
        are lists of gene indices. No penalty: both measures are in [-1,1] / [0,1] and
        only credit similarity."""
        true_idx = [int(t) for t in true_idx]
        chosen_idx = [int(c) for c in chosen_idx]
        if not true_idx or not chosen_idx:
            return {'soft_recall_corr': np.nan, 'soft_recall_clust': np.nan,
                    'n_true': len(true_idx)}
        # graded: max corr(pred[chosen], obs[true]) over chosen, per true gene
        chosen_pred = self._zscore_rows(np.stack([self.predicted_profile(c)
                                                  for c in chosen_idx]))      # [Ch, G]
        corrs = []
        chosen_clusters = {self.assign_cluster(c) for c in chosen_idx}
        clust_hits = 0
        for t in true_idx:
            g = self.data.gene_names[t]
            t_obs = (self.E_obs_z[self.ref_genes.index(g)] if g in self.gene_cluster
                     else self._zscore_rows(self.predicted_profile(t)[None])[0])
            corrs.append(float(np.max(chosen_pred @ t_obs)))
            t_clust = self.gene_cluster.get(g, self.assign_cluster(t))
            clust_hits += int(t_clust in chosen_clusters)
        return {'soft_recall_corr': float(np.mean(corrs)),
                'soft_recall_clust': clust_hits / len(true_idx),
                'n_true': len(true_idx)}

    def effect_bonus(self, chosen_idx, true_idx):
        """Non-negative reward bonus = relu(graded soft recall). For optional use as a
        small additive term in MPC/RL reward (never penalises)."""
        r = self.soft_recall(chosen_idx, true_idx)['soft_recall_corr']
        return float(max(0.0, r)) if np.isfinite(r) else 0.0
