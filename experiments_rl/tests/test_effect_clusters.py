"""Fast synthetic unit test for EffectClusters soft-recall (no WM load).

Mocks the minimal op/data interface and checks:
  - construction clusters genes by observed-DE correlation,
  - exact recovery → soft_recall_corr≈1, soft_recall_clust==1,
  - an effect-similar (same-module) pick scores high; a dissimilar pick scores low,
  - effect_bonus is non-negative.
Run:  python tests/test_effect_clusters.py
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.effect_clusters import EffectClusters

G = 30                                   # genes (expression dim)
rng = np.random.default_rng(0)

# Two clear effect modules: genes 0-3 share profile P0, genes 4-7 share profile P1.
P0 = rng.normal(size=G); P1 = rng.normal(size=G)
profiles = {}
names = []
for i in range(4):
    profiles[f'A{i}'] = P0 + 0.05 * rng.normal(size=G); names.append(f'A{i}')
for i in range(4):
    profiles[f'B{i}'] = P1 + 0.05 * rng.normal(size=G); names.append(f'B{i}')
# plus some unscreened pool genes (not in de_single) that the planner could pick
pool = [f'X{i}' for i in range(10)]
all_names = names + pool
gene_to_idx = {g: i for i, g in enumerate(all_names)}


class FakeOp:
    device = 'cpu'
    def apply(self, eps0, gene_list, A):
        # predicted profile of gene g = its true module profile (so prediction is faithful)
        gi = gene_list[0]; g = all_names[gi]
        if g in profiles:
            z = profiles[g]
        elif g.startswith('X') and int(g[1:]) % 2 == 0:
            z = P0 + 0.05 * rng.normal(size=G)        # even pool genes ~ module A
        else:
            z = P1 + 0.05 * rng.normal(size=G)        # odd pool genes ~ module B
        return torch.tensor(z, dtype=torch.float32).unsqueeze(0), None, None
    def decode(self, z):
        return z                                      # mock: latent space == gene space


class FakeData:
    def __init__(self):
        self.gene_names = np.array(all_names)
        self.gene_to_idx = gene_to_idx
        self.de_single = {g: {'lfc': profiles[g].astype(np.float32)} for g in names}
        self.A_ref = None
        self.z0 = torch.zeros(G)                      # mock: latent space == gene space
        self.eps_ctrl_mean = torch.zeros(G)


op, data = FakeOp(), FakeData()
ec = EffectClusters(op, data, planners=None, n_clusters=2, min_ref=1)
assert ec.K == 2, f'expected 2 modules, got {ec.K}'
# the two true modules must be separated
ca = {ec.gene_cluster[g] for g in ['A0','A1','A2','A3']}
cb = {ec.gene_cluster[g] for g in ['B0','B1','B2','B3']}
assert len(ca) == 1 and len(cb) == 1 and ca != cb, f'modules not separated: {ca} {cb}'
print(f'[ok] {len(names)} screened genes → {ec.K} modules, A-module={ca}, B-module={cb}')

idx = gene_to_idx
# exact recovery
r = ec.soft_recall([idx['A0'], idx['A1']], [idx['A0'], idx['A1']])
assert r['soft_recall_corr'] > 0.95 and r['soft_recall_clust'] == 1.0, r
print(f"[ok] exact recovery: corr={r['soft_recall_corr']:.3f} clust={r['soft_recall_clust']:.2f}")

# same-module but DIFFERENT genes (effect-similar) → high corr + cluster hit
r = ec.soft_recall([idx['A2'], idx['A3']], [idx['A0']])
assert r['soft_recall_corr'] > 0.9 and r['soft_recall_clust'] == 1.0, r
print(f"[ok] same-module diff-gene: corr={r['soft_recall_corr']:.3f} clust={r['soft_recall_clust']:.2f}")

# wrong module (dissimilar effect) → low corr, cluster miss
r = ec.soft_recall([idx['B0'], idx['B1']], [idx['A0']])
assert r['soft_recall_corr'] < 0.5 and r['soft_recall_clust'] == 0.0, r
print(f"[ok] wrong-module: corr={r['soft_recall_corr']:.3f} clust={r['soft_recall_clust']:.2f}")

# zero-shot pool gene in module A (even X) credited toward an A true gene
r = ec.soft_recall([idx['X0']], [idx['A0']])           # X0 ~ module A
assert r['soft_recall_clust'] == 1.0, r
print(f"[ok] zero-shot pool pick assigned to right module: clust={r['soft_recall_clust']:.2f}")

# effect_bonus non-negative
b = ec.effect_bonus([idx['B0']], [idx['A0']])
assert b >= 0.0, b
print(f'[ok] effect_bonus non-negative ({b:.3f})')
print('ALL PASS')
