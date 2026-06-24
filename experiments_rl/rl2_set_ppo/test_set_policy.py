"""Tests for the set-based autoregressive policy (M2) — no Norman load needed.

Checks the masking + factorization invariants on a tiny random policy:
  - STOP never sampled at slot 0 (no empty action);
  - no duplicate genes within a sampled set;
  - evaluate(sampled tokens) reproduces the sampled summed log-prob (consistency);
  - log π is a sum of ≤0 per-slot log-softmaxes; entropy ≥ 0;
  - STOP yields variable cardinality (singles and doubles both reachable).
"""
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl2_set_ppo.set_policy import SetPolicy

_FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILS.append(name)


def main():
    torch.manual_seed(0)
    d, C, k = 8, 12, 2
    pol = SetPolicy(state_dim=d, n_candidates=C, k_max=k, emb_dim=8, hidden=32, slot_dim=4)

    no_stop0, no_dup, consistent, card = True, True, True, set()
    max_abs_err = 0.0
    for _ in range(200):
        s = torch.randn(d); t = torch.randn(d)
        tokens, genes, logp, ent = pol.sample(s, t)
        if tokens[0] == 0:
            no_stop0 = False
        if len(genes) != len(set(genes)):
            no_dup = False
        card.add(len(genes))
        # evaluate must reproduce the sampled summed log-prob
        lp2, ent2 = pol.evaluate(s.unsqueeze(0), t.unsqueeze(0), torch.tensor([tokens]))
        max_abs_err = max(max_abs_err, abs(float(lp2[0]) - logp))
        if logp > 1e-5 or ent < -1e-6:   # log-prob ≤ 0, entropy ≥ 0
            consistent = False

    check("no STOP at slot 0", no_stop0)
    check("no duplicate genes in a set", no_dup)
    check("evaluate reproduces sample log-prob", max_abs_err < 1e-4, f"max|Δlogp|={max_abs_err:.2e}")
    check("log π ≤ 0 and entropy ≥ 0", consistent)
    check("variable cardinality via STOP (singles & doubles)", card == {1, 2} or card.issuperset({1, 2}),
          f"cardinalities seen={sorted(card)}")

    print(f"\n{'ALL SET-POLICY TESTS PASSED' if not _FAILS else 'FAILURES: ' + ', '.join(_FAILS)}")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
