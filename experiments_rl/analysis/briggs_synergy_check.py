"""Are the HSPC pairs still near-additive in a given WM? Reports rho-frac & S_int/S_singles vs a random-pair
null, for HSPC pairs + Joung-TRAINED pairs (in-coverage positive control). Run on any checkpoint to compare.

Uncertainty: every rho-frac is computed over N=64 SAMPLED DAGs (op.sample_dags), so each pair gets a
distribution → we report the median and a [5,95] percentile CI. The random-pair null is a distribution of
64-DAG *medians* over `n_null` random pairs; a pair's percentile is its 64-DAG-median vs that null. S_int/S_sing
is DAG-independent (shift + interaction head, no A), so it's a single value."""
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_stack import add_stack_args, build_stack

HSPC = [('TAL1', 'LMO2'), ('TAL1', 'ETV2'), ('GATA2', 'ETV2'), ('GATA2', 'TAL1'), ('LMO2', 'ETV2'), ('GATA2', 'LMO2')]


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--pool', default='curated')
    ap.add_argument('--n_null', type=int, default=400)
    ap.add_argument('--n_dag', type=int, default=64, help='sampled DAGs per rho-frac (confidence interval)')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, meta, op, data = build_stack(a, dev, n_dag=a.n_dag); data.set_pool(a.pool)
    g2i = data.gene_to_idx; eps0 = data.eps_ctrl_mean.to(dev)
    has_head = model.interaction_U is not None
    # N=64 sampled DAGs (epistemic ensemble; hard-gate Gumbel centered on eval threshold) + per-DAG control decode
    A_s = op.sample_dags(a.n_dag)                                         # [N, d, d]
    ctrl_s = torch.stack([op.decode(data.z0.unsqueeze(0))[0] for _ in range(1)])  # ctrl is A-independent (z0 fixed)
    ctrl = ctrl_s[0]
    print(f"[head present: {has_head}] | N_dag={a.n_dag} sampled DAGs per rho-frac", flush=True)

    @torch.no_grad()
    def rho_frac_dist(a_tf, b_tf):
        """rho-frac = ||Δ_pair − Δ_a − Δ_b|| / ||Δ_pair|| for EACH of the N sampled DAGs → [N]."""
        ai, bi = g2i[a_tf], g2i[b_tf]
        out = np.empty(a.n_dag, np.float32)
        for k in range(a.n_dag):
            Ak = A_s[k]
            Da = op.decode(op.apply_sets(eps0, [[ai]], Ak))[0] - ctrl
            Db = op.decode(op.apply_sets(eps0, [[bi]], Ak))[0] - ctrl
            Dp = op.decode(op.apply_sets(eps0, [[ai, bi]], Ak))[0] - ctrl
            out[k] = float((Dp - Da - Db).norm() / (Dp.norm() + 1e-9))
        return out

    @torch.no_grad()
    def s_ratio(a_tf, b_tf):
        if not has_head:
            return 0.0
        ai, bi = g2i[a_tf], g2i[b_tf]
        gi = torch.as_tensor([ai, bi], device=dev)
        return float(model.compute_interaction(gi).norm() / (op.shift_for([ai, bi]).sum(0).norm() + 1e-9))

    # ── null: distribution of 64-DAG MEDIANS over random pairs ──────────────────
    cand = sorted(set(g2i) & set(data.de_single)); rng = np.random.default_rng(1)
    null_med, null_sr = [], []
    for _ in range(a.n_null):
        i, j = rng.choice(cand, 2, replace=False)
        null_med.append(float(np.median(rho_frac_dist(i, j)))); null_sr.append(s_ratio(i, j))
    null_med, null_sr = np.array(null_med), np.array(null_sr)
    print(f"null (n={a.n_null}, each a {a.n_dag}-DAG median): rho-frac med={np.median(null_med):.3f} "
          f"[{np.percentile(null_med,5):.3f},{np.percentile(null_med,95):.3f}] | S_int/S_sing med={np.median(null_sr):.3f}",
          flush=True)

    # ── Joung TRAINED pairs (in-coverage positive control) ──────────────────────
    jt = [tuple(k) for k in data.de_combo if len([t for t in k if t in g2i]) == 2][:20]
    if jt:
        jmed = [float(np.median(rho_frac_dist(*[t for t in k if t in g2i][:2]))) for k in jt]
        pct = float((null_med[None, :] < np.array(jmed)[:, None]).mean()) * 100
        print(f"Joung-trained pairs (n={len(jmed)}): rho-frac med={np.median(jmed):.3f} (pctile vs null {pct:.0f})", flush=True)

    print(f"\nHSPC pairs (out of combo-coverage — the question) [median (5-95 CI over {a.n_dag} DAGs), pctile vs null]:",
          flush=True)
    for x, y in HSPC:
        if x in g2i and y in g2i:
            d = rho_frac_dist(x, y); med = float(np.median(d)); lo, hi = np.percentile(d, [5, 95])
            sr = s_ratio(x, y)
            pf = float((null_med < med).mean()) * 100; ps = float((null_sr < sr).mean()) * 100
            print(f"  {x}x{y}: rho-frac={med:.3f} [{lo:.3f},{hi:.3f}] (pctile {pf:.0f}) | "
                  f"S_int/S_sing={sr:.3f} (pctile {ps:.0f})", flush=True)
    print("\n[synergy-check done]", flush=True)


if __name__ == '__main__':
    main()
