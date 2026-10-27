"""Does the (current best) frozen WM predict HELD-OUT Joung TRIPLES?  Training uses PAIRS only
(load_joung keeps len==2); triples are genuinely zero-shot. For each usable triple (a,b,c):
  observed Δ = data.de_combo[key] (ground truth, DEGs = |Δ|>thresh)
  WM+head    = decode(apply_sets([a,b,c])) - ctrl_recon      (interaction head ON, K=3)
  WM-nohead  = same with interaction_U := None                (additive-in-WM: SCM composes singles)
  additive   = Σ de_single[TF]                                (observed single sum)
  shuffled-A = WM+head with a SHUFFLED A null (correctness-gain reference)
Reports signed-Pearson-on-DEGs (torch_match) per triple + mean. If WM+head >= additive on triples,
the head's 2nd-order composition generalizes to 3-way sets it never trained on.
"""
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_stack import add_stack_args, build_stack
from core.scoring import torch_match


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--pool', default='curated')
    ap.add_argument('--min_real', type=int, default=2, help='min on-axis TFs in the triple key')
    ap.add_argument('--n_dag', type=int, default=64, help='sampled DAGs per match (confidence interval)')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, meta, op, data = build_stack(a, dev, n_dag=a.n_dag); data.set_pool(a.pool)
    g2i = data.gene_to_idx; eps0 = data.eps_ctrl_mean.to(dev)
    ctrl = op.decode(data.z0.unsqueeze(0))[0].detach()
    thr = data.thresh
    has_head = model.interaction_U is not None
    triples = list(data.combo_split.get('triples', []))
    # N sampled DAGs → each match becomes a distribution → median + [5,95] CI (WM-nohead/additive are the
    # per-DAG references; additive is A-free so constant across DAGs). shuffled-A: one permuted-A reference.
    A_s = op.sample_dags(a.n_dag)                                         # [N, d, d]
    rng = np.random.default_rng(0); perm = torch.tensor(rng.permutation(A_s.shape[1]), device=dev)
    print(f"[head] interaction head present: {has_head} | usable Joung triples = {len(triples)} | "
          f"N_dag={a.n_dag}", flush=True)
    print(f"[triples] {triples}", flush=True)

    def preds(key):
        gi = [g2i[t] for t in key if t in g2i]
        if len(gi) < a.min_real:
            return None
        obs = np.asarray(data.de_combo[key]['lfc'], np.float32)
        sel = torch.tensor(np.where(np.abs(obs) > thr)[0], dtype=torch.long, device=dev)
        if len(sel) < 3:
            return None
        ot = torch.tensor(obs, device=dev)
        m = lambda p: float(torch_match(p.unsqueeze(0), ot[sel], sel)[0])
        wm_d, wmnh_d, sh_d = [], [], []
        with torch.no_grad():
            for k in range(a.n_dag):
                Ak = A_s[k]; ck = op.decode(data.z0.unsqueeze(0))[0]
                wm_d.append(m(op.decode(op.apply_sets(eps0, [gi], Ak))[0] - ck))
                sh_d.append(m(op.decode(op.apply_sets(eps0, [gi], Ak[perm][:, perm]))[0] - ck))
                savedU = model.interaction_U; model.interaction_U = None
                wmnh_d.append(m(op.decode(op.apply_sets(eps0, [gi], Ak))[0] - ck))
                model.interaction_U = savedU
        add = torch.tensor(np.sum([np.asarray(data.de_single[t]['lfc'], np.float32)
                                   for t in key if t in data.de_single], 0), device=dev)
        return key, len(gi), len(sel), np.array(wm_d), np.array(wmnh_d), m(add), np.array(sh_d)

    rows = [r for r in (preds(k) for k in triples) if r]
    if not rows:
        print("[triples] no evaluable triples (need >=2 on-axis TFs and >=3 DEGs)", flush=True); return
    print(f"\n{'triple':<30}{'k':>2}{'nDEG':>6}{'WM+head[5-95]':>22}{'WM-noh':>8}{'add':>7}{'shufA':>8}", flush=True)
    for key, k, nd, wm, wmnh, add, wsh in rows:
        lo, hi = np.percentile(wm, [5, 95])
        print(f"{'x'.join(key):<30}{k:>2}{nd:>6}   {np.median(wm):+.3f}[{lo:+.3f},{hi:+.3f}]"
              f"{np.median(wmnh):>8.3f}{add:>7.3f}{np.median(wsh):>8.3f}", flush=True)
    wm = np.mean([np.median(r[3]) for r in rows]); wmnh = np.mean([np.median(r[4]) for r in rows])
    add = np.mean([r[5] for r in rows]); wsh = np.mean([np.median(r[6]) for r in rows])
    print(f"\n[MEAN over {len(rows)} triples, {a.n_dag}-DAG medians] WM+head {wm:+.3f} | WM-nohead {wmnh:+.3f} | "
          f"additive {add:+.3f} | shuffled-A {wsh:+.3f}", flush=True)
    print(f"[gains] head-vs-additive Δ={wm-add:+.3f} | head-vs-nohead Δ={wm-wmnh:+.3f} | "
          f"correctness-gain (WM+head − shuffA) Δ={wm-wsh:+.3f}", flush=True)
    print("\n[joung-triple-eval done]", flush=True)


if __name__ == '__main__':
    main()
