"""Held-out-combo generalization: does the interaction head predict HELD-OUT Joung combos better than
additive?  Architecture-vs-coverage test for why the WM composes HSPC TFs near-additively.

For each combo: observed Δ = data.de_combo[key] (ground truth). Compare (signed-Pearson on the combo's DEGs):
  WM+head   = decode(apply_sets(combo)) - ctrl_recon      (interaction head ON)
  WM-nohead = same with model.interaction_U := None        (additive-in-WM: singles composed by SCM only)
  additive  = Σ de_single[TF]                               (observed single sum)
Split into TRAIN combos (head saw them) vs HELD-OUT (it didn't). If WM+head > additive/WM-nohead on HELD-OUT
=> head generalizes non-additivity => HSPC near-additivity is a COVERAGE problem. If ~equal => ARCHITECTURE limit.
"""
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_stack import add_stack_args, build_stack
from core.scoring import torch_match


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--pool', default='curated')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, meta, op, data = build_stack(a, dev, n_dag=8); data.set_pool(a.pool)
    g2i = data.gene_to_idx; A = op.mean_A(); eps0 = data.eps_ctrl_mean.to(dev)
    ctrl = op.decode(data.z0.unsqueeze(0))[0].detach()
    thr = data.thresh

    # held-out / train combo split from the checkpoint config (falls back to hash split seed1 frac0.5)
    import torch as _t
    sd = _t.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = sd.get('config', {}) if isinstance(sd, dict) else {}
    p1 = cfg.get('phase1', {})
    test_keys = {tuple(sorted(k)) for k in p1.get('combo_test_keys', [])}
    train_keys = {tuple(sorted(k)) for k in p1.get('combo_train_keys', [])}
    print(f"[split from ckpt] train combos={len(train_keys)} | held-out={len(test_keys)}", flush=True)
    has_head = model.interaction_U is not None
    print(f"[head] interaction head present: {has_head}", flush=True)

    def preds(key):
        gi = [g2i[t] for t in key if t in g2i]
        if len(gi) < 2:
            return None
        obs = np.asarray(data.de_combo[key]['lfc'], np.float32)
        sel = torch.tensor(np.where(np.abs(obs) > thr)[0], dtype=torch.long, device=dev)
        if len(sel) < 3:
            return None
        ot = torch.tensor(obs, device=dev)
        with torch.no_grad():
            wm = op.decode(op.apply_sets(eps0, [gi], A))[0] - ctrl
            savedU = model.interaction_U
            model.interaction_U = None
            wmnh = op.decode(op.apply_sets(eps0, [gi], A))[0] - ctrl
            model.interaction_U = savedU
        add = torch.tensor(np.sum([np.asarray(data.de_single[t]['lfc'], np.float32) for t in key if t in data.de_single], 0), device=dev)
        m = lambda p: float(torch_match(p.unsqueeze(0), ot[sel], sel)[0])
        return m(wm), m(wmnh), m(add)

    for split, keys in [('HELD-OUT', test_keys), ('TRAIN', train_keys)]:
        rows = [preds(k) for k in keys if k in data.de_combo]
        rows = [r for r in rows if r]
        if not rows:
            print(f"[{split}] no evaluable combos", flush=True); continue
        wm, wmnh, add = np.array(rows).T
        print(f"[{split}] n={len(rows)} | match-to-observed: WM+head {wm.mean():+.3f} | WM-nohead {wmnh.mean():+.3f} | "
              f"additive {add.mean():+.3f} || head-vs-additive Δ={wm.mean()-add.mean():+.3f} | "
              f"head-vs-nohead Δ={wm.mean()-wmnh.mean():+.3f}", flush=True)
    print("\n[combo-generalize done]", flush=True)


if __name__ == '__main__':
    main()
