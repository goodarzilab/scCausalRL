"""Which component is load-bearing?  If the latent DAG A is inert (grn_dag_sweep showed
real≈zero≈shuf), then the prediction Δ = decode(apply_shift(ε, S, A)) − ctrl_recon must be
carried by the SHIFT ENCODER (per-perturbation latent shift S) and/or the DECODER D
(latent→gene map). This script ablates each component and measures the drop in the
drift-removed eval metrics, so we can attribute load-bearing quantitatively.

Ablations (per WM, on held-out single perts; combos too):
  baseline   : real S, real A, real D
  A_zero     : A := 0                      (SCM solve becomes identity)
  A_shuf     : A columns permuted
  shift_perm : S reassigned across perts   (kills perturbation IDENTITY, keeps a real shift)
  shift_rand : S := Gaussian, matched ‖·‖  (kills all shift structure)
  shift_zero : S := 0                       (predict control → floor)
  D_perm     : D rows (genes) permuted      (kills latent→gene mapping; ctrl_recon recomputed)
  D_rand     : D := random orthonormal, matched ‖·‖_F

Verdict: the component whose ablation drops the metric MOST is the load-bearing one.
  • disc (retrieval)  ~ perturbation specificity  → expected to die under shift_perm/rand
  • spear/pear (Δ)    ~ gene-space structure       → expected to die under D_perm/rand

Run:  python analysis/component_ablation.py --grid {grnfix|sena} --device cpu
"""
import os, sys, argparse, gc
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from core.setup_stack import build_stack
from analysis.grn_dag_sweep import predict, wm_metrics, CHUNK, GRNFIX_WMS, SENA_WMS, EPSD_WMS, X20_WMS, X20PO_WMS, _A, EXP19, EXP20


def build_eval(op, data):
    """Test-set shift lists + observed Δ + drift + ctrl_recon (mirror grn_dag_sweep)."""
    A = op.mean_A(); ctrl = op.decode(data.z0.unsqueeze(0))[0]
    drift = np.mean([data.de_single[g]['lfc'] for g in data.single_split['train']], 0).astype(np.float64)
    sg = [g for g in data.single_split['test'] if g in data.de_single]
    Ss = [op.shift_for([data.gene_to_idx[g]]).squeeze(0) for g in sg]
    Os = np.stack([data.de_single[g]['lfc'] for g in sg]).astype(np.float64)
    ck = list(data.combo_split['test']) + list(data.combo_split['train'])
    Sc = [op.shift_for([data.gene_to_idx[x] for x in k]).sum(0) for k in ck]
    Oc = np.stack([data.de_combo[k]['lfc'] for k in ck]).astype(np.float64) if ck else None
    return A, ctrl, drift, Ss, Os, Sc, (Oc if ck else None)


def perm_shift(S, rng):
    p = rng.permutation(len(S))
    while len(S) > 1 and (p == np.arange(len(S))).all():
        p = rng.permutation(len(S))
    return [S[i] for i in p]


def rand_shift(S, g):
    out = []
    for s in S:
        r = torch.randn(s.shape, generator=g, device=s.device, dtype=s.dtype)
        out.append(r / (r.norm() + 1e-9) * s.norm())              # matched magnitude
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='sena', choices=['grnfix', 'sena', 'epsd', 'x20', 'x20po'])
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_grad_enabled(False)
    wm_list = {'grnfix': GRNFIX_WMS, 'sena': SENA_WMS, 'epsd': EPSD_WMS, 'x20': X20_WMS, 'x20po': X20PO_WMS}[a.grid]
    ROOT = EXP20 if a.grid in ('x20', 'x20po') else EXP19
    rng = np.random.default_rng(0); g = torch.Generator(device='cpu').manual_seed(0)
    rows = []
    for tag in wm_list:
        ckpt = f'{ROOT}/results/grid/{tag}/checkpoints/phase1_epoch60.pt'
        if not os.path.exists(ckpt):
            print(f"[skip] {tag}"); continue
        args = _A(); args.checkpoint = ckpt; args.device = dev; args.wm_root = ROOT
        model, meta, op, data = build_stack(args, dev, 8); data.set_pool('curated')
        A, ctrl, drift, Ss, Os, Sc, Oc = build_eval(op, data)
        d = A.shape[0]

        def M(S, Av, cr, O):                                       # metric tuple for a config
            return wm_metrics(predict(op, data, S, Av, cr), O, drift, data.thresh)

        # ---- decoder ablation needs D swapped + ctrl_recon recomputed ----
        def with_D(Dnew, S, Av, O):
            Dorig = model.decoder.D.data.clone()
            model.decoder.D.data = Dnew.to(Dorig.dtype)
            cr = op.decode(data.z0.unsqueeze(0))[0]
            r = M(S, Av, cr, O)
            model.decoder.D.data = Dorig
            return r
        Dorig = model.decoder.D.data
        ng = Dorig.shape[0]; fro = Dorig.norm()
        D_perm = Dorig[torch.as_tensor(rng.permutation(ng))]                      # permute genes
        Q, _ = torch.linalg.qr(torch.randn(ng, d, generator=g))
        D_rand = Q / (Q.norm() + 1e-9) * fro                                       # matched Frobenius

        A0 = torch.zeros_like(A); Ashuf = A[:, torch.as_tensor(rng.permutation(d), device=dev)]
        for split, S, O in [('S', Ss, Os), ('C', Sc, Oc)]:
            if O is None or len(S) == 0:
                continue
            base = M(S, A, ctrl, O)
            abl = {
                'A_zero':     M(S, A0, ctrl, O),
                'A_shuf':     M(S, Ashuf, ctrl, O),
                'shift_perm': M(perm_shift(S, rng), A, ctrl, O),
                'shift_rand': M(rand_shift(S, g), A, ctrl, O),
                'shift_zero': M([torch.zeros_like(s) for s in S], A, ctrl, O),
                'D_perm':     with_D(D_perm, S, A, O),
                'D_rand':     with_D(D_rand, S, A, O),
            }
            row = {'WM': tag.replace('grnfix_', ''), 'split': split, 'n': len(S),
                   'disc': base[0], 'spear': base[1], 'pear': base[2]}
            for name, (dc, sp, pe) in abl.items():
                row[f'{name}.disc'] = dc; row[f'{name}.spear'] = sp; row[f'{name}.pear'] = pe
            rows.append(row)
            print(f"  [{tag} {split}] base disc/spear/pear = {base[0]:.3f}/{base[1]:.3f}/{base[2]:.3f} "
                  f"| Δspear: shiftperm={base[1]-abl['shift_perm'][1]:+.3f} "
                  f"shiftrand={base[1]-abl['shift_rand'][1]:+.3f} "
                  f"Dperm={base[1]-abl['D_perm'][1]:+.3f} Azero={base[1]-abl['A_zero'][1]:+.3f}", flush=True)
        del model, op, data; gc.collect()
        if dev.type == 'cuda': torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = f'{HERE}/component_ablation_{a.grid}.csv'; df.to_csv(out, index=False)
    pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
    print("\n===== COMPONENT ABLATION (drop from baseline; larger drop = more load-bearing) =====")
    for split in ('S', 'C'):
        sub = df[df.split == split]
        if not len(sub): continue
        print(f"\n--- {'SINGLES' if split=='S' else 'COMBOS'}: Δspear (base − ablation) ---")
        t = pd.DataFrame({'WM': sub.WM})
        for c in ['A_zero', 'A_shuf', 'shift_perm', 'shift_rand', 'shift_zero', 'D_perm', 'D_rand']:
            t[c] = (sub['spear'].values - sub[f'{c}.spear'].values)
        print(t.round(3).to_string(index=False))
        print(f"--- {'SINGLES' if split=='S' else 'COMBOS'}: Δdisc (base − ablation) ---")
        t = pd.DataFrame({'WM': sub.WM})
        for c in ['shift_perm', 'shift_rand', 'D_perm', 'D_rand', 'A_zero']:
            t[c] = (sub['disc'].values - sub[f'{c}.disc'].values)
        print(t.round(3).to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == '__main__':
    main()
