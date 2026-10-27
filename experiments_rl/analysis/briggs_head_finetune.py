"""STAGE A — fine-tune ONLY the interaction head (interaction_U / interaction_out) on combo Δ, to CAUSALLY
test the #5 coverage law: does adding hematopoietic-pair coverage lift held-out HSPC-pair non-additivity?

  cov0  = Joung combos only (control — Joung has NO hematopoietic pairs, so HSPC should stay at chance).
  cov+  = Joung + Briggs combo Δ (--briggs_npz), the added-modality fine-tune.

Everything frozen except interaction_U/out. Grad-enabled do-op = model.intervene_shift_mutilate (the exact
twin of op.apply_sets). Loss = MSE(decode(do(pair)) - ctrl_recon, observed combo Δ). Held-out eval pairs
(NEVER trained): the canonical epistasis pairs + P1/P2 pairs. Eval = ρ-frac & S_int/S_singles coverage
dose-response (in-coverage / out-of-coverage / held-out-HSPC), before vs after. Logs to wandb.
"""
import os, sys, argparse, itertools
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_stack import add_stack_args, build_stack
from core.wblog import wb_init, wb_log, wb_summary, wb_finish

EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
P1 = ['TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2']
P2 = ['DDIT3', 'MZF1', 'HOXB3', 'MIER1', 'SIRT6', 'ZNF85']
# held-out eval pairs (never trained on): canonical hematopoietic epistasis + within-P1 / within-P2 pairs
HSPC_EVAL_PAIRS = [('TAL1', 'LMO2'), ('TAL1', 'ETV2'), ('GATA2', 'ETV2'),
                   ('GATA2', 'TAL1'), ('LMO2', 'ETV2'), ('GATA2', 'LMO2')]


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--pool', default='curated')
    ap.add_argument('--briggs_npz', default=None, help='cov+: npz with obs_delta[n,G] + pairs[n,2] gene-symbol combo Δ')
    ap.add_argument('--heldout_npz', default='/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs/briggs_combo_delta.npz',
                    help='source of OBSERVED Δ for the held-out HSPC test pairs (always loaded; NOT added to training)')
    ap.add_argument('--n_briggs_pairs', type=int, default=0, help='dose-response: subsample added Briggs pairs to N (0=all)')
    ap.add_argument('--heldout_frac', type=float, default=0.2, help='random fraction of Briggs pairs held out as a large-n test set')
    ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n_ctrl_ft', type=int, default=128); ap.add_argument('--l2_sint', type=float, default=1e-3)
    ap.add_argument('--n_null', type=int, default=300)
    ap.add_argument('--wandb', action='store_true'); ap.add_argument('--run_name', default=None)
    ap.add_argument('--out', default=None, help='save fine-tuned head weights (.pt)')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, meta, op, data = build_stack(a, dev, n_dag=8); data.set_pool(a.pool)
    g2i = data.gene_to_idx; gene_names = list(map(str, data.gene_names))
    assert model.interaction_U is not None, "WM has no interaction head"

    A = op.mean_A().detach()
    eps_all = data.eps_ctrl
    sub = eps_all[torch.randperm(eps_all.shape[0])[:a.n_ctrl_ft]].to(dev)
    z_prev = op.scm_forward(sub, A).detach()
    ctrl_recon = op.decode(data.z0.unsqueeze(0))[0].detach()
    umode = getattr(op, 'u_mode', 'shift'); sbeta = float(getattr(op, 'shift_beta', 1.0))

    def do_delta(gi, grad=False):
        """decode(do(gene-idx set)) - ctrl_recon  [G], grad-enabled do-op (interaction head active)."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            zp, _ = model.intervene_shift_mutilate(sub, gi, A, z_prev, u_mode=umode, shift_beta=sbeta)
            return model.decoder(zp).mean(0) - ctrl_recon

    # ---- training targets ----
    heldout = {frozenset(p) for p in HSPC_EVAL_PAIRS}
    heldout |= {frozenset((P1[i], P1[j])) for i in range(len(P1)) for j in range(i + 1, len(P1))}
    heldout |= {frozenset((P2[i], P2[j])) for i in range(len(P2)) for j in range(i + 1, len(P2))}
    train = []                                                          # (gene_idx_list, obs_delta_tensor, source, gene_mask|None)
    for key, de in data.de_combo.items():                              # cov0: Joung combos (full 4366-gene target)
        tfs = [t for t in key if t in g2i]
        if len(tfs) < 2 or frozenset(tfs) in heldout:
            continue
        train.append(([g2i[t] for t in tfs], torch.tensor(np.asarray(de['lfc'], np.float32), device=dev), 'joung', None))
    n_briggs = 0; briggs_test = []
    npz_path = a.heldout_npz if (a.heldout_npz and os.path.exists(a.heldout_npz)) else a.briggs_npz
    if npz_path and os.path.exists(npz_path):
        bz = np.load(npz_path, allow_pickle=True)
        # Briggs Δ is defined only on genes SHARED with the WM -> mask the loss to those.
        smask = torch.as_tensor(np.asarray(bz['shared_idx']), dtype=torch.long, device=dev) if 'shared_idx' in bz else None
        allb = [(od, pr) for od, pr in zip(bz['obs_delta'], bz['pairs'])
                if len([t for t in pr if str(t) in g2i]) >= 2 and frozenset(str(t) for t in pr) not in heldout]
        # FIXED random held-out test set (identical across dose-response runs); pool = candidates for training
        perm = np.random.default_rng(12345).permutation(len(allb))
        n_test = int(len(allb) * a.heldout_frac)
        briggs_test = [(allb[i][0], list(allb[i][1]), smask) for i in perm[:n_test]]
        pool = [allb[i] for i in perm[n_test:]]
        add = len(pool) if a.n_briggs_pairs < 0 else min(a.n_briggs_pairs, len(pool))   # 0=none(cov0), -1=all, N=N
        if 0 < add < len(pool):
            pool = [pool[i] for i in np.random.default_rng(0).choice(len(pool), add, replace=False)]
        elif add == 0:
            pool = []
        for od, pr in pool:
            tfs = [str(t) for t in pr if str(t) in g2i]
            train.append(([g2i[t] for t in tfs], torch.tensor(np.asarray(od, np.float32), device=dev), 'briggs', smask))
            n_briggs += 1
    print(f"[train] {len(train)} combo sentences (joung {sum(t[2]=='joung' for t in train)}, briggs {n_briggs}) "
          f"| held-out eval pairs excluded: {len(heldout)}", flush=True)

    # ---- coverage dose-response eval (ρ-frac & S_int/S_singles), grad-off ----
    cand_tfs = sorted(set(g2i) & set(data.de_single))

    def pair_metrics(a_tf, b_tf):
        ai, bi = g2i[a_tf], g2i[b_tf]
        Da, Db, Dp = do_delta([ai]), do_delta([bi]), do_delta([ai, bi])
        frac = float((Dp - Da - Db).norm() / (Dp.norm() + 1e-9))
        gi = torch.as_tensor([ai, bi], device=dev)
        sr = float(model.compute_interaction(gi).norm() / (op.shift_for([ai, bi]).sum(0).norm() + 1e-9))
        return frac, sr

    def coverage_eval(tag):
        groups = {}
        groups['in-coverage'] = [pair_metrics(gene_names[s[0][0]], gene_names[s[0][1]]) for s in train if len(s[0]) == 2][:60]
        rng = np.random.default_rng(3); oc = []
        while len(oc) < a.n_null:
            i, j = rng.choice(cand_tfs, 2, replace=False)
            oc.append(pair_metrics(i, j))
        groups['out-of-coverage'] = oc
        groups['held-out-HSPC'] = [pair_metrics(x, y) for x, y in HSPC_EVAL_PAIRS if x in g2i and y in g2i]
        oc_rf = np.array([v[0] for v in groups['out-of-coverage']])
        out = {}
        print(f"\n[{tag}] coverage dose-response (ρ-frac, S_int/S_singles):", flush=True)
        for g, vals in groups.items():
            rf = np.array([v[0] for v in vals]); sr = np.array([v[1] for v in vals])
            pct = float((oc_rf[None, :] < rf[:, None]).mean()) * 100
            out[g] = (float(np.median(rf)), float(np.median(sr)), pct)
            print(f"  {g:16s} n={len(vals):3d}  ρ-frac med={np.median(rf):.3f}  S_int/S_sing med={np.median(sr):.3f}  "
                  f"ρ-frac pctile-vs-out={pct:.0f}", flush=True)
        return out

    # ---- held-out-pair PREDICTION ACCURACY (the decisive, magnitude-free test): does the head predict
    # the OBSERVED Δ of held-out HSPC pairs better after coverage? corr(pred Δ, observed Δ) on shared genes. ----
    # groups: 'HSPC' = the 6 canonical pairs (special-interest, small-n); 'random-briggs' = the fixed 20% held-out
    # Briggs pairs (large-n, statistically robust). Both NEVER in training.
    heldout_groups = {'HSPC': [], 'random-briggs': []}
    if npz_path and os.path.exists(npz_path):
        hz = np.load(npz_path, allow_pickle=True)
        hmask = np.asarray(hz['shared_idx']) if 'shared_idx' in hz else np.arange(len(gene_names))
        obs_by_pair = {frozenset(str(t) for t in pr): od for od, pr in zip(hz['obs_delta'], hz['pairs'])}
        for x, y in HSPC_EVAL_PAIRS:
            if x in g2i and y in g2i and frozenset((x, y)) in obs_by_pair:
                heldout_groups['HSPC'].append(([g2i[x], g2i[y]], np.asarray(obs_by_pair[frozenset((x, y))], np.float32), hmask))
    for od, pr, _ in briggs_test:
        tfs = [str(t) for t in pr if str(t) in g2i]
        if len(tfs) >= 2:
            heldout_groups['random-briggs'].append(([g2i[tfs[0]], g2i[tfs[1]]], np.asarray(od, np.float32), hmask))
    print(f"[heldout-test] HSPC pairs n={len(heldout_groups['HSPC'])} | random-briggs n={len(heldout_groups['random-briggs'])}", flush=True)

    def heldout_pred_acc(tag):
        out = {}
        for g, items in heldout_groups.items():
            if not items:
                out[g] = None; continue
            cs = [float(np.corrcoef(do_delta(gi).cpu().numpy()[m], obs[m])[0, 1]) for gi, obs, m in items]
            out[g] = float(np.nanmean(cs))
        print(f"[{tag}] held-out PREDICTION corr: " + " | ".join(f"{g} n={len(heldout_groups[g])} mean={out[g]:+.3f}"
              for g in out if out[g] is not None), flush=True)
        return out
        return mc

    wb = wb_init(a.run_name or ('headft_cov' + ('plus' if n_briggs else '0')), vars(a),
                 tags=['briggs', 'head-finetune', 'coverage']) if a.wandb else None
    pre_acc = heldout_pred_acc('PRE-finetune')
    pre = coverage_eval('PRE-finetune')
    if wb is not None:
        wb_summary(wb, {f'pre/{g}/rho_frac_pctile': pre[g][2] for g in pre})

    # ---- fine-tune the head only ----
    for p in model.parameters():
        p.requires_grad_(False)
    head_params = list(model.interaction_U.parameters()) + list(model.interaction_out.parameters())
    for p in head_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(head_params, lr=a.lr)
    rng = np.random.default_rng(0)
    for ep in range(1, a.epochs + 1):
        order = rng.permutation(len(train))
        tot = 0.0
        for idx in order:
            gi, obs, _, gmask = train[idx]
            pred = do_delta(gi, grad=True)
            gi_t = torch.as_tensor(gi, device=dev)
            sint = model.compute_interaction(gi_t)
            resid = (pred - obs)
            mse = (resid[gmask] ** 2).mean() if gmask is not None else (resid ** 2).mean()
            loss = mse + a.l2_sint * (sint ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if ep % 25 == 0 or ep == 1:
            hs = coverage_eval(f'ep{ep}') if ep % 100 == 0 else None
            print(f"[ep {ep:3d}] mean loss {tot/len(train):.4f}", flush=True)
            if wb is not None:
                d = {'ft/loss': tot / len(train), 'ft/epoch': ep}
                if hs is not None:
                    d.update({f'ft/{g}/rho_frac_pctile': hs[g][2] for g in hs})
                wb_log(wb, d, step=ep)

    post_acc = heldout_pred_acc('POST-finetune')
    post = coverage_eval('POST-finetune')
    print("\n===== CAUSAL COVERAGE TEST =====", flush=True)
    print(f"  ρ-frac pctile (magnitude): PRE {pre['held-out-HSPC'][2]:.0f} -> POST {post['held-out-HSPC'][2]:.0f}", flush=True)
    print(f"  n_briggs_train={n_briggs} | held-out PREDICTION corr (decisive, magnitude-free):", flush=True)
    for g in heldout_groups:
        if pre_acc.get(g) is not None:
            print(f"    {g:14s} PRE {pre_acc[g]:+.3f} -> POST {post_acc[g]:+.3f}  (Δ {post_acc[g]-pre_acc[g]:+.3f})", flush=True)
    print("  RULE: cov+ should IMPROVE corr on held-out pairs; cov0 should NOT.", flush=True)
    if wb is not None:
        wb_summary(wb, {f'post/{g}/rho_frac_pctile': post[g][2] for g in post})
        wb_summary(wb, {'n_briggs_train': n_briggs})
        for g in heldout_groups:
            if pre_acc.get(g) is not None:
                wb_summary(wb, {f'pred_corr/{g}/pre': pre_acc[g], f'pred_corr/{g}/post': post_acc[g]})
        wb_finish(wb)
    if a.out:
        torch.save({'interaction_U': model.interaction_U.state_dict(),
                    'interaction_out': model.interaction_out.state_dict()}, a.out)
        print(f"saved head -> {a.out}", flush=True)
    print("\n[head-finetune done]", flush=True)


if __name__ == '__main__':
    main()
