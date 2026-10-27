"""GFlowNet (trajectory-balance) variant of the H-step set-planner — the principled fix for the mode-collapse
the PPO sweep diagnosed (greedy argmax → ZNF680/DUXA for every fate). Instead of maximizing E[R] (argmax-seeking),
a GFlowNet learns π(cocktail) ∝ R(cocktail), so it COVERS the modes by construction — diverse high-reward
cocktails for downstream screening (Jain et al. 2022, "Biological Sequence Design with GFlowNets").

Our H-step already IS a GFlowNet trajectory: append one TF at a time to a partial cocktail, terminal reward at
STOP. Autoregressive ⇒ single parent ⇒ P_B ≡ 1, so the TB loss is  ( logZ(g) + Σ_t logP_F(a_t|s_t) − β·Φ(x) )².
Two simplifications vs the paper: (1) our REWARD ORACLE is the frozen WM (cheap, queryable) — no surrogate/
uncertainty/acquisition proxy needed; (2) goal-conditioned ⇒ logZ is a small head logZ(z_goal). Off-policy replay
(a buffer of past high-reward cocktails, mix γ) = the "memory of past TFs"; TB is off-policy-valid (unlike PPO).

Reuses agent.policy (SetPolicy) as P_F. Reward R(x)=exp(β·Φ), Φ = env drift-removed match to the goal.
"""
import numpy as np
import torch
import torch.nn as nn


def train_gflownet(agent, iters, tag, *, op, data, env, cand, keys, TR, names, cent, drift, ctrl_recon, A,
                   dev, d, a, wb, wb_log, model, EXP20, maha_fn=None):
    from collections import Counter, deque
    from analysis.fate_rl import pred_resid
    _bc = float(getattr(a, 'beta_contrast', 0.0)); _cm = getattr(a, 'contrast_mode', 'mean')   # fate-specificity reward (folded into Φ)
    centn = cent / (cent.norm(dim=1, keepdim=True) + 1e-9)                                       # [K,G] unit fate targets for the contrast
    if _bc > 0: print(f"[gflownet-contrast] mode={_cm} beta={_bc} tau={getattr(a,'contrast_tau',0.2)} => R=exp(beta_gfn·(Φ+contrast))", flush=True)
    rng2 = np.random.default_rng(1); _cum = Counter(); _tlog = []
    eps0 = data.eps_ctrl_mean.unsqueeze(0).to(dev); z0 = data.z0.to(dev)
    C = agent.policy.C; ed = agent.policy.emb_dim; kmax = a.kmax
    beta = float(getattr(a, 'gflownet_beta', 6.0))        # reward exponent: R = exp(beta·Φ)  (sharper => more greedy)
    gamma = float(getattr(a, 'gflownet_gamma', 0.5))      # fraction of each batch drawn from the replay buffer (off-policy memory)
    from analysis.hstep_mdp import causal_cand_mask                    # causal_gate for GFlowNet (was PPO-only)
    _gate = {}
    _ext_gate = getattr(env, '_goal_gate', None)                       # RETRIEVER/cosine gate (built in main) -> overrides causal_gate
    if _ext_gate is not None:
        _gate = _ext_gate
        _nz = [len(v) for v in _gate.values() if v]; print(f"[ext_gate] (gflownet) using env._goal_gate | median allowed cand/goal = {int(np.median(_nz)) if _nz else 0}/{C}", flush=True)
    elif getattr(a, 'causal_gate', False):
        _kf = float(getattr(a, 'causal_gate_keep', 0.5))
        for _li in set(TR): _gate[_li] = causal_cand_mask(model, op, cand, cent[_li].to(dev), keep_frac=_kf)
        _nz = [len(v) for v in _gate.values() if v]
        print(f"[causal_gate] (gflownet) median allowed cand/goal = {int(np.median(_nz)) if _nz else 0}/{C} keep_frac={_kf}", flush=True)
    pess = float(getattr(a, 'gflownet_pess', 0.0))        # >0 => DAG-pessimistic Φ = mean − pess·std over N DAGs (matches H-step)
    B = 16                                                 # trajectories per update
    # logZ (partition fn). 'cond' = MLP head on z_goal (can ABSORB goal-dependence => P_F under-conditions => low
    # specificity); 'shared' = one scalar for ALL goals => forces P_F to carry the goal signal (diverse AND precise).
    # logz_wd = weight decay on the logZ head (soft version of the same pressure).
    logz_mode = getattr(a, 'gflownet_logz', 'cond'); logz_wd = float(getattr(a, 'gflownet_logz_wd', 0.0))
    tdim = env.z_target.numel(); logZ = None; logZ_p = None
    if logz_mode == 'shared':
        logZ_p = nn.Parameter(torch.zeros(1, device=dev)); zparams = [logZ_p]
    else:
        logZ = nn.Sequential(nn.Linear(tdim, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev); zparams = list(logZ.parameters())
    _logZ = lambda t: (logZ_p[0] if logz_mode == 'shared' else logZ(t.unsqueeze(0))[0, 0])
    opt = torch.optim.Adam([{'params': agent.policy.parameters(), 'lr': agent.lr_actor},
                            {'params': zparams, 'lr': 1e-2, 'weight_decay': logz_wd}])
    PRI = bool(getattr(a, 'gflownet_prioritized', False)); CAP = 5000   # prioritized replay: Φ-weighted sampling + keep-best eviction (uses the stored Φ)
    buf = [] if PRI else deque(maxlen=CAP)                 # replay: (li, tokens, phi)
    print(f"[gflownet] TB objective | beta={beta} R=exp(beta·Φ) | replay gamma={gamma} | logZ={logz_mode} (wd={logz_wd}) "
          f"{'shared-scalar => forces P_F to condition' if logz_mode=='shared' else 'cond MLP on z_goal'}", flush=True)

    def rollout(li, t_emb, teacher=None, allowed=None):
        """Roll out a cocktail. teacher=token list => teacher-force (replay re-eval); else sample.
        Returns (tokens, sum_logPF[grad tensor], phi[float], gene_set)."""
        chosen, S = [], []; sum_lp = torch.zeros((), device=dev); toks = []
        _kcap = int(getattr(env, 'kmax_active', None) or kmax)             # PER-SOURCE cap (set by env.reset per goal): Joung<=2, Briggs<=8 (was: global a.kmax => Joung extrapolated to k=8)
        for step in range(_kcap):
            h = agent.policy.encode(z_cur(S).unsqueeze(0), t_emb.unsqueeze(0))
            cmask = torch.zeros(1, C, dtype=torch.bool, device=dev)
            if chosen: cmask[0, chosen] = True
            if allowed is not None:                                        # causal_gate: restrict to DAG-reachable candidates (shrink the haystack)
                block = torch.ones(C, dtype=torch.bool, device=dev); block[list(allowed)] = False
                cmask = cmask | block.unsqueeze(0)
            slot = min(step, agent.policy.k_max - 1)
            logits = agent.policy._slot_logits(h, torch.zeros(1, ed, device=dev), slot, cmask)
            if step == 0: logits[:, 0] = -1e9                             # no STOP at slot 0
            logp = torch.log_softmax(logits, -1)[0]
            if teacher is not None:
                tok = teacher[step] if step < len(teacher) else 0
            else:
                tok = int(torch.distributions.Categorical(logits=logits).sample()[0])
            sum_lp = sum_lp + logp[tok]; toks.append(tok)
            if tok == 0: break
            g = tok - 1; chosen.append(g); S.append(int(cand[g]))
        with torch.no_grad():
            if pess > 0 and S:                                       # DAG-pessimistic Φ = mean − α·std over N sampled DAGs (matches H-step robustness)
                zens, _, _ = op.apply_ensemble(eps0.repeat(a.n_dag, 1), S, op.sample_dags(a.n_dag))
                phi_ens = env._phi(zens); phi = float(phi_ens.mean() - pess * phi_ens.std())
            else:
                z = op.apply(eps0, S, A)[0] if S else z0.unsqueeze(0)
                phi = float(env._phi(z if z.dim() == 2 else z.unsqueeze(0))[0])
            if maha_fn is not None and a.beta_maha > 0 and S:
                phi = phi - a.beta_maha * max(0.0, float(maha_fn(op.apply(eps0, S, A)[0])) - a.maha_tau)
            if _bc > 0 and S:                                        # fate-SPECIFICITY reward folded into Φ => R=exp(β·(Φ+contrast)) samples goal-specific cocktails (attacks generic-basin collapse)
                rv = pred_resid(op, data, S, drift, ctrl_recon, A); cos_all = centn @ (rv / (rv.norm() + 1e-9))
                if _cm == 'focus':
                    _bg = int(cos_all.argmax()); info = 0.0 if _bg == li else -float(cos_all[_bg] - cos_all[li])
                elif _cm == 'infonce':
                    info = float(cos_all[li]) / a.contrast_tau - float(torch.logsumexp(cos_all / a.contrast_tau, 0))
                elif _cm == 'max':
                    _cmp = cos_all.clone(); _cmp[li] = -1e9; info = float(cos_all[li] - _cmp.max())
                else:
                    info = float(cos_all[li] - cos_all.mean())
                phi = phi + _bc * info
        return toks, sum_lp, phi, {data.gene_names[int(cand[g])] for g in chosen}

    def z_cur(S):                                                         # cumulative post-state [d] (re-solved), no grad
        with torch.no_grad():
            return op.apply(eps0, S, A)[0].squeeze(0) if S else z0.clone()

    it = 0; upd = 0; r0 = []; _ckpt = 0
    while it < iters:
        loss = torch.zeros((), device=dev); phis = []; nfresh = max(1, int(round((1 - gamma) * B)))
        batch = [('fresh', int(rng2.integers(len(TR)))) for _ in range(nfresh)]
        if len(buf) >= B - nfresh:
            if PRI:                                        # Φ-weighted (reward-prioritized) replay sampling
                _ph = np.array([b[2] for b in buf], np.float64); _w = np.exp((_ph - _ph.max()) / 0.1); _w /= _w.sum()
                _idxs = rng2.choice(len(buf), size=B - nfresh, p=_w)
            else:
                _idxs = [int(rng2.integers(len(buf))) for _ in range(B - nfresh)]
            for _i in _idxs:
                li_r, toks_r, _ = buf[int(_i)]; batch.append(('replay', (li_r, toks_r)))
        for kind, payload in batch:
            if kind == 'fresh':
                li = TR[payload]; env.reset(keys[li]); t_emb = env.z_target.to(dev)
                toks, sum_lp, phi, gs = rollout(li, t_emb, allowed=_gate.get(li))
                buf.append((li, toks, phi))
                if PRI and len(buf) > CAP:                 # keep-best eviction: drop the lowest-Φ trajectory (retain discovered modes)
                    buf.pop(int(np.argmin([b[2] for b in buf])))
                for gname in gs: _cum[gname] += 1
            else:
                li, toks = payload; env.reset(keys[li]); t_emb = env.z_target.to(dev)
                _, sum_lp, phi, _ = rollout(li, t_emb, teacher=toks, allowed=_gate.get(li))
            logZg = _logZ(t_emb)
            loss = loss + (logZg + sum_lp - beta * phi) ** 2               # trajectory-balance residual²
            phis.append(phi); it += 1
        loss = loss / len(batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm); opt.step()
        r0.append(float(np.mean(phis))); upd += 1
        if upd % 20 == 0:
            with torch.no_grad(): _z = float(_logZ(env.z_target.to(dev)))
            wb_log(wb, {f'{tag}/phi': float(np.mean(r0[-64:])), f'{tag}/tb_loss': float(loss),
                        f'{tag}/logZ': _z, f'{tag}/n_distinct': len(_cum), f'{tag}/buffer': len(buf)}, step=it)
            _tlog.append(dict(it=it, phi=float(np.mean(r0[-64:])), tb_loss=float(loss), logZ=_z, n_distinct=len(_cum)))
            print(f"  [{tag}] it {it:6d} Φ {np.mean(r0[-64:]):+.3f} TB_loss {float(loss):.3f} logZ {_z:+.2f} "
                  f"distinct {len(_cum)} buf {len(buf)}", flush=True)
        _cad = 50000 if it <= 400000 else 100000                             # dense early (capture diversity-collapse), coarse late
        if getattr(a, 'save_scratch', None) and it - _ckpt >= _cad:          # periodic checkpoint (wall-out safety + mid-run eval)
            import torch as _t; _t.save(agent.policy.state_dict(), a.save_scratch); _ckpt = it
            if getattr(a, 'save_ladder', False):                             # RETAIN a per-iter ladder (not just the rolling file)
                _stem = a.save_scratch[:-3] if a.save_scratch.endswith('.pt') else a.save_scratch
                _t.save(agent.policy.state_dict(), f"{_stem}_it{it//1000}k.pt")
            print(f"  [{tag}] checkpoint @ it {it} -> {a.save_scratch}", flush=True)
    import pandas as pd
    _sfx = (a.run_name or 'gflownet').replace('/', '_')
    pd.DataFrame(_tlog).to_csv(f'{EXP20}/results/briggs/rl_trainlog_{tag}_{_sfx}.csv', index=False)
    pd.DataFrame(_cum.most_common(), columns=['TF', 'count']).to_csv(f'{EXP20}/results/briggs/rl_pickcum_{tag}_{_sfx}.csv', index=False)
    print(f"  [{tag}] GFlowNet done | updates {upd} | distinct TFs {len(_cum)} | top {_cum.most_common(8)}", flush=True)
    return agent
