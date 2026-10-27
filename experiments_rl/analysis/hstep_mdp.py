"""#1 H-step MDP + #2 causal action gating for the multi-goal set-PPO.

Promotes the one-step set-bandit to a real H-step MDP: each TOKEN is a step; the state is the CUMULATIVE
cocktail's post-state z_t = z_post(S<=t) RE-SOLVED on the full partial set (so the rank-32 interaction head
sees the whole partial cocktail and z_H = z_post(S) exactly). Per-step potential shaping r_t = gamma*Phi(z_t)
- Phi(z_{t-1}) (telescopes -> optimum unchanged, densified), value V(z_t, z_g) with GAE(lambda), STOP action,
and per-step DAG-std pessimism (partial sets are mostly in-distribution). Per-token PPO ratios+clip.

Reuses SetPolicy (encode + _slot_logits per step, slot=min(t,kmax-1)) and SetCritic. self-contained update.
"""
import numpy as np
import torch


def causal_cand_mask(model, op, cand, cent_li, topP=40, keep_frac=0.5):
    """#2: allowed candidate LOCAL indices = TFs whose latent shift PROPAGATES (via the SCM (I-A^T)^-1) to the
    goal's DEG-programs. goal DEG-programs = latents whose masked-GO genes carry |cent_li|. Uses the dense SCM
    influence (raw A is near-empty at any threshold), keeps the top keep_frac TFs by reachability -> non-empty."""
    dev = cent_li.device
    A = op.mean_A().detach().to(dev)
    if A.dim() == 3: A = A[0]
    d = A.shape[0]
    M = getattr(model.decoder, 'dec_mask', None)                       # masked-GO decoder: [n_genes, d] pathway mask
    if M is None: M = getattr(model.decoder, 'mask', None)
    if M is None: M = getattr(model.decoder, 'sena_mask', None)
    if M is None: return None                                           # no pathway mask -> gate inert
    M = M.detach().to(dev)
    if M.shape[0] != d: M = M.T                                        # [d, n_genes]
    prog_w = (M * cent_li.abs().unsqueeze(0)).sum(1)                   # [d] goal-program weight
    prog_w = prog_w / (prog_w.sum() + 1e-9)
    I = torch.eye(d, device=dev)
    try:
        G = torch.linalg.inv(I - A.T)                                  # [d,d] SCM influence: G[i,j] = influence of j on i
    except Exception:
        G = torch.linalg.inv(I - A.T + 1e-3 * I)
    with torch.no_grad():
        shifts = torch.stack([model.compute_shift([int(gi)]).flatten() for gi in cand])  # [C,d]
    prop = shifts.abs() @ G.abs().T                                    # [C,d] TF-shift propagated to every program
    score = (prop * prog_w.unsqueeze(0)).sum(1)                        # [C] TF -> goal reachability under A
    k = max(1, int(keep_frac * len(cand)))
    return set(torch.argsort(-score)[:k].tolist())


def cosine_cand_mask(op, data, env, cand, A, keep_frac=0.25, zsingle=None):
    """Cheap ANSWER-AGNOSTIC retriever: allowed local idx = top-K candidates by SINGLE-TF Φ to the goal
    (env must already be reset to the goal so env._phi uses its target). Φ_single[c] = drift-removed match of
    the WM-predicted single-TF Δ to the fate target — i.e. cos(single-TF effect, goal) in the exact reward space.
    Drivers are high-cosine by construction, so this reproduces the (answer-informed) driver_pool WITHOUT labels.
    NOTE: circular for drv_recall (annotation IS cosine) => score such runs on reachability/Jaccard/HSPC, not drv_recall.
    zsingle: optional precomputed [C,d] single-TF post-states (goal-independent) to avoid re-solving per goal."""
    eps0 = data.eps_ctrl_mean.unsqueeze(0).to(A.device)
    if zsingle is None:
        zsingle = torch.stack([op.apply(eps0, [int(c)], A)[0].squeeze(0) for c in cand])   # [C,d]
    with torch.no_grad():
        phi_c = env._phi(zsingle)                                              # [C] single-TF match to THIS goal
    k = max(1, int(keep_frac * len(cand)))
    return set(torch.argsort(-phi_c)[:k].tolist()), zsingle


def train_hstep(agent, iters, tag, *, op, data, env, cand, keys, TR, names, cent, drift, ctrl_recon, A,
                dev, d, a, wb, wb_log, model, EXP20, maha_fn=None):
    """H-step PPO with GAE(lambda). Reuses agent.policy/critic/opt_a/opt_c.
    Reward levers (all read off `a`): CVaR-lower DAG pessimism (--hstep_cvar_frac>0, else symmetric --alpha·std),
    InfoNCE fate-specificity (--contrast_mode infonce, --beta_contrast/--contrast_tau), and a manifold penalty
    (--beta_maha·maha_fn, where maha_fn = unified min-dist to the Joung∪Briggs perturbation manifolds)."""
    from collections import Counter
    from analysis.fate_rl import pred_resid
    rng2 = np.random.default_rng(1); r0 = []; _tlog = []; _cum = Counter(); binfo = {}   # binfo = per-goal EMA baseline for the SEPARATE specificity term (--contrast_separate)
    eps0 = data.eps_ctrl_mean.unsqueeze(0).to(dev)                      # [1,d] control eps
    z0 = data.z0.to(dev); C = agent.policy.C; ed = agent.policy.emb_dim
    gam = float(getattr(a, 'gamma', 0.99)); lam = float(getattr(a, 'gae_lambda', 0.95)); Ndag = a.n_dag
    kfloor = agent.policy.k_max if getattr(a, 'mask_stop', False) else int(getattr(a, 'k_floor', 1))  # mask_stop => forbid STOP at ALL slots => forced to the per-fate/per-source kmax_active cap
    cvar_frac = float(getattr(a, 'hstep_cvar_frac', 0.0))              # >0 => reward-level CVaR-LOWER over N DAGs (else std pessimism)
    _hg = float(getattr(a, 'phi_hinge', 0.0)); _pw = float(getattr(a, 'phi_pow', 0.0))   # DE-SHALLOW the potential before shaping
    def _ft(x):                                                        # hinge: max(0,Φ−T) zeros the shallow region; pow: sign·|Φ|^p steepens toward the peak (keeps gradient everywhere)
        if _hg > 0: return max(0.0, x - _hg)
        if _pw > 0: return (abs(x) ** _pw) if x >= 0 else -(abs(x) ** _pw)
        return x
    if _hg > 0 or _pw > 0: print(f"[phi_transform] DE-SHALLOW reward: hinge={_hg} pow={_pw} (applied to Φ before potential-shaping)", flush=True)
    _bc = float(getattr(a, 'beta_contrast', 0.0)); _cm = getattr(a, 'contrast_mode', 'mean') if _bc > 0 else 'off'
    centn = cent / (cent.norm(dim=1, keepdim=True) + 1e-9)             # [K,G] for the fate-specificity contrast term
    print(f"[hstep-levers] cvar_frac={cvar_frac} (0=std pess) | contrast={_cm} (beta={_bc}) | "
          f"maha={'on' if (maha_fn is not None and a.beta_maha>0) else 'off'} (beta={a.beta_maha})", flush=True)
    # #2 causal gate: precompute allowed candidates per goal
    gate = {}
    _ext_gate = getattr(env, '_goal_gate', None)                     # RETRIEVER/cosine gate (built in main): per-goal allowed local idx -> overrides causal_gate
    _use_gate = _ext_gate is not None or getattr(a, 'causal_gate', False)
    if _ext_gate is not None:
        gate = _ext_gate
        _nz = [len(v) for v in gate.values() if v]; print(f"[ext_gate] (hstep) using env._goal_gate | median allowed cand/goal = {int(np.median(_nz)) if _nz else 0}/{C}", flush=True)
    elif getattr(a, 'causal_gate', False):
        for li in set(TR):
            gate[li] = causal_cand_mask(model, op, cand, cent[li].to(dev), keep_frac=float(getattr(a,'causal_gate_keep',0.5)))
        _nz = [len(v) for v in gate.values() if v]; print(f"[causal_gate] median allowed cand/goal = {int(np.median(_nz)) if _nz else 0}/{C}", flush=True)
        if getattr(a, 'gate_audit', False):                              # AUDIT: do the gates retain each fate's TRUE answer drivers?
            _n2l = {data.gene_names[cand[i]]: i for i in range(len(cand))}
            _rets = []
            for _li in set(TR):
                _drv = set(data.lineage_drivers[names[_li]].get('train', []) + data.lineage_drivers[names[_li]].get('test', []))
                _dl = {_n2l[g] for g in _drv if g in _n2l}
                if _dl and gate.get(_li): _rets.append(len(_dl & gate[_li]) / len(_dl))
            if _rets:
                import numpy as _np
                print(f"[gate_audit] answer-driver retention in gate: mean={_np.mean(_rets):.2f} min={_np.min(_rets):.2f} median={_np.median(_rets):.2f} frac_fates_ALL_retained={_np.mean([r>=0.999 for r in _rets]):.2f} (n={len(_rets)} train fates w/ drivers)", flush=True)
            import sys as _sys; _sys.exit(0)
    buf = []; it = 0; upd = 0
    _terms = {'shape': [], 'pess': [], 'info': [], 'maha_pen': [], 'maha_raw': [], 'cvar_gap': [], 'tf_cost': []}  # per-term magnitudes => catch INERT levers
    from collections import deque as _deque
    sil_c = float(getattr(a, 'hstep_sil', 0.0)); sil_cons = bool(getattr(a, 'hstep_sil_conservative', False))
    _sil = _deque(maxlen=30000)                                       # SELF-IMITATION buffer: (s, t_emb, token, slot, ctx, R). R = exact terminal Φ (frozen WM => no bootstrap staleness)
    if sil_c > 0: print(f"[hstep-SIL] self-imitation on (coeff={sil_c}, return={'conservative min_k Φ' if sil_cons else 'Φ'}) — MLE on past actions weighted by (R−V)_+", flush=True)
    while it < iters:
        li = TR[int(rng2.integers(len(TR)))]; env.reset(keys[li]); t_emb = env.z_target.to(dev)
        eps0 = env.eps0_active.unsqueeze(0).to(dev); z0 = env.z0_active.to(dev)   # FRAME-ROUTED control start (per goal); = Joung control when frame routing off
        allowed = gate.get(li) if _use_gate else None
        chosen = []; S = []; zt = z0.clone(); phi_prev = _ft(float(env._phi(zt.unsqueeze(0))[0])); ep = []; last_phi_ens = None
        _kcap = int(getattr(env, 'kmax_active', None) or a.kmax)          # PER-SOURCE cap (Joung k<=3 support): stop rollout at the goal's cap
        for step in range(_kcap):
            with torch.no_grad():
                h = agent.policy.encode(zt.unsqueeze(0), t_emb.unsqueeze(0))
                cmask = torch.zeros(1, C, dtype=torch.bool, device=dev)
                if chosen: cmask[0, chosen] = True
                if allowed is not None:
                    block = torch.ones(C, dtype=torch.bool, device=dev); block[list(allowed)] = False
                    cmask = cmask | block.unsqueeze(0)
                slot = min(step, agent.policy.k_max - 1)
                logits = agent.policy._slot_logits(h, torch.zeros(1, ed, device=dev), slot, cmask)
                if step < kfloor: logits[:, 0] = -1e9                   # k-floor: forbid STOP before slot k_floor (min cocktail size)
                dist = torch.distributions.Categorical(logits=logits)
                tok = int(dist.sample()[0]); logp = float(dist.log_prob(torch.tensor([tok], device=dev))[0])
                V = float(agent.critic(zt.unsqueeze(0), t_emb.unsqueeze(0))[0])
            ctx = tuple(chosen)                                         # chosen-mask context BEFORE this token
            if tok == 0:                                                # STOP -> terminal, r=0
                ep.append((zt.detach().cpu(), 0, slot, ctx, logp, V, 0.0, 0.0)); it += 1; break
            g = tok - 1; chosen.append(g); S.append(int(cand[g]))
            with torch.no_grad():
                spec_t = 0.0                                            # separate-specificity signal (set below if --contrast_separate)
                zt_new, _, _ = op.apply(eps0, S, A)                     # cumulative re-solve (full interaction head)
                zens, _, _ = op.apply_ensemble(eps0.repeat(Ndag, 1), S, op.sample_dags(Ndag))
                phi_ens = env._phi(zens)                                # [Ndag] potential under sampled DAGs
                last_phi_ens = phi_ens                                   # retained for the SIL conservative return label (min_k Φ at the final cocktail)
                _phi_mean = float(phi_ens.mean())
                if cvar_frac > 0:                                       # reward-level CVaR-LOWER: mean of the worst-frac DAG outcomes (downside-robust)
                    kk = max(1, int(np.ceil(cvar_frac * phi_ens.numel())))
                    phi_pot = float(torch.sort(phi_ens).values[:kk].mean()); pess = 0.0
                    _terms['cvar_gap'].append(_phi_mean - phi_pot)     # >0 => CVaR actually penalizes downside (else inert)
                else:                                                  # symmetric-std pessimism (original)
                    phi_pot = _ft(float(env._phi(zt_new)[0])); pess = a.alpha * float(phi_ens.std())
                shape = a.reward_scale * (gam * phi_pot - phi_prev)
                r_t = shape - a.reward_scale * pess
                _terms['shape'].append(abs(shape)); _terms['pess'].append(a.reward_scale * pess)
                _tfc = float(getattr(a, 'tf_cost', 0.0))                # PER-TF cost: −λ per non-STOP step => terminal −λ|S| (parsimony; anti-breadth-inflation)
                if _tfc:
                    r_t -= _tfc; _terms['tf_cost'].append(_tfc)
                if a.beta_contrast > 0:                                # fate-SPECIFICITY penalty (targets the goal-agnostic mode collapse)
                    rv = pred_resid(op, data, S, drift, ctrl_recon, A)
                    cos_all = centn @ (rv / (rv.norm() + 1e-9))
                    cm = getattr(a, 'contrast_mode', 'mean')
                    if cm == 'infonce':                                # mild: softmax over fates
                        info = float(cos_all[li]) / a.contrast_tau - float(torch.logsumexp(cos_all / a.contrast_tau, 0))
                    elif cm == 'max':                                  # SHARP: margin to the TOP competing fate
                        comp = cos_all.clone(); comp[li] = -1e9
                        info = float(cos_all[li] - comp.max())
                    elif cm == 'focus':                                # SHARPEST: hard focus-region — penalize if the cocktail's BEST fate isn't the conditioned one
                        bg = int(cos_all.argmax()); info = 0.0 if bg == li else -float(cos_all[bg] - cos_all[li])
                    else:                                              # 'mean': mild mean-reference (goal-agnostic penalty, soft)
                        info = float(cos_all[li] - cos_all.mean())
                    if getattr(a, 'contrast_separate', False):        # SEPARATE policy-grad term (own baseline), NOT in reward
                        spec_t = float(info); _terms['info'].append(abs(a.beta_contrast * info))
                    else:
                        info = a.beta_contrast * info; r_t += info; _terms['info'].append(abs(info))
                if maha_fn is not None and a.beta_maha > 0:            # HINGE manifold penalty (Joung ∪ Briggs), matching env: beta·max(0, maha−tau)
                    _mraw = float(maha_fn(zt_new)); _mpen = a.beta_maha * max(0.0, _mraw - a.maha_tau)
                    r_t -= _mpen; _terms['maha_pen'].append(_mpen); _terms['maha_raw'].append(_mraw)
            ep.append((zt.detach().cpu(), tok, slot, ctx, logp, V, r_t, spec_t))
            zt = zt_new.squeeze(0); phi_prev = phi_pot; _cum[data.gene_names[int(cand[g])]] += 1; it += 1
        # GAE(lambda), terminal V=0
        adv = 0.0; gae = [0.0] * len(ep)
        for k in range(len(ep) - 1, -1, -1):
            Vn = ep[k + 1][5] if k + 1 < len(ep) else 0.0
            delta = ep[k][6] + gam * Vn - ep[k][5]; adv = delta + gam * lam * adv; gae[k] = adv
        _cs = getattr(a, 'contrast_separate', False) and a.beta_contrast > 0
        if _cs:                                                         # per-goal EMA baseline of the specificity signal (own baseline, decoupled from the critic)
            _ts = float(np.mean([e[7] for e in ep if e[1] != 0])) if any(e[1] != 0 for e in ep) else 0.0
            binfo[li] = 0.9 * binfo.get(li, _ts) + 0.1 * _ts
        for k, tr in enumerate(ep):
            _sp = a.beta_contrast * (tr[7] - binfo.get(li, 0.0)) if _cs else 0.0   # centered, scaled specificity advantage
            buf.append((tr[0], t_emb.detach().cpu(), tr[1], tr[2], tr[3], tr[4], gae[k], gae[k] + tr[5], _sp))
        if sil_c > 0:                                                    # SIL: buffer the trajectory keyed by its EXACT terminal return R (frozen WM); conservative = min_k Φ over the DAG ensemble
            R_sil = float(last_phi_ens.min()) if (sil_cons and last_phi_ens is not None) else float(phi_prev)
            _te = t_emb.detach().cpu()
            for tr in ep:
                if tr[1] != 0: _sil.append((tr[0], _te, tr[1], tr[2], tr[3], R_sil))   # skip STOP; (s, t_emb, token, slot, ctx, R)
        r0.append(sum(e[6] for e in ep))
        if len(buf) >= 256:
            _eat = float(getattr(a, 'ent_anneal_to', -1.0))                 # entropy anneal: linearly decay ent_c from ent_coeff -> ent_anneal_to over training
            if _eat >= 0: agent.ent_c = float(a.ent_coeff) + (_eat - float(a.ent_coeff)) * min(1.0, it / max(1, iters))
            st = _hstep_update(agent, buf, C, ed, dev, kfloor); upd += 1
            if sil_c > 0 and len(_sil) >= 256:                          # auxiliary self-imitation: MLE on past actions weighted by (R−V)_+
                _sil_update(agent, _sil, C, ed, dev, sil_c, kfloor=kfloor)
            tm = {k: (float(np.mean(v[-512:])) if v else 0.0) for k, v in _terms.items()}   # mean per-term magnitude => 0 == INERT
            wb_log(wb, {f'{tag}/reward': float(np.mean(r0[-64:])), f'{tag}/adv_std': st['adv_std'],
                        f'{tag}/entropy': st['entropy'], f'{tag}/approx_kl': st['approx_kl'],
                        f'{tag}/explained_variance': st['ev'],
                        f'{tag}/term_shape': tm['shape'], f'{tag}/term_pess': tm['pess'], f'{tag}/term_info': tm['info'],
                        f'{tag}/term_maha_pen': tm['maha_pen'], f'{tag}/maha_raw': tm['maha_raw'], f'{tag}/cvar_gap': tm['cvar_gap'],
                        f'{tag}/term_tf_cost': tm['tf_cost']}, step=it)
            _tlog.append(dict(it=it, reward=float(np.mean(r0[-64:])), adv_std=st['adv_std'], entropy=st['entropy'],
                              approx_kl=st['approx_kl'], ev=st['ev'], n_distinct=len(_cum), **{f't_{k}': tm[k] for k in tm}))
            buf = []
            if upd % 20 == 0:
                print(f"  [{tag}] it {it:6d} reward {np.mean(r0[-64:]):+.3f} EV {st['ev']:+.2f} ent {st['entropy']:.2f} | "
                      f"TERMS shape {tm['shape']:.3f} pess {tm['pess']:.3f} info {tm['info']:.3f} "
                      f"maha_pen {tm['maha_pen']:.3f} (raw {tm['maha_raw']:.0f}) cvar_gap {tm['cvar_gap']:.3f} "
                      f"tf_cost {tm['tf_cost']:.3f}", flush=True)
    import pandas as pd
    _sfx = (a.run_name or 'hstep').replace('/', '_')
    pd.DataFrame(_tlog).to_csv(f'{EXP20}/results/briggs/rl_trainlog_{tag}_{_sfx}.csv', index=False)
    pd.DataFrame(_cum.most_common(), columns=['TF', 'count']).to_csv(f'{EXP20}/results/briggs/rl_pickcum_{tag}_{_sfx}.csv', index=False)
    print(f"  [{tag}] H-step done | updates {upd} | distinct TFs {len(_cum)} | top {_cum.most_common(8)}", flush=True)
    return agent


def _hstep_update(agent, buf, C, ed, dev, kfloor=1):
    """Per-token PPO update over the H-step transitions (per-sample re-eval: slot + chosen-mask vary)."""
    S = torch.stack([b[0] for b in buf]).to(dev); Tg = torch.stack([b[1] for b in buf]).to(dev)
    TOK = torch.tensor([b[2] for b in buf], device=dev); SLOT = [b[3] for b in buf]; CTX = [b[4] for b in buf]
    OLD = torch.tensor([b[5] for b in buf], device=dev); ADV = torch.tensor([b[6] for b in buf], device=dev)
    RET = torch.tensor([b[7] for b in buf], device=dev)
    SPEC = torch.tensor([b[8] if len(b) > 8 else 0.0 for b in buf], device=dev)   # separate specificity advantage (0 unless --contrast_separate)
    V0 = agent.critic(S, Tg).detach()
    ev = float(1.0 - (RET - V0).var() / (RET.var() + 1e-8)); adv_std = float(ADV.std())
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    n = len(buf); kls = []; ents = []
    for _ in range(agent.n_epochs):
        idx = torch.randperm(n, device=dev)
        for s0 in range(0, n, agent.batch_size):
            b = idx[s0:s0 + agent.batch_size]
            h = agent.policy.encode(S[b], Tg[b])                       # [B,H]
            newlp = torch.zeros(len(b), device=dev); ent = torch.zeros(len(b), device=dev)
            for r, bi in enumerate(b.tolist()):                        # per-sample: slot + chosen-mask differ
                cmask = torch.zeros(1, C, dtype=torch.bool, device=dev)
                if CTX[bi]: cmask[0, list(CTX[bi])] = True
                logits = agent.policy._slot_logits(h[r:r + 1], torch.zeros(1, ed, device=dev), SLOT[bi], cmask)
                if SLOT[bi] < kfloor: logits[:, 0] = -1e9
                dist = torch.distributions.Categorical(logits=logits)
                newlp[r] = dist.log_prob(TOK[b][r:r + 1])[0]; ent[r] = dist.entropy()[0]
            ratio = torch.exp(newlp - OLD[b])
            with torch.no_grad(): kls.append(float(((ratio - 1) - (newlp - OLD[b])).mean()))
            surr1 = ratio * ADV[b]; surr2 = torch.clamp(ratio, 1 - agent.clip, 1 + agent.clip) * ADV[b]
            aloss = -torch.min(surr1, surr2).mean() - agent.ent_c * ent.mean() - (SPEC[b] * newlp).mean()   # + separate specificity REINFORCE (own baseline; 0 unless --contrast_separate)
            agent.opt_a.zero_grad(); aloss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm); agent.opt_a.step()
            val = agent.critic(S[b], Tg[b]); closs = agent.val_c * ((val - RET[b]) ** 2).mean()
            agent.opt_c.zero_grad(); closs.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), agent.max_grad_norm); agent.opt_c.step()
            ents.append(float(ent.mean()))
    return dict(adv_std=adv_std, entropy=float(np.mean(ents)), approx_kl=float(np.mean(kls)), ev=ev)


def _sil_update(agent, sil, C, ed, dev, sil_c, n=256, iters=4, kfloor=1):
    """Self-imitation (Oh et al. 2018): the PPO-side analogue of TB's off-policy validity. An auxiliary MLE loss on
    buffered past actions weighted by the clipped advantage (R−V)_+, plus a value regression toward R on the same
    positive-advantage samples:
        L_SIL = E_(s,a,R)~buffer[ −log π_θ(a|s)·(R−V_θ(s))_+ ]  +  ½·(R−V_θ(s))_+²
    Off-policy-valid because it is *weighted imitation*, not an IS-corrected gradient — no ratio, no clip pathology.
    R is the EXACT terminal return (frozen WM + deterministic transition => the buffered (S,Φ) never goes stale),
    and when the buffer is keyed on the conservative score (min_k Φ over the DAG ensemble) SIL imitates *robust*
    high-reward cocktails and pulls the policy away from single-DAG gamers."""
    for _ in range(iters):
        idx = np.random.randint(0, len(sil), min(n, len(sil)))
        S = torch.stack([sil[i][0] for i in idx]).to(dev); Tg = torch.stack([sil[i][1] for i in idx]).to(dev)
        TOK = [int(sil[i][2]) for i in idx]; SLOT = [sil[i][3] for i in idx]; CTX = [sil[i][4] for i in idx]
        R = torch.tensor([sil[i][5] for i in idx], device=dev)
        with torch.no_grad(): adv = torch.relu(R - agent.critic(S, Tg))  # (R−V)_+ imitation weight (detached: no grad to critic here)
        h = agent.policy.encode(S, Tg); lp = torch.zeros(len(idx), device=dev)
        for r in range(len(idx)):                                        # per-sample slot + chosen-mask (mirrors _hstep_update)
            cmask = torch.zeros(1, C, dtype=torch.bool, device=dev)
            if CTX[r]: cmask[0, list(CTX[r])] = True
            logits = agent.policy._slot_logits(h[r:r + 1], torch.zeros(1, ed, device=dev), SLOT[r], cmask)
            if SLOT[r] < kfloor: logits[:, 0] = -1e9
            lp[r] = torch.log_softmax(logits, -1)[0, TOK[r]]
        ploss = -(lp * adv).mean()                                       # −log π·(R−V)_+  (imitate only actions beating their value)
        agent.opt_a.zero_grad(); (sil_c * ploss).backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm); agent.opt_a.step()
        V = agent.critic(S, Tg); vloss = 0.5 * (torch.relu(R - V) ** 2).mean()   # ½(R−V)_+²  (value climbs toward R on positive-adv)
        agent.opt_c.zero_grad(); (sil_c * vloss).backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), agent.max_grad_norm); agent.opt_c.step()
