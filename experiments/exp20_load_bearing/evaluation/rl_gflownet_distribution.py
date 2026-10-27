"""GFlowNet DISTRIBUTIONAL-CORRESPONDENCE + DIVERSITY-AT-QUALITY analysis for the cell-fate set-planner.

WHY: A GFlowNet learns P(x) proportional to R(x)=exp(beta*Phi(x)) (Bengio 2021; Malkin 2022, Trajectory
Balance) -- it is a *sampler*, so the axes where it can legitimately beat beam search (which returns ONE mode)
are (1) distributional correspondence P(x)~R(x) and (2) diverse-at-quality mode discovery (Jain 2023, MOGFN).
Beam/greedy maximise reward and collapse to ~1 set, so they cover ~0 of the distribution / ~1 mode.

This module is CALLED from experiments_rl/analysis/briggs_multigoal_rl.py under the OPTIONAL, default-off
flag `--distrib_check` (so the exact frozen-WM stack, candidate pool, fate centroids, drift, frame-routing,
per-fate answer-k caps and env._phi reward are BYTE-IDENTICAL to what each policy was trained with -- a
standalone re-derivation of that ~700-line setup would risk silently diverging and making the correspondence
numbers wrong). It does NOT change any default eval behaviour.

TARGET DISTRIBUTION (rigorous):
  The GFlowNet trajectory-balance loss (analysis/gflownet_hstep.py) balances to
      R(x) = exp( beta * ( Phi(x) + beta_contrast * info(x) ) )
  where Phi(x) = env._phi(x) = drift-removed signed-Pearson of the WM-predicted residual to the fate-target
  residual on the fate's DEGs (the "resid" reward mode), and info(x) is the InfoNCE goal-specificity shaping
  term (only for the *_inf arms; the *_noc arms have beta_contrast=0 so their target is the CANONICAL pure
  exp(beta*Phi)). We score correspondence against BOTH targets:
     - 'phi'       : R=exp(beta*Phi)            (the canonical GFlowNet claim the task asks for)
     - 'astrained' : R=exp(beta*(Phi+bc*info))  (what THIS policy's TB objective actually balanced to)
  For the *_noc arms the two coincide. Spearman & Pearson are invariant to the positive scale beta, so they
  are reported vs the score directly; L1/JSD use the ACTUAL R with beta (=48 for the flagship sweep).

OUTPUTS (one CSV per run to avoid SLURM write races; aggregate() merges them with a `run` column):
  results/rl_eval_condz/gflownet_distribution__<run>.csv   (part 1)
  results/rl_eval_condz/gflownet_diversity__<run>.csv       (part 2)
  aggregate() -> results/rl_eval_condz/gflownet_distribution.csv + gflownet_diversity.csv
"""
import os, glob, itertools
import numpy as np
import pandas as pd
import torch

try:
    from scipy.stats import spearmanr, pearsonr
    _HAVE_SCIPY = True
except Exception:                                             # pragma: no cover
    _HAVE_SCIPY = False

OUT_DIR = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/rl_eval_condz"


# ----------------------------------------------------------------------------- metric helpers
def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float('nan')
    if _HAVE_SCIPY:
        return float(spearmanr(x, y).correlation)
    rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float('nan')
    if _HAVE_SCIPY:
        return float(pearsonr(x, y)[0])
    return float(np.corrcoef(x, y)[0, 1])


def _jsd_bits(p, q):
    """Jensen-Shannon divergence in bits (base-2), in [0,1]. p,q are non-negative and each sums to 1."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask] + 1e-300))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _jaccard(a, b):
    a = set(a); b = set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _count_modes(sets_phi, T, jac_sim_cut=0.5):
    """# MODES = distinct cocktails with Phi>T that are mutually diverse (greedy: a cocktail opens a NEW mode
    only if its max Jaccard SIMILARITY to every already-accepted mode is < jac_sim_cut, i.e. Jaccard
    DISTANCE > 1-jac_sim_cut). sets_phi: list[(frozenset, phi)]."""
    cand = sorted([(s, p) for (s, p) in sets_phi if p > T], key=lambda t: -t[1])
    modes = []
    for s, _ in cand:
        if all(_jaccard(s, m) < jac_sim_cut for m in modes):
            modes.append(s)
    return len(modes)


def _mean_pairwise_jac_dist(sets, cap=400):
    sets = list(sets)
    if len(sets) < 2:
        return 0.0
    if len(sets) > cap:                                        # subsample for the O(n^2) pairwise loop
        idx = np.random.default_rng(0).choice(len(sets), cap, replace=False)
        sets = [sets[i] for i in idx]
    ds = [1.0 - _jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(ds)) if ds else 0.0


# ----------------------------------------------------------------------------- core physics (mirror trainer)
def _phi_of_global_set(op, env, A, eps0, S_global):
    """EXACT training Phi = env._phi(op.apply(joung_eps0, S, A)) -- mirrors gflownet_hstep.rollout (applies from
    the Joung control eps0, scores with the fate's frame-routed ctrl_recon set by env.reset)."""
    if not S_global:
        return float('nan')
    z = op.apply(eps0, list(S_global), A)[0]                   # [1,d]
    if z.dim() == 1:
        z = z.unsqueeze(0)
    return float(env._phi(z)[0])


def _info_of_global_set(op, data, pred_resid, centn, drift, ctrl_recon, A, S_global, li, contrast_mode, tau):
    """InfoNCE goal-specificity term folded into the GFlowNet reward (mirrors gflownet_hstep.rollout)."""
    if not S_global:
        return 0.0
    rv = pred_resid(op, data, list(S_global), drift, ctrl_recon, A)
    cos_all = centn @ (rv / (rv.norm() + 1e-9))
    if contrast_mode == 'focus':
        bg = int(cos_all.argmax()); return 0.0 if bg == li else -float(cos_all[bg] - cos_all[li])
    if contrast_mode == 'infonce':
        return float(cos_all[li]) / tau - float(torch.logsumexp(cos_all / tau, 0))
    if contrast_mode == 'max':
        cmp = cos_all.clone(); cmp[li] = -1e9; return float(cos_all[li] - cmp.max())
    return float(cos_all[li] - cos_all.mean())


@torch.no_grad()
def _sample_policy(agent, s_emb, t_emb, n, cand, allowed_block=None):
    """Sample the policy n times exactly as gen_eval does (SetPolicy.sample via agent.select). Returns a list
    of gene-index tuples (global), in sampling order. allowed_block: [C] bool (True=blocked) or None."""
    agent.policy._allowed_block = allowed_block
    out = []
    for _ in range(n):
        _, genes, _, _ = agent.select(s_emb, t_emb)
        out.append(tuple(int(cand[g]) for g in genes))
    agent.policy._allowed_block = None
    return out


# ----------------------------------------------------------------------------- PART 1: distributional
def _feasible_pool_m(keep_sizes, m0, maxenum):
    """Largest pool size m in [max(keep)+1 .. m0] whose enumerated support Sum_k C(m,k) over the kept sizes is
    <= maxenum (so C(pool_m, k*) stays enumerable). Falls back to keeping only the single dominant size."""
    from math import comb
    keep = sorted(keep_sizes)
    while keep:
        lo = max(keep) + 1                                                 # need >= a few sets of the largest kept size
        for m in range(min(m0, 60), lo - 1, -1):
            if sum(comb(m, k) for k in keep if k <= m) <= maxenum:
                return m, keep
        keep = keep[-1:]                                                   # drop all but the dominant size, retry
    return None, []


def _run_distributional_for_fate(li, name, source, agent, op, data, env, cand, cent, centn, drift,
                                  ctrl_recon, A, dev, pred_resid, *, run, meta, pool_m0, maxenum, cover, nsamp,
                                  beam_planner, fix_k=0):
    from collections import Counter
    obs = env.reset(env._key_for(li) if hasattr(env, '_key_for') else meta['keys'][li])
    d = op.d; s_emb, t_emb = obs[:d], obs[d:]
    eps0 = data.eps_ctrl_mean.unsqueeze(0).to(dev)
    kmax_active = int(getattr(env, 'kmax_active', None) or meta['kmax'])

    # ---- WM-derived (arm-independent) candidate ranking by single-TF training-Phi -> top-M0 starting pool ----
    single_phi = np.array([_phi_of_global_set(op, env, A, eps0, [int(c)]) for c in cand], float)
    order_cand = list(np.argsort(-single_phi))

    if fix_k > 0:
        # ---- FORCED support: enumerate exactly the size-fix_k subsets over a fixed pool (identical across
        #      arms => q, ESS(p*) are byte-identical for cond vs shared; the fair comparison the coordinator wants) ----
        pool_m, keep = _feasible_pool_m([int(fix_k)], pool_m0, maxenum)
        if pool_m is None:
            pool_m, keep = min(pool_m0, 13), [int(fix_k)]
    else:
        pool0_local = order_cand[:pool_m0]
        blk0 = torch.ones(len(cand), dtype=torch.bool, device=dev)
        for i in pool0_local:
            blk0[i] = False
        # ---- STAGE 1: measure the policy's ACTUAL set-size distribution on the restricted pool. The beta=48
        #      GFlowNet uses the full cocktail budget (k~kmax), so a fixed small-k enumeration would miss all its
        #      mass; instead enumerate the sizes the policy really samples (>=cover of the mass). ----
        probe = _sample_policy(agent, s_emb, t_emb, min(nsamp, 1000), cand, allowed_block=blk0)
        size_ct = Counter(len(s) for s in probe if s)
        keep, cum, tot = [], 0, sum(size_ct.values())
        for k, c in size_ct.most_common():
            keep.append(k); cum += c
            if cum / max(1, tot) >= cover:
                break
        pool_m, keep = _feasible_pool_m(keep, pool_m0, maxenum)
        if pool_m is None:                                                 # degenerate fallback (shouldn't happen for k<=kmax)
            pool_m, keep = min(pool_m0, 13), [min(size_ct, key=size_ct.get)]
    pool_local = order_cand[:pool_m]
    if meta.get('driver_pool', False):                                     # DRIVER-INCLUSIVE pool: force the fate's answer-key drivers into the enumerable pool (oracle-ceiling that is NOT capped by top-single-Phi pool membership)
        _g2i = getattr(data, 'gene_to_idx', {})
        _cand_pos = {int(c): j for j, c in enumerate(cand)}
        _drvn = data.lineage_drivers.get(name, {}).get('full', [])
        _drv_local = [_cand_pos[_g2i[g]] for g in _drvn if g in _g2i and _g2i.get(g) in _cand_pos]
        pool_local = list(dict.fromkeys(list(_drv_local) + list(order_cand)))[:pool_m]   # drivers first, fill with top single-Phi
    l2g = {i: int(cand[i]) for i in pool_local}; g2l = {g: i for i, g in l2g.items()}

    # ---- enumerate ALL subsets of the KEPT sizes over the (possibly shrunk) pool; score Phi + info ----
    enum_local = []
    for k in keep:
        if k <= pool_m:
            enum_local.extend(itertools.combinations(pool_local, k))
    phis, infos = [], []
    for combo in enum_local:
        Sg = [l2g[i] for i in combo]
        phis.append(_phi_of_global_set(op, env, A, eps0, Sg))
        infos.append(_info_of_global_set(op, data, pred_resid, centn, drift, ctrl_recon, A, Sg, li,
                                         meta['contrast_mode'], meta['contrast_tau']) if meta['beta_contrast'] > 0 else 0.0)
    phis = np.array(phis, float); infos = np.array(infos, float)
    key_of = {frozenset(c): i for i, c in enumerate(enum_local)}

    # ---- STAGE 2: sample the policy on the final restricted pool; tally onto the enumerated support ----
    blk = torch.ones(len(cand), dtype=torch.bool, device=dev)
    for i in pool_local:
        blk[i] = False
    samples = _sample_policy(agent, s_emb, t_emb, nsamp, cand, allowed_block=blk)
    counts = np.zeros(len(enum_local), float); n_on = 0; sampled_sizes = []
    keepset = set(keep)
    for s in samples:
        sampled_sizes.append(len(s))
        fs_local = frozenset(g2l[g] for g in s if g in g2l)
        if len(fs_local) == len(s) and len(s) in keepset and fs_local in key_of:
            counts[key_of[fs_local]] += 1; n_on += 1
    on_frac = n_on / max(1, len(samples))

    rows = []
    for target, score in [('phi', meta['beta'] * phis),
                          ('astrained', meta['beta'] * (phis + meta['beta_contrast'] * infos))]:
        if target == 'astrained' and meta['beta_contrast'] == 0:
            continue                                                       # identical to 'phi' for the *_noc arms
        q = np.exp(score - score.max()); q = q / q.sum()                   # target R/Z over the enumerated support
        p = counts / max(1.0, counts.sum())                               # empirical (renormalised over on-support)
        # ---- ORACLE: exact target TF-marginal q_TF(a|g)=sum_{x:a in x} q(x); rank pool TFs; driver recall (CEILING) ----
        logZ_star = float(score.max() + np.log(np.sum(np.exp(score - score.max()))))   # exact log-normalizer over enum support
        _pool_genes = [str(data.gene_names[int(cand[i])]) for i in pool_local]
        _loc = {i: j for j, i in enumerate(pool_local)}                    # cand-index -> pool position
        _qTF = np.zeros(len(pool_local), float)
        for _xi, _combo in enumerate(enum_local):
            for _i in _combo:
                _qTF[_loc[_i]] += q[_xi]
        _rankg = [_pool_genes[j] for j in np.argsort(-_qTF)]
        _drv = set(map(str, data.lineage_drivers.get(name, {}).get('full', [])))
        _nd = len(_drv)
        _orec = (lambda k: (len(set(_rankg[:k]) & _drv) / _nd) if _nd else float('nan'))
        _pool_cov = (len(set(_pool_genes) & _drv) / _nd) if _nd else float('nan')   # drivers even present in the top-M single-phi pool
        sampled_mask = counts > 0
        # ---- MONTE-CARLO ORACLE: finite-sample-matched ESS reference. Draw the SAME number of on-support
        #      samples the policy got (N_on) directly from the target q, many times, compute empirical ess
        #      the identical way; report median + 95% band. Corrects the analytic ess_pstar (infinite-sample)
        #      which unfairly penalizes the policy for finite-sampling a perfect sampler also suffers. ----
        _N_on = int(counts.sum())
        if _N_on > 0:
            _rng = np.random.default_rng(0)
            _oc = _rng.multinomial(_N_on, q, size=2000).astype(float)       # [2000, n_enum]
            _ph = _oc / _N_on
            _ess_mc = 1.0 / np.sum(_ph ** 2, axis=1)                         # empirical ess of a PERFECT sampler @ N_on
            _ess_mc_med = float(np.median(_ess_mc)); _ess_mc_lo = float(np.percentile(_ess_mc, 2.5)); _ess_mc_hi = float(np.percentile(_ess_mc, 97.5))
            _essq_now = (1.0 / float(np.sum(p[p > 0] ** 2))) if p.sum() > 0 else float('nan')
            _essq_pctile = float(np.mean(_ess_mc <= _essq_now)) if _essq_now == _essq_now else float('nan')  # where policy falls in the oracle dist
        else:
            _ess_mc_med = _ess_mc_lo = _ess_mc_hi = _essq_pctile = float('nan')
        emp_freq = counts[sampled_mask] / max(1.0, counts.sum())
        sp_all = _spearman(counts, score)                                  # over ALL enum sets (ties@0 handled)
        sp_log = _spearman(np.log(emp_freq + 1e-12), score[sampled_mask])  # over sampled>=1 sets
        pe_log = _pearson(np.log(emp_freq + 1e-12), score[sampled_mask])
        order = np.argsort(-score)
        cov10 = float(np.mean(counts[order[:min(10, len(order))]] > 0))
        top25 = max(1, int(round(0.25 * len(order))))
        cov25 = float(np.mean(counts[order[:top25]] > 0))
        rows.append(dict(run=run, logz=meta['logz'], head=meta['head'], infonce=meta['beta_contrast'] > 0,
                         beta=meta['beta'], beta_contrast=meta['beta_contrast'], fate=name, source=source,
                         target=target, pool_m=pool_m, enum_k=','.join(map(str, sorted(keep))),
                         n_enum_sets=len(enum_local), n_samples=len(samples), n_distinct_on=int(sampled_mask.sum()),
                         on_support_frac=round(on_frac, 4), mean_sampled_k=round(float(np.mean(sampled_sizes)), 3),
                         spearman_count_vs_score=round(sp_all, 4), spearman_logfreq=round(sp_log, 4),
                         pearson_logfreq=round(pe_log, 4), l1=round(float(np.abs(p - q).sum()), 4),
                         jsd_bits=round(_jsd_bits(p, q), 4), mode_cov_top10=round(cov10, 4),
                         mode_cov_top25pct=round(cov25, 4), phi_max_enum=round(float(phis.max()), 4),
                         phi_argmax_sampled=int(counts[int(np.argmax(phis))] > 0),
                         ess_pstar=round(1.0 / float(np.sum(q ** 2)), 2),
                         ess_q=(round(1.0 / float(np.sum(p[p > 0] ** 2)), 2) if p.sum() > 0 else float('nan')),
                         reward_mass_covered=round(float(q[sampled_mask].sum()), 4),
                         ess_mc_median=round(_ess_mc_med, 2), ess_mc_lo=round(_ess_mc_lo, 2), ess_mc_hi=round(_ess_mc_hi, 2),
                         essq_oracle_pctile=round(_essq_pctile, 3), n_on_support=int(_N_on),
                         logZ_star=round(logZ_star, 4), n_drivers=_nd,
                         oracle_pool_cov=round(_pool_cov, 4) if _nd else float('nan'),
                         oracle_recall5=round(_orec(5), 4), oracle_recall10=round(_orec(10), 4),
                         oracle_recall20=round(_orec(20), 4)))
    # beam contrast: beam returns ONE set => distribution coverage ~ 1/n_enum (the anti-sampler baseline)
    try:
        beam_pick = set(beam_planner(li)); beam_local = frozenset(g2l[g] for g in beam_pick if g in g2l)
        beam_on = int(len(beam_local) == len(beam_pick) and len(beam_pick) in keepset and beam_local in key_of)
    except Exception:
        beam_on = -1
    for r in rows:
        r['beam_covers_1set'] = beam_on
        r['beam_dist_coverage'] = round(1.0 / max(1, len(enum_local)), 6)
    return rows


def run_distributional(agent, op, data, env, cand, keys, names, cent, centn, thr, drift, ctrl_recon, A,
                       d, HO, dev, pred_resid, *, run, meta, fates, pool_m0=16, maxenum=2000, cover=0.9,
                       nsamp=5000, beam_planner=None, fix_k=0):
    meta = dict(meta); env._key_for = lambda li: keys[li]
    rows = []
    src_of = lambda n: data.lineage_drivers.get(n, {}).get('source', 'joung')
    for li in fates:
        nm = names[li]; src = 'HSPC' if nm == 'HSPC' else src_of(nm)
        print(f"[distrib] fate={nm} (source={src}, li={li}) pool_m0={pool_m0} maxenum={maxenum} nsamp={nsamp}", flush=True)
        rows.extend(_run_distributional_for_fate(li, nm, src, agent, op, data, env, cand, cent, centn, drift,
                                                 ctrl_recon, A, dev, pred_resid, run=run,
                                                 meta={**meta, 'keys': keys}, pool_m0=pool_m0, maxenum=maxenum,
                                                 cover=cover, nsamp=nsamp, beam_planner=beam_planner, fix_k=fix_k))
    return rows


# ----------------------------------------------------------------------------- PART 2: diversity-at-quality
def run_diversity(agent, op, data, env, cand, keys, names, cent, centn, thr, drift, ctrl_recon, A, d, HO,
                  dev, pred_resid, *, run, meta, beam_planner, beam_topn_planner=None, nsamp=256, jac_sim_cut=0.5, topk=10, seed=0):
    """Per held-out SOURCE: sample the policy nsamp x per fate, score Phi (=env._phi, the reward), and report
    diversity ALWAYS PAIRED WITH a quality bar. The bar is arm-INDEPENDENT (50th/75th pct of the matched
    RANDOM-cocktail Phi for that fate), so `# modes above T75` is comparable across arms. Contrast rows:
    planner in {policy, beam(=~1 mode), random(=diverse but low Phi)}."""
    rng = np.random.default_rng(seed)
    eps0 = data.eps_ctrl_mean.unsqueeze(0).to(dev)
    src_of = lambda n: data.lineage_drivers.get(n, {}).get('source', 'joung')
    per = {}                                                               # source -> planner -> list of per-fate dicts
    for li in HO:
        nm = names[li]; src = 'HSPC' if nm == 'HSPC' else src_of(nm)
        obs = env.reset(keys[li]); s_emb, t_emb = obs[:d], obs[d:]
        kact = int(getattr(env, 'kmax_active', None) or meta['kmax'])
        gate = getattr(env, '_goal_gate', None)
        blk = None
        if gate is not None and li in gate:
            blk = torch.ones(len(cand), dtype=torch.bool, device=dev); blk[list(gate[li])] = False
        pool_g = [int(cand[i]) for i in gate[li]] if (gate is not None and li in gate) else [int(c) for c in cand]

        # ---- policy samples ----
        pol_samp = _sample_policy(agent, s_emb, t_emb, nsamp, cand, allowed_block=blk)
        pol_sp = [(frozenset(s), _phi_of_global_set(op, env, A, eps0, list(s))) for s in pol_samp if s]
        # ---- matched-random samples (k matched per-draw to the policy's sampled sizes) ----
        sizes = [len(s) for s in pol_samp if s] or [min(kact, len(pool_g))]
        rnd_sp = []
        for j in range(nsamp):
            k = int(sizes[j % len(sizes)]); k = max(1, min(k, len(pool_g)))
            s = tuple(int(x) for x in rng.choice(pool_g, size=k, replace=False))
            rnd_sp.append((frozenset(s), _phi_of_global_set(op, env, A, eps0, list(s))))
        # ---- beam (MPC) single pick ----
        try:
            bset = list(beam_planner(li)); bphi = _phi_of_global_set(op, env, A, eps0, bset) if bset else float('nan')
            beam_sp = [(frozenset(bset), bphi)] if bset else []
        except Exception:
            beam_sp = []
        # ---- beam-top-N frontier (fair search-asked-for-N diversity baseline) ----
        beamN_sp = []
        if beam_topn_planner is not None:
            try:
                for _bs in beam_topn_planner(li):
                    _bs = [int(x) for x in _bs]
                    if _bs: beamN_sp.append((frozenset(_bs), _phi_of_global_set(op, env, A, eps0, _bs)))
            except Exception:
                beamN_sp = []

        # arm-INDEPENDENT quality bars from the random Phi distribution
        rphi = np.array([p for _, p in rnd_sp], float)
        T50 = float(np.nanpercentile(rphi, 50)); T75 = float(np.nanpercentile(rphi, 75))

        for planner, sp in [('policy', pol_sp), ('random', rnd_sp), ('beam', beam_sp), ('beam_topN', beamN_sp)]:
            uniq = {}
            for s, p in sp:
                if s not in uniq or p > uniq[s]:
                    uniq[s] = p
            usp = list(uniq.items())
            phis = np.array([p for _, p in usp], float) if usp else np.array([np.nan])
            topk_phi = float(np.mean(np.sort(phis)[::-1][:topk])) if usp else float('nan')
            per.setdefault(src, {}).setdefault(planner, []).append(dict(
                fate=nm, n_samp=len(sp), n_distinct=len(usp),
                n_modes_T50=_count_modes(usp, T50, jac_sim_cut), n_modes_T75=_count_modes(usp, T75, jac_sim_cut),
                topk_phi=topk_phi, mean_phi=float(np.nanmean(phis)),
                frac_above_T75=float(np.mean(phis > T75)) if usp else float('nan'),
                mpd=_mean_pairwise_jac_dist([s for s, _ in usp]), T50=T50, T75=T75))

    rows = []
    for src, planners in per.items():
        for planner, lst in planners.items():
            agg = lambda k: float(np.nanmean([r[k] for r in lst])) if lst else float('nan')
            rows.append(dict(run=run, head=meta['head'], infonce=meta['beta_contrast'] > 0, planner=planner,
                             source=src, n_fates=len(lst), nsamp=nsamp, jac_sim_cut=jac_sim_cut,
                             T50=round(agg('T50'), 4), T75=round(agg('T75'), 4),
                             n_modes_T50=round(agg('n_modes_T50'), 3), n_modes_T75=round(agg('n_modes_T75'), 3),
                             topk_phi=round(agg('topk_phi'), 4), mean_phi=round(agg('mean_phi'), 4),
                             frac_above_T75=round(agg('frac_above_T75'), 4),
                             mean_pairwise_jac_dist=round(agg('mpd'), 4),
                             n_distinct=round(agg('n_distinct'), 1)))
    return rows


# ----------------------------------------------------------------------------- entry point (called by the flag)
def run_from_stack(a, agent, op, data, env, cand, keys, names, cent, centn, thr, drift, ctrl_recon, A, d,
                   HO, TR, dev, pred_resid, mpc_p, beam_topn_planner=None):
    """Called from briggs_multigoal_rl.py --distrib_check (all objects already built = same stack as training)."""
    run = (a.run_name or 'policy').replace('/', '_').replace('E_', '')
    head = 'grn' if getattr(a, 'slot_head', 'mlp') != 'mlp' else 'mlp'
    logz = getattr(a, 'gflownet_logz', 'na') if getattr(a, 'gflownet', False) else ('na' if 'ppo' in run else 'na')
    if '_condz' in run:
        logz = 'cond'
    elif getattr(a, 'gflownet', False):
        logz = getattr(a, 'gflownet_logz', 'shared')
    meta = dict(beta=float(getattr(a, 'gflownet_beta', 48.0)),
                beta_contrast=float(getattr(a, 'beta_contrast', 0.0)),
                contrast_mode=getattr(a, 'contrast_mode', 'infonce'),
                contrast_tau=float(getattr(a, 'contrast_tau', 0.2)),
                head=head, logz=logz, kmax=int(a.kmax),
                driver_pool=bool(getattr(a, 'distrib_driver_pool', False)))
    print(f"[distrib_check] run={run} head={head} logz={logz} beta={meta['beta']} "
          f"beta_contrast={meta['beta_contrast']} | target = exp(beta*(Phi{'+bc*info' if meta['beta_contrast']>0 else ''}))",
          flush=True)
    beam = mpc_p(a.beam, int(a.kmax))                        # beam over the full budget (its deployed setting)

    # enumerable fate(s): default = the always-held-out HSPC (common to every fold/arm) + optionally 1 held-out Cao
    want = [s.strip() for s in getattr(a, 'distrib_fates', 'HSPC').split(',') if s.strip()]
    fates = []
    for w in want:
        if w.lower() in ('allheldout', 'allho', 'all'):
            fates.extend(list(HO))                                          # every held-out fate this fold (matched oracle/shared-vs-cond)
        elif w.upper() == 'HSPC' and 'HSPC' in names:
            fates.append(names.index('HSPC'))
        elif w.lower() == 'firstcao':
            cao = [i for i in HO if names[i] != 'HSPC']
            if cao:
                fates.append(cao[0])
        else:
            hit = [i for i in HO if w.lower() in names[i].lower()]
            if hit:
                fates.append(hit[0])
    fates = list(dict.fromkeys(fates)) or ([names.index('HSPC')] if 'HSPC' in names else HO[:1])

    dist_rows = run_distributional(agent, op, data, env, cand, keys, names, cent, centn, thr, drift, ctrl_recon,
                                   A, d, HO, dev, pred_resid, run=run, meta=meta, fates=fates,
                                   pool_m0=int(getattr(a, 'distrib_pool_m', 16)),
                                   maxenum=int(getattr(a, 'distrib_maxenum', 2000)),
                                   cover=float(getattr(a, 'distrib_cover', 0.9)),
                                   nsamp=int(getattr(a, 'distrib_nsamp', 5000)), beam_planner=beam,
                                   fix_k=int(getattr(a, 'distrib_fix_k', 0)))
    div_rows = run_diversity(agent, op, data, env, cand, keys, names, cent, centn, thr, drift, ctrl_recon, A, d,
                             HO, dev, pred_resid, run=run, meta=meta, beam_planner=beam, beam_topn_planner=beam_topn_planner,
                             nsamp=int(getattr(a, 'distrib_div_nsamp', 256)))
    os.makedirs(OUT_DIR, exist_ok=True)
    dp = f"{OUT_DIR}/gflownet_distribution__{run}.csv"; vp = f"{OUT_DIR}/gflownet_diversity__{run}.csv"
    pd.DataFrame(dist_rows).to_csv(dp, index=False); pd.DataFrame(div_rows).to_csv(vp, index=False)
    print(f"\n[distrib_check] DISTRIBUTIONAL rows ({len(dist_rows)}):\n{pd.DataFrame(dist_rows).to_string(index=False)}", flush=True)
    print(f"\n[distrib_check] DIVERSITY rows ({len(div_rows)}):\n{pd.DataFrame(div_rows).to_string(index=False)}", flush=True)
    print(f"\n[distrib_check] saved -> {dp}\n[distrib_check] saved -> {vp}", flush=True)


def aggregate(out_dir=OUT_DIR):
    """Merge per-run CSVs into the two final files with a `run` column (beam/random reference rows deduped)."""
    for kind in ['distribution', 'diversity']:
        files = sorted(glob.glob(f"{out_dir}/gflownet_{kind}__*.csv"))
        if not files:
            print(f"[aggregate] no per-run files for {kind}"); continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv(f"{out_dir}/gflownet_{kind}.csv", index=False)
        print(f"[aggregate] {kind}: {len(files)} files -> {len(df)} rows -> {out_dir}/gflownet_{kind}.csv")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--out_dir', default=OUT_DIR)
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.out_dir)
