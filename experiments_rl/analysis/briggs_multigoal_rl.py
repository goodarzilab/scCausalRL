"""Multi-goal fate RL — the fix for the single-goal EV=0 degeneracy.

Instead of one fixed HSPC goal (where V(s) can only predict the batch mean -> explained-variance=0 and
PPO is REINFORCE-with-a-useless-baseline), each episode samples a DIFFERENT target fate from Joung's fetal
atlas cell types (neurons, cardiac, trophoblast, erythroblasts, astrocytes, satellite cells, ...). The
SetPolicy/SetCritic are already goal-conditioned (they read z_target), so a varied goal makes V(s,t)
genuinely goal-dependent -> EV>0 -> PPO has a real advantage signal.

Generalization test: HOLD OUT the hematopoietic/erythroid fate (the HSPC-adjacent goal, --heldout_substr),
train every learned planner ONLY on the other fates, then evaluate ZERO-SHOT on the held-out goal — does the
policy recover its atlas drivers (recall@1/@3, driver-hit) and the Briggs HSPC TF sets (P1/P2/known drivers)?

Planners (all goal-conditioned, all logged to W&B):
  MPC (beam)                 per-goal planner reference (searches the held-out goal directly).
  MPC-distill (BC)           BC a SetPolicy on TRAIN-goal MPC picks; eval zero-shot on the held-out goal.
  set-PPO                    from-scratch multi-goal PPO on the TRAIN goals; eval zero-shot.
  BC-init + PPO (4-fix)      BC-init then PPO with the anti-BC-destruction fixes: (1) beta*KL(pi||pi_BC)
                             anchor, (2) target-KL early stop, (3) LR warmup + tiny peak LR + few epochs,
                             (4) tight clip. Diagnostics (EV/approx_kl/clip_frac/kl_bc) logged per update.

NOTE on gamma: the Briggs Mahalanobis support penalty (gamma) lives in the single-goal fate-READOUT env; this
multi-goal env is the lineage-CENTROID (drift-removed DE-match) env, whose pessimism is the DAG-ensemble std
(alpha_pess). gamma does not transfer here. Run via scripts/multigoal_rl.sbatch.
"""
import os, sys, argparse, copy, json
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import add_stack_args, build_stack
from core.scoring import torch_match, Lineage
from core.env import CausalLatentEnv
from analysis.fate_rl import fate_mpc, pred_resid
from rl2_set_ppo.set_ppo import SetPPO
from rl2_set_ppo.set_policy import SetPolicy
from rl2_set_ppo.train import make_planner as planner_set
from core.briggs_fate_reward import FateReward
from core.wblog import wb_init, wb_log, wb_summary, wb_finish

EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
H17 = "/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/analysis/human_dev_annotation"
BRIGGS_SUP = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/Briggs_Supplementary"


def _briggs_library():
    try:
        iso = pd.concat([pd.read_excel(f'{BRIGGS_SUP}/mmc5.xlsx', s)['TF']
                         for s in ['Pool A', 'Pool B', 'Pool C', 'Pool D']]).astype(str)
        return set(iso.str.split('|').str[0].str.strip())
    except Exception:
        return set()


def build_cao_goals(data, op, model, dev, n_goals=43, goal_rep='encode',
                    holdout=('Liver-Hematopoietic stem cells', 'Spleen-Erythroblasts'),
                    joung_labels=None, goal_de='cao_meandiff', ztarget_raw=False, blood_only=False,
                    fate_select='denorm', fate_fdr_q=0.10, fate_fdr_min_drv=1):
    """~n_goals Cao-celltype-centroid goals via the IDENTICAL pipeline for every goal (train + held-out):
    de_lineage = centroid - drift (gene space, the reward target); z_target = scm_forward(encode(centroid), A)
    (encoded cell-state — same construction as z_target_single). Drivers (recall metric) = top cosine TFs.
    Goals = top-n_goals by DE-norm (most distinctive) + the two holdouts (kept for eval, excluded from TR by
    the caller). Cao->encoder scale-mismatch is common-mode across ALL goals; S2 goal-load Jaccard gates it."""
    from core.scoring import Lineage
    cao = pd.read_csv(f'{H17}/cao_type_mean_logexpr.csv', index_col=0)
    cao.columns = cao.columns.astype(str); cao.index = cao.index.astype(str)
    gene_names = list(map(str, data.gene_names)); g2i = data.gene_to_idx
    shared = [g for g in gene_names if g in cao.columns]
    sidx = np.array([g2i[g] for g in shared])
    C = cao[shared].values.astype(np.float32); drift = C.mean(0)
    order = list(np.argsort(-np.linalg.norm(C - drift, axis=1)))               # most-distinctive first
    if joung_labels is not None:                                                # RESTRICT the goal atlas to JOUNG's own annotated fates
        sel = [f for f in joung_labels['fates'] if f in cao.index]
        print(f"[cao-goals] JOUNG-LABEL mode: goal atlas = {len(sel)} Joung-annotated fates (drivers from mmc7 S6A/S6B)", flush=True)
    elif fate_select == 'fdr':                                                  # FDR-REACHABILITY: a fate qualifies iff >= min_drv Joung drivers annotate to it at perm-FDR q<fate_fdr_q (principled, data-set count; replaces arbitrary top-N)
        _ann = pd.read_csv(f'{H17}/joung_human_annotation_celllevel.csv')
        _conf = _ann[_ann['fdr_qval'] < fate_fdr_q]
        _cnt = _conf.groupby('top_celltype').size()
        _keepU = {str(f).upper() for f, n in _cnt.items() if n >= fate_fdr_min_drv}   # case-insensitive match to cao.index
        sel = [cao.index[ci] for ci in order if str(cao.index[ci]).upper() in _keepU]  # keep FDR-confident, ordered by distinctiveness
        print(f"[cao-goals] FDR-reachability: {len(sel)} fates with >={fate_fdr_min_drv} Joung driver(s) at q<{fate_fdr_q} "
              f"(of {len(_cnt)} annotated; vs top-{n_goals} DE-norm)", flush=True)
    else:
        sel = []
        for ci in order:
            if cao.index[ci] not in sel:
                sel.append(cao.index[ci])
            if len(sel) >= n_goals:
                break
    if blood_only:                                          # blood/immune-adjacent companions for HSPC multi-goal
        import re as _re
        _bl = [f for f in cao.index if _re.search(r'eryth|megakary|myeloid|lympho|hematop|HSC|thymocyte|granulo|macrophage|monocyte|platelet|mast|dendritic|microglia', f, _re.I)]
        sel = _bl
        print(f"[companion_blood] atlas = {len(sel)} blood/immune-adjacent Cao fates", flush=True)
    for h in holdout:
        if h in cao.index and h not in sel:
            sel.append(h)
    cos = pd.read_csv(f'{H17}/joung_human_cosine.csv', index_col=0)
    A_ref = op.mean_A()
    # goal_de: how the gene-space REWARD TARGET is defined, in the SAME frame as rv=pred_resid (Joung day-7 Δ-from-hESC).
    #   driversig   = mean of the lineage's Joung single-TF Δ (de_single lfc) -> Joung-native Δ-space (matches rv exactly)
    #   cao_shift   = Cao centroid - Joung hESC control (decoded), one gene space -> hESC reference frame (endpoint still fetal)
    #   cao_meandiff= Cao centroid - Cao cross-type mean (LEGACY: fetal-centroid-from-fetal-mean; cross-space vs rv)
    ctrl_shared = None
    if goal_de == 'cao_shift':
        ctrl_shared = op.decode(data.z0.unsqueeze(0))[0].detach().cpu().numpy()[sidx].astype(np.float32)   # Joung hESC control, gene space
    # Joung train-single drift (SAME as main's cent = de_lineage - drift): the gene z_target must project the
    # DRIFT-REMOVED reward target, else the conditioning channel keeps the common OE component and collapses (STEP0~1.0)
    _trs = getattr(data, 'single_split', {}).get('train', []) if hasattr(data, 'single_split') else []
    jdrift = np.mean([np.asarray(data.de_single[t]['lfc'], np.float32) for t in _trs if t in data.de_single], 0) \
             if _trs else np.zeros(len(gene_names), np.float32)
    data.de_lineage, data.z_target_lineage, data.lineage_drivers, data.lineage_keys = {}, {}, {}, []
    for nm in sel:
        row = cao.loc[nm, shared].values.astype(np.float32)
        if joung_labels is not None:                                          # recall/answer-key = JOUNG's drivers (mmc7), not cosine
            drv = [t for t in joung_labels['drivers'].get(nm, []) if t in g2i]
        else:
            drv = [t for t in (list(cos[nm].sort_values(ascending=False).index[:8]) if nm in cos.columns else []) if t in g2i]
        # ---- gene-space REWARD TARGET (de_full) + disjoint recall answer-key ----
        recall_drv = drv                                                      # default: recall answer-key = all drivers
        if goal_de == 'driversig':
            rd = [t for t in drv if t in data.de_single]
            reward_drv = rd[0::2] or rd                                       # even-rank drivers build the target...
            recall_drv = rd[1::2] or rd                                       # ...odd-rank held out as a DISJOINT recall answer-key
            if reward_drv:
                de_full = np.mean([np.asarray(data.de_single[t]['lfc'], np.float32) for t in reward_drv], 0)
            else:
                de_full = np.zeros(len(gene_names), np.float32)               # no Joung Δ for this lineage's drivers
        elif goal_de == 'cao_shift':
            de_full = np.zeros(len(gene_names), np.float32); de_full[sidx] = row - ctrl_shared
        else:                                                                # cao_meandiff (legacy)
            de_full = np.zeros(len(gene_names), np.float32); de_full[sidx] = row - drift
        data.de_lineage[nm] = data._de_entry(de_full)
        drv_z = [t for t in drv if t in data.z_target_single]
        if goal_rep == 'gene':                                               # gene-space DE -> distinct d-dim embed (JL random proj), matched to de_full
            R = getattr(build_cao_goals, '_R', None)
            if R is None:
                R = torch.tensor((np.random.default_rng(0).standard_normal((op.d, len(sidx))) / np.sqrt(op.d)).astype(np.float32), device=dev)
                build_cao_goals._R = R
            build_cao_goals._sidx = sidx
            _zv = de_full[sidx] if ztarget_raw else (de_full - jdrift)[sidx]    # ztarget_raw=old collapsed channel (with drift); default=drift-removed matched target
            data.z_target_lineage[nm] = R @ torch.tensor(_zv, dtype=torch.float32, device=dev)
        elif goal_rep == 'driver' and drv_z:                                 # in-distribution: mean of driver z_target_single
            data.z_target_lineage[nm] = torch.stack([data.z_target_single[t].to(dev) for t in drv_z]).mean(0)
        else:                                                                # encode the Cao centroid (Stage-B path)
            x = np.zeros(len(gene_names), np.float32); x[sidx] = row
            with torch.no_grad():
                _, _, eps = model.encoder(torch.tensor(x, device=dev).unsqueeze(0))
                data.z_target_lineage[nm] = op.scm_forward(eps, A_ref).squeeze(0)
        data.lineage_drivers[nm] = {'train': recall_drv or [gene_names[0]], 'test': [], 'full': drv}   # 'full' = ALL experimental drivers (richer answer key for TF-recovery; 'train' is the disjoint odd-rank subset)
        data.lineage_keys.append(Lineage(nm))
    print(f"[cao-goals] goal_de={goal_de} (reward target in {'Joung Δ-space' if goal_de=='driversig' else 'hESC-shift space' if goal_de=='cao_shift' else 'LEGACY fetal-meandiff'})", flush=True)
    # ---- STEP 0: goal DISTINCTNESS — pairwise cosine of z_targets (near-1 => collapse => policy can't condition) ----
    Z = torch.stack([data.z_target_lineage[k.name] for k in data.lineage_keys])
    Zn = Z / (Z.norm(dim=1, keepdim=True) + 1e-9)
    cosm = (Zn @ Zn.T)
    off = cosm[~torch.eye(len(Zn), dtype=torch.bool)]
    print(f"[cao-goals] {len(data.lineage_keys)} goals (goal_rep={goal_rep}, top-{n_goals} by DE-norm + {len(holdout)} holdouts) | "
          f"shared {len(shared)}/{len(gene_names)} | holdouts: {[h for h in holdout if h in sel]}", flush=True)
    print(f"[STEP0 goal-distinctness] pairwise z_target cosine mean={off.mean():.3f} std={off.std():.3f} "
          f"(near 1.0 => goals COLLAPSED => policy cannot condition)", flush=True)
    return data.lineage_keys


def add_briggs_fate_goals(data, dev, ann_csv, allk_npz, drift, dual_reach='briggs', goal_rep='gene',
                          min_combos=8, min_spec=0.10, max_fates=50, verbose=True, fate_fdr_q=0.10, source='briggs',
                          skip_sources=()):
    """Unified-fate goals: annotate all-k Briggs combos to Cao fates (same method as Joung), and for each
    Cao fate with >= min_combos CLEAN (non-answer-key) Briggs combos at spec >= min_spec, build a
    Briggs-manifold goal: cent_g = mean drift-removed + drift-orthogonalized Δ of its combos; drivers =
    its combos' most-common TFs (soft-recovery answer key). Answer-key combos already dropped in the CSV.

    dual_reach='briggs': if the fate is ALSO Joung-reached (already in data.lineage_keys), REPLACE its
        cent_g/z_target/drivers with the Briggs (on-manifold) version.
    dual_reach='both':   add it as a SEPARATE goal '<fate>|B' alongside the Joung version (frame-transfer test).
    Returns list of (fate_name, n_combos, replaced?)."""
    from collections import Counter
    from core.scoring import Lineage
    ann = pd.read_csv(ann_csv); ann = ann[ann.is_answerkey == 0]
    dd = np.load(allk_npz, allow_pickle=True); obs = dd['obs_delta'].astype(np.float32)
    labels = ['|'.join(map(str, p)) for p in dd['pairs']]; lab2row = {l: i for i, l in enumerate(labels)}
    _R = getattr(build_cao_goals, '_R', None); _sidx = getattr(build_cao_goals, '_sidx', None)
    dn = drift / (drift.norm() + 1e-9)
    existing = {k.name for k in data.lineage_keys}
    # FDR-reachability (SAME pipeline as Joung: gene-permutation null on cosine-specificity + BH-FDR): a Cao fate
    # qualifies iff >= min_combos Briggs combos annotate to it at perm-FDR q < fate_fdr_q. Falls back to spec>=min_spec
    # only if the annotation CSV predates the FDR column.
    _conf = (ann[ann.fdr_qval < fate_fdr_q] if 'fdr_qval' in ann.columns else ann[ann.specificity >= min_spec])
    _crit = f'q<{fate_fdr_q}' if 'fdr_qval' in ann.columns else f'spec>={min_spec} (NO FDR — old CSV)'
    cnt = _conf.groupby('nearest_fate').size().sort_values(ascending=False)
    cand_fates = [f for f in cnt.index if cnt[f] >= min_combos and 'ematopoietic stem' not in f.lower()][:max_fates]
    added = []
    for fate in cand_fates:
        if fate in existing and data.lineage_drivers.get(fate, {}).get('source') in skip_sources:
            continue                                                    # keep the existing target (e.g. don't clobber a Briggs fate with a Joung-combo one)
        grp = _conf[_conf.nearest_fate == fate]
        rows = [lab2row[l] for l in grp.combo if l in lab2row]
        if not rows: continue
        cg = torch.tensor(obs[rows].mean(0), device=dev) - drift        # mean Δ, drift-removed
        cg = cg - (cg @ dn) * dn                                        # drift-orthogonalized (on-Briggs-manifold, HSPC-style)
        if not hasattr(data, 'fate_std'): data.fate_std = {}
        _fstd = obs[rows].std(0) if len(rows) > 1 else np.ones(obs.shape[1], np.float32)  # per-gene member spread (for --var_weight)
        tfc = Counter(t for s in grp.tfs for t in str(s).split(';'))
        drv = [t for t, _ in tfc.most_common(8)]
        nm = fate if (fate not in existing or dual_reach == 'briggs') else f'{fate}|B'
        if dual_reach == 'both' and fate in existing: nm = f'{fate}|B'
        replaced = (fate in existing and dual_reach == 'briggs')
        data.de_lineage[nm] = data._de_entry((cg + drift).detach().cpu().numpy())   # cent = de_lineage - drift = cg
        if goal_rep == 'gene' and _R is not None:
            data.z_target_lineage[nm] = _R @ cg[_sidx].to(dev)
        else:
            data.z_target_lineage[nm] = cg.clone()
        data.lineage_drivers[nm] = {'train': drv, 'test': [], 'full': drv, 'source': source}
        data.fate_std[nm] = _fstd                                            # per-gene member spread keyed by final fate name (for --var_weight)
        if not hasattr(data, 'answer_combos'): data.answer_combos = {}
        data.answer_combos[nm] = [set(str(s).split(';')) for s in grp.tfs]   # the actual answer COMBO SETS (for set-Jaccard, not just the union of TFs)
        if nm not in existing:
            data.lineage_keys.append(Lineage(nm))
        added.append((nm, len(rows), replaced))
    if verbose:
        nrep = sum(r for _, _, r in added)
        print(f"[unified_fates] {source} fate-goals: +{len(added)} ({nrep} replaced Joung-reached, "
              f"{len(added)-nrep} new) | dual_reach={dual_reach} min_combos={min_combos} select={_crit}", flush=True)
        print(f"[unified_fates] examples: {[(n, c) for n, c, _ in added[:8]]}", flush=True)
    return added


def build_marker_goals(data, dev, op, bundle_dir, drift, goal_rep='gene'):
    """MARKER-signature fate goals (production). Loads the precomputed bundle (step14): per Tier-B fate,
    cent_raw = mean RAW observed Δ (4366) of its marker-argmax members, + drivers / answer_combos / fold / tier.
    Builds the SAME structures as build_cao_goals+add_briggs_fate_goals (de_lineage, z_target_lineage via the
    shared-gene JL, lineage_drivers, answer_combos) so ALL downstream machinery is unchanged. K=0 (no OE removal).
    Sets data.fate_fold[nm]=fold and data.fate_tier[nm]=tier for the fold-based held-out split."""
    import json as _json
    from core.scoring import Lineage
    b = np.load(f'{bundle_dir}/fate_targets_bundle.npz', allow_pickle=True)
    meta = _json.load(open(f'{bundle_dir}/fate_targets_meta.json'))
    sidx = [int(i) for i in b['sidx']]
    R = torch.tensor((np.random.default_rng(0).standard_normal((op.d, len(sidx))) / np.sqrt(op.d)).astype(np.float32), device=dev)
    build_cao_goals._R = R; build_cao_goals._sidx = sidx                          # HSPC + all goals share this JL channel
    dn = drift / (drift.norm() + 1e-9)
    if not hasattr(data, 'answer_combos'): data.answer_combos = {}
    data.fate_fold, data.fate_tier = {}, {}
    cent_raw = torch.tensor(np.asarray(b['cent'], np.float32), device=dev)         # [F,4366] raw observed Δ centroid
    for i, m in enumerate(meta):
        nm = m.get('key', m['fate'])                                              # source-qualified unique key (avoids cross-source name clobber)
        cg = cent_raw[i] - drift; cg = cg - (cg @ dn) * dn                         # drift-removed + drift-orthogonalized (on-manifold)
        data.de_lineage[nm] = data._de_entry((cg + drift).detach().cpu().numpy())  # cent = de_lineage - drift = cg
        data.z_target_lineage[nm] = R @ cg[sidx] if goal_rep == 'gene' else cg.clone()
        drv = [t for t in m['drivers'] if t in data.gene_to_idx]
        data.lineage_drivers[nm] = {'train': drv or [data.gene_names[0]], 'test': [], 'full': drv, 'source': m['source']}
        data.answer_combos[nm] = [set(s.split(';')) for s in m['answer_combos']]
        data.fate_fold[nm] = int(m['fold']); data.fate_tier[nm] = m['tier']
        data.lineage_keys.append(Lineage(nm))
    Z = torch.stack([data.z_target_lineage[k.name] for k in data.lineage_keys]); Zn = Z / (Z.norm(dim=1, keepdim=True) + 1e-9)
    off = (Zn @ Zn.T)[~torch.eye(len(Zn), dtype=torch.bool)]
    from collections import Counter as _C
    print(f"[marker-goals] {len(meta)} fates from {bundle_dir} | by source {dict(_C(m['source'] for m in meta))} "
          f"| Tier-A {sum(m['tier']=='A' for m in meta)} | folds {dict(sorted(_C(m['fold'] for m in meta).items()))}", flush=True)
    print(f"[marker-goals] K=0 (no OE removal) | STEP0 goal-distinctness pairwise z cos mean={off.mean():.3f} std={off.std():.3f}", flush=True)
    return data.lineage_keys


# Canonical HSC/HSPC PROGRAM genes (definitional, up in HSPC) — the non-gameable readout panel. Master TFs +
# surface/program markers; at eval we RESTRICT to those NOT in the candidate pool, so a policy cannot trivially
# self-satisfy the score by over-expressing a marker TF (ZNF680 or any single off-target can't induce the whole program).
HSPC_MARKERS = {'RUNX1', 'GATA2', 'TAL1', 'LMO2', 'LYL1', 'HHEX', 'MEIS1', 'HOXA9', 'HOXA10', 'HOXA7', 'ERG',
                'FLI1', 'GFI1B', 'SPI1', 'MECOM', 'MYB', 'GATA1', 'KLF1', 'IKZF1', 'MLLT3', 'HLF', 'CD34',
                'KIT', 'SPINK2', 'MEF2C', 'NFE2', 'ETV6', 'RUNX3', 'LDB1', 'CBFA2T3', 'ZFPM1'}
# Briggs hESC->HSPC TF sets (the held-out fate's "answer key" — recovery target at eval)
P1 = {'TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2'}
P2 = {'DDIT3', 'MZF1', 'HOXB3', 'MIER1', 'SIRT6', 'ZNF85'}
DRIVERS = {'ETV2', 'GATA2', 'RUNX1', 'LMO2', 'TAL1', 'LYL1', 'GFI1B', 'SPI1', 'FLI1', 'ERG', 'MEIS1',
           'HOXA9', 'GATA1', 'MYB', 'HHEX', 'RUNX3', 'IKZF1', 'CEBPA', 'KLF1', 'MECOM', 'LMO4'}
# Published hPSC->HSPC directed-differentiation TF cocktails (literature baselines to beat)
LIT_COCKTAILS = {
    'Sugimura2017 (7)': ['ERG', 'HOXA5', 'HOXA9', 'HOXA10', 'LCOR', 'RUNX1', 'SPI1'],
    'Sandler/Lis (4)': ['FOSB', 'GFI1', 'RUNX1', 'SPI1'],   # endothelial->HSC (Sandler'14 / Lis'17)
}


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--pool', default='curated'); ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--kmax', type=int, default=2, help='cocktail size for the LEARNED planners + HSPC MPC (high-k proposal). Amortization recall@k over fates stays k2.')
    ap.add_argument('--maha_exclude_tfs', default='', help='leave-hit-out: drop these TFs from the Joung support manifold so recovering them is a real generalization (leak-free)')
    ap.add_argument('--qmax', type=float, default=0.15); ap.add_argument('--min_drivers', type=int, default=2)
    ap.add_argument('--lineage_csv', default=f'{EXP20}/results/briggs/joung_tf_celltype.csv')
    ap.add_argument('--heldout_frac', type=float, default=0.0, help='hold out a RANDOM fraction of goals (n>1 generalization estimate); 0=use --heldout_substr')
    ap.add_argument('--heldout_substr', default='Erythro',
                    help='hold out lineages whose name contains this (the HSPC-adjacent fate); eval zero-shot here')
    ap.add_argument('--iters', type=int, default=5000); ap.add_argument('--n_dag', type=int, default=8)
    ap.add_argument('--bc_epochs', type=int, default=400); ap.add_argument('--device', default=None)
    # --- the 4 anti-BC-destruction fixes (applied to BC-init+PPO) ---
    ap.add_argument('--kl_bc', type=float, default=0.5, help='(1) beta*KL(pi||pi_BC) anchor to the BC policy')
    ap.add_argument('--target_kl', type=float, default=0.02, help='(2) break the epoch loop when approx-KL exceeds this')
    ap.add_argument('--warmup', type=int, default=300, help='(3) linear LR warmup over this many updates')
    ap.add_argument('--ft_lr', type=float, default=1e-5, help='(3) tiny peak actor LR')
    ap.add_argument('--ft_epochs', type=int, default=2, help='(3) few epochs per batch')
    ap.add_argument('--clip', type=float, default=0.1, help='(4) tight PPO clip epsilon')
    ap.add_argument('--ent_coeff', type=float, default=0.02)
    ap.add_argument('--ent_anneal_to', type=float, default=-1.0, help='(--hstep) linearly anneal the entropy coeff from --ent_coeff to this value over training (>=0 to enable). Forces late-training commitment on flat landscapes — diagnostic for whether full-280 seed variance is entropy-driven.')
    # --- claim 2: zero-shot HSPC goal (Briggs, cross-dataset) ---
    ap.add_argument('--readout_briggs', default=f'{EXP20}/results/briggs/readout_ridge_v2b_prod_deg8_all40_rho_briggs_holdhits.npz',
                    help='Briggs fate-readout (latent->fate) whose direction defines the zero-shot HSPC goal; holdhits=leak-free vs P1/P2')
    ap.add_argument('--hspc_scale', type=float, default=0.0,
                    help='||z_target_hspc - z0|| along the readout direction; 0=auto (mean train-centroid displacement)')
    ap.add_argument('--support_feats', default=f'{EXP20}/results/briggs/readout_feats_v2b_prod_deg8_all40_ans_joungctrl.npz',
                    help='observed-Briggs-combo latent features (on-manifold support) for the Mahalanobis off-manifold flag')
    # --- Stage B: goal source + attribution axis ---
    ap.add_argument('--goal_source', default='joung', choices=['joung', 'cao'],
                    help="cao = ~n_cao_goals Cao-celltype-centroid goals (Liver-HSC + Spleen-Erythroblasts held out)")
    ap.add_argument('--joung_labels', default=None,
                    help="(cao goal_source) restrict the goal atlas to JOUNG's OWN annotated fates + use JOUNG's driver TFs "
                         "(mmc7 S6A/S6B) as the recall answer-key, from results/briggs/joung_vs_cosine_labels.json")
    ap.add_argument('--joung_reach_thr', type=float, default=-9.0,
                    help="keep only Joung fates with observed-reachability obs_joung >= this (default -9 = all Joung-annotated fates)")
    ap.add_argument('--n_cao_goals', type=int, default=43)
    ap.add_argument('--fate_select', default='denorm', choices=['denorm', 'fdr'], help="how to pick which Cao types are fates: 'denorm'=top-n_cao_goals by distinctiveness (arbitrary count); 'fdr'=data-set — every Cao type with >=fate_fdr_min_drv Joung drivers at perm-FDR q<fate_fdr_q (confidently reachable)")
    ap.add_argument('--fate_fdr_q', type=float, default=0.10); ap.add_argument('--fate_fdr_min_drv', type=int, default=1)
    ap.add_argument('--goal_rep', default='encode', choices=['encode', 'driver', 'gene'],
                    help="goal channel: encode(Cao centroid) & driver-TF centroid are both LATENT (Step-0 showed "
                         "collinear/collapsed); gene = random-projection of the gene-space DE centroid to d dims "
                         "(distinct by Johnson-Lindenstrauss => the only rep that lets the policy distinguish goals)")
    ap.add_argument('--goal_de', default='cao_meandiff', choices=['cao_meandiff', 'driversig', 'cao_shift'],
                    help="gene-space REWARD TARGET frame (must match rv=pred_resid, a Joung day-7 Δ-from-hESC): "
                         "driversig=mean Joung single-TF Δ of the lineage's drivers (Joung-native Δ-space; reward/recall drivers kept disjoint); "
                         "cao_shift=Cao centroid - Joung hESC control (hESC reference frame); "
                         "cao_meandiff=LEGACY Cao centroid - Cao cross-type mean (cross-space vs rv => the mismatch)")
    ap.add_argument('--ztarget_raw', action='store_true',
                    help="CONTROL: project the RAW de_full (with drift) into the goal channel = the OLD collapsed z_target (STEP0~0.95); default projects the drift-removed distinct target")
    ap.add_argument('--s_int', default='on', choices=['on', 'off'], help='Axis 3: disable the interaction head (attribution)')
    ap.add_argument('--beta_maha', type=float, default=0.0, help='Mahalanobis support penalty weight (to the Joung-perturbation manifold)')
    ap.add_argument('--maha_tau', type=float, default=0.0, help='hinge threshold: penalty = beta*max(0, maha - tau)')
    ap.add_argument('--beta_contrast', type=float, default=0.0, help='(b) contrastive reward: + beta*(cos(rv,goal) - mean_k cos(rv,goal_k)) -> rewards reaching THIS fate specifically (rank-1/pass@N objective)')
    ap.add_argument('--contrast_separate', action='store_true', help='(--hstep) apply the specificity signal as a SEPARATE policy-gradient term with its own per-goal EMA baseline (like the PPO entropy bonus) instead of adding it into the reward — avoids critic-absorption + advantage-cancellation of the near-constant penalty that made in-reward contrast inert')
    ap.add_argument('--alpha', type=float, default=1.0, help='DAG-std pessimism weight in the resid reward (lower => less negative reward)')
    ap.add_argument('--reward_scale', type=float, default=1.0, help='scale the resid reward (denser gradient)')
    ap.add_argument('--cvar_tau', type=float, default=0.0, help='best-of-N / CVaR-upper for from-scratch set-PPO: advantage vs this HIGH reward quantile (e.g. 0.75) instead of the critic (risk-seeking; attacks the "never sample a peak" collapse)')
    ap.add_argument('--best_of_n', type=int, default=1, help='PREDICTIVE-UNCERTAINTY objective: sample N cocktails PER GOAL per visit; with --cvar_pergoal the advantage rewards the top quantile of THOSE N (risk-seeking best-of-N). Total env steps held ~= --iters (fewer goal-visits).')
    ap.add_argument('--cvar_pergoal', action='store_true', help='compute the CVaR-upper quantile WITHIN each goal\'s best_of_n samples (true per-goal best-of-N) instead of over the mixed batch')
    ap.add_argument('--rloo', action='store_true', help='RLOO/GRPO advantage: A_i = R_i - leave-one-out mean of its best_of_n group (drops the value/critic baseline). Natural for SINGLE-goal (--train_goal_hspc): sample G cocktails/step, baseline = the other G-1.')
    ap.add_argument('--cvar_anneal', type=int, default=0, help='ramp the advantage from R-V (critic) to CVaR over this many update() calls (~iters/128). Warm-starts the policy so per-group CVaR can learn instead of stagnating from a uniform cold start.')
    ap.add_argument('--cvar_reward_alpha', type=float, default=0.0, help='REWARD-level CVaR-lower over the N-DAG (n_dag) outcome distribution: reward = mean of the worst ceil(alpha*N) per-DAG matches − γ·Maha. REPLACES the −α·std pessimism (set --alpha 0). Use --n_dag>=64. This is model-uncertainty pessimism, distinct from the advantage-level --cvar_tau.')
    ap.add_argument('--cvar_loo', action='store_true', help='per-goal CVaR (--cvar_pergoal): compute the within-goal quantile LEAVE-ONE-OUT (over the other G-1 samples) — the unbiased GRPO-with-a-quantile-baseline form.')
    ap.add_argument('--contrast_mode', default='mean', choices=['mean', 'max', 'infonce', 'focus'], help="contrastive reward: mean=above-avg; max=margin to top competitor; infonce=InfoNCE over fates (hard-negative = nearest wrong fate, in-batch) => rank-1/pass@N pressure; focus=(H-step only) penalize when the cocktail's best-matching fate != the conditioned fate.")
    ap.add_argument('--contrast_tau', type=float, default=0.2, help='InfoNCE temperature (--contrast_mode infonce)')
    ap.add_argument('--adv_norm_pergoal', action='store_true', help='normalize PPO advantages WITHIN each goal (PopArt-lite) so low-Phi-range goals still contribute gradient')
    ap.add_argument('--prioritized_goals', action='store_true', help='sample training goals by current regret (low running reward => sampled more) so compute goes to improvable goals')
    ap.add_argument('--hstep', action='store_true', help='#1: promote the one-step set-bandit to an H-STEP MDP (each token a step; z_t=z_post(S<=t) cumulative re-solve; GAE(lambda); STOP; per-step pessimism; per-token PPO). Strict generalization of the bandit (H=1).')
    ap.add_argument('--gae_lambda', type=float, default=0.95, help='(--hstep) GAE lambda')
    ap.add_argument('--causal_gate', action='store_true', help='#2: restrict per-goal candidate set to DAG-ANCESTORS of the goal DEG-programs (reachable under A). Ablation = A_gap probe: drv_recall preserved => A load-bearing.')
    ap.add_argument('--cosine_gate', action='store_true', help='ANSWER-AGNOSTIC retriever (cheap): per-goal shortlist = top cosine_gate_keep candidates by SINGLE-TF Φ to the goal (drivers are high-cosine by construction => reproduces driver_pool WITHOUT labels). Applied at train AND eval (policy._allowed_block); null drawn from the same per-goal shortlist. CIRCULAR for drv_recall (annotation IS cosine) => judge on reachability/Jaccard/HSPC.')
    ap.add_argument('--cosine_gate_keep', type=float, default=0.25, help='(--cosine_gate) fraction of candidates kept per goal')
    ap.add_argument('--learned_retriever', action='store_true', help='LEARNED goal-conditioned GNN retriever: train on TRAIN-fate drivers, predict a per-goal shortlist for held-out fates (answer-agnostic). Deployable version of driver_pool. Applied at train+eval; null pool-matched to the shortlist. HO drv_recall IS valid here (retriever never saw HO labels).')
    ap.add_argument('--retriever_epochs', type=int, default=300, help='(--learned_retriever) supervised training epochs')
    ap.add_argument('--retriever_keep', type=float, default=0.25, help='(--learned_retriever) fraction of candidates in the per-goal shortlist')
    ap.add_argument('--retriever_hops', type=int, default=2, help='(--learned_retriever) message-passing hops over the candidate GRN')
    ap.add_argument('--gen_eval_n', type=int, default=64, help='N samples per goal in the generative best-of-N eval (pass@N/resid_best). Larger N => more chances for best-of-N to hit rank-1 (raises pass@N).')
    ap.add_argument('--train_goal_hspc', action='store_true', help='SINGLE-goal RL: train the policy directly on the HSPC (Briggs) goal (not the multi-goal Joung atlas). High-k cocktail recovery of P1/P2. Eval = HSPC recovery.')
    ap.add_argument('--hspc_multigoal', action='store_true', help='(with --train_goal_hspc) put HSPC into the Cao atlas as ONE of many TRAIN goals (multi-goal) so V(s,t) is goal-dependent (EV>0) and FROM-SCRATCH set-PPO is viable — vs the single-goal EV=0 degeneracy.')
    ap.add_argument('--hspc_heldout', action='store_true', help='(with --train_goal_hspc) HSPC is ALWAYS HELD OUT (zero-shot eval); train on the intermixed Cao/Joung fates (random heldout_frac split), never on HSPC. The true generalization test.')
    ap.add_argument('--hspc_ortho_drift', action='store_true', help='(with --train_goal_hspc) orthogonalize o_hspc against the Joung drift so the goal rewards HSPC-SPECIFIC genes, not the generic OE response — attacks the drift-dominated ceiling (control ~optimal).')
    ap.add_argument('--hspc_obs_de', default='', help='OBSERVED Briggs hit-combo mean Δ (npz with obs_delta [n,G]) as the HSPC goal target, instead of the linearized readout gradient — a real endpoint, less drift-dominated.')
    ap.add_argument('--hspc_answerkey', action='store_true', help='ANSWER-KEY-GROUNDED HSPC target: o_hspc = mean OBSERVED Δ of the P1/P2 answer-key combos (from briggs_combo_delta_allk + annotation is_answerkey), a full drift-removed HSPC transcriptome — NON-gameable (a linear-readout maximizer like ZNF680 cannot match the whole program) and grounded in the real validated combos. Target is observed DATA, so it works with HSPC held out (Rung A).')
    ap.add_argument('--companion_blood', action='store_true', help='(--hspc_multigoal) use blood/immune-adjacent Cao fates (erythroid/mega/myeloid/lymphoid/HSC/thymocyte/...) as the companion goals, so HSPC conditioning has NEIGHBOURS instead of unrelated fetal fates.')
    ap.add_argument('--hspc_modality', type=int, default=-1, help='(--train_goal_hspc) re-encode the control in this modality token (0=Joung, 1=Briggs) so the rollout START is in the Briggs frame. Encoder-side only (no modality_shift => the TF do-op is modality-agnostic).')
    ap.add_argument('--reward_mode', default='resid', choices=['resid', 'raw', 'blend'], help='resid=drift-removed match (default); raw=match on the RAW Δ (drift kept — may carry fate info); blend=0.5 each')
    ap.add_argument('--reward_match', default='corr', choices=['corr', 'proj'], help="TRAINING reward match: "
                    "'corr'=direction-only Pearson (original); 'proj'=magnitude-aware projection (the REPROcode "
                    "fix — rewards reaching far, not just aligning). Eval metrics stay 'corr' for comparability.")
    ap.add_argument('--seed', type=int, default=0, help='training seed (for top-k seed-stability diagnostic)')
    ap.add_argument('--wandb', action='store_true'); ap.add_argument('--run_name', default=None)
    ap.add_argument('--save_agent', default=None, help='Goal-1: save the atlas-wide goal-conditioned policy artifact (.pt)')
    ap.add_argument('--eval_only', action='store_true', help='skip all training; load --load_policy and run the policy/MPC/greedy recall ladder + per-goal timing + gen_eval on BOTH the in-domain(train) and held-out fate sets, dump a CSV, exit')
    ap.add_argument('--load_policy', default=None, help='(with --eval_only) path to a saved SetPolicy state_dict (.pt) to evaluate')
    ap.add_argument('--distrib_check', action='store_true', help='(--eval_only, DEFAULT OFF) GFlowNet distributional-correspondence (P(x)~exp(beta*Phi) on an enumerable small-k sub-pool: Spearman/Pearson log-freq vs beta*Phi, L1, JSD, mode coverage) + diversity-at-quality (modes above a Phi bar, top-K Phi, pairwise-Jaccard; policy vs beam vs random). Reuses the built stack; writes results/rl_eval_condz/gflownet_{distribution,diversity}__<run>.csv. Does NOT change default eval behaviour.')
    ap.add_argument('--distrib_fates', default='HSPC', help='(--distrib_check) comma list of enumerable fate selectors: HSPC (always-held-out, common to all folds/arms), firstcao (first held-out Cao fate), or a name substring')
    ap.add_argument('--distrib_pool_m', type=int, default=16, help='(--distrib_check) starting sub-pool size M0 (top-M0 candidates by single-TF training-Phi; WM-derived => arm-independent). The pool is auto-shrunk so the enumerated support stays <= --distrib_maxenum.')
    ap.add_argument('--distrib_maxenum', type=int, default=2000, help='(--distrib_check) cap on enumerated sets; pool_m is chosen as the largest m<=M0 with sum_k C(m,k) over the policy-sampled sizes <= this')
    ap.add_argument('--distrib_cover', type=float, default=0.9, help='(--distrib_check) enumerate the smallest set of policy-sampled cocktail SIZES covering >= this fraction of samples (the beta=48 policy uses ~full k=kmax budget, so this adapts the enumeration to where the mass actually is)')
    ap.add_argument('--distrib_kcap', type=int, default=4, help='(--distrib_check) DEPRECATED (size now auto-detected); kept for backward-compat')
    ap.add_argument('--distrib_fix_k', type=int, default=0, help='(--distrib_check) >0 => FORCE the enumerable support to exactly size-k subsets over the fixed top-pool (skips the adaptive size probe) so q / ESS(p*) are byte-identical across arms for a clean cond-vs-shared comparison')
    ap.add_argument('--distrib_beam_topn', type=int, default=0, help='(--distrib_check) >0 => add a BEAM-TOP-N diversity baseline: beam search with width N returns its N-set frontier (fair search-asked-for-N-candidates comparison vs the sampler) scored on the same diversity-at-quality metrics.')
    ap.add_argument('--distrib_driver_pool', action='store_true', help='(--distrib_check) force the fate answer-key drivers INTO the enumerable pool (else pool=top-single-Phi TFs, which excludes low-single-Phi combination drivers) => oracle target-marginal recall is not capped by pool membership; the true reward ceiling for driver recovery')
    ap.add_argument('--distrib_nsamp', type=int, default=5000, help='(--distrib_check) policy samples per fate for the empirical set-frequency')
    ap.add_argument('--distrib_div_nsamp', type=int, default=256, help='(--distrib_check) policy samples per held-out fate for the diversity-at-quality table')
    ap.add_argument('--save_scratch', default=None, help='save the FROM-SCRATCH set-PPO policy (ag_s) state_dict here (the diverse-explorer sibling; --save_agent only saves the BC-init+PPO ag_b). Works for both bandit and --hstep.')
    ap.add_argument('--save_ladder', action='store_true', help='(with --save_scratch, GFlowNet --hstep) ALSO retain a per-iter checkpoint ladder {stem}_it{k}k.pt at each periodic save (dense 50k early, 100k late) for early-stopping / diversity-collapse sweeps, instead of only the rolling file.')
    ap.add_argument('--unified_fates', action='store_true', help='UNIFIED Cao fate space: add Briggs-combo-derived Cao-fate goals (all-k, answer-key held out) so the policy trains on ON-BRIGGS-MANIFOLD targets alongside the Joung/driversig fates. cent_g for a Briggs fate = mean drift-orthogonalized Δ of its combos.')
    ap.add_argument('--dual_reach', default='briggs', choices=['briggs', 'both'], help="fork for fates reachable by BOTH Joung and Briggs: 'briggs' replaces cent_g with the on-manifold Briggs version; 'both' keeps a separate '<fate>|B' goal (frame-transfer test)")
    ap.add_argument('--briggs_ann', default='/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs/briggs_combo_cao_annotation.csv')
    ap.add_argument('--marker_targets', default='', help='PRODUCTION marker-signature fate goals: load the step14 bundle dir (fate_targets_bundle.npz + fate_targets_meta.json). REPLACES build_cao_goals + unified_fates; K=0 (no OE removal). Fate space = 84 Tier-B marker fates + dedicated HSPC.')
    ap.add_argument('--fold', type=int, default=-99, help='(--marker_targets) held-out = fates with fate_fold==this (+ fold -1 always, incl. Liver-HSC). -99 => fall back to --heldout_frac/substr.')
    ap.add_argument('--briggs_allk', default='/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs/briggs_combo_delta_allk.npz')
    ap.add_argument('--briggs_min_combos', type=int, default=8); ap.add_argument('--briggs_min_spec', type=float, default=0.10)
    ap.add_argument('--briggs_max_fates', type=int, default=50)
    ap.add_argument('--hstep_cvar_frac', type=float, default=0.0, help='(--hstep) >0 => reward-level CVaR-LOWER pessimism over the N-DAG ensemble (mean of the worst frac); 0 => symmetric --alpha·std pessimism')
    ap.add_argument('--tf_cost', type=float, default=0.0, help='(--hstep) PER-TF cost: subtract this λ at EACH non-STOP step => terminal −λ|S| parsimony penalty. Anti-breadth-inflation (sharpens freq-ranked recall/SOFT-disc) + lab feasibility. Keep SMALL & sweep {0,0.01,0.03,0.1}; large λ collapses to k=1 and biases against large-combo (Briggs/HSPC) fates. Reported in the per-term inert-lever diagnostic.')
    ap.add_argument('--slot_head', default='mlp', choices=['mlp', 'grn_latent', 'grn_gene'], help='actor slot head: mlp=Linear(256->V) (default); grn_latent=message-pass per-slot scores k hops over the WM latent graph A^T (357 GO-program factors) then SENA-decode to candidate logits; grn_gene=message-pass directly on the candidate-gene GRN=D·A^T·D^T. Injects the WM causal mechanism as an action-space prior (A is load-bearing under the corrected A_gap metric).')
    ap.add_argument('--mp_hops', type=int, default=0, help='(--slot_head grn_*) # message-passing hops k over A^T (Σ_{h≤k}). Sweep {1,2,3}.')
    ap.add_argument('--mp_gate', default='gated', choices=['gated', 'none'], help="(--slot_head grn_*) 'gated'=MLP base + learnable beta·MP (residual, safe); 'none'=pure MP replaces the head.")
    # ── reward-space fixes (debugging root cause: direction-only + OE-residual-dominated reward => k=1-2 + cross-fate mode collapse) ──
    ap.add_argument('--oe_remove_target', type=int, default=0, help='>0 => project each fate reward-target (cent) off the top-K shared PCs of the fate-target matrix (the OE/proliferation residual that makes all fates collinear). Makes the reward FATE-SPECIFIC (specificity space), breaking the cross-fate mode collapse. K~4 matches the annotation drift_pcs.')
    ap.add_argument('--gamma', type=float, default=0.99, help='(--hstep) discount γ. Episodes are ≤kmax steps; γ<1 turns a plateaued potential into an implicit per-step STOP-cost (−(1−γ)Φ). Set 1.0 to remove the length penalty (lets cocktails grow).')
    ap.add_argument('--k_floor', type=int, default=1, help='(--hstep) forbid STOP before this slot => minimum cocktail size. k_floor=1 = current (STOP only forbidden at slot 0); k_floor=3 forces ≥3 TFs. Blunt lever to counter the direction-Φ-saturation collapse to k=1.')
    ap.add_argument('--driver_pool', action='store_true', help='restrict candidate pool to the GLOBAL union of all fates answer drivers (~95-150 TFs) — contains every answer, same pool for all goals (non-leaky), a valid driver-preserving alternative to the broken causal_gate.')
    ap.add_argument('--gate_audit', action='store_true', help='audit: print answer-driver retention of causal_gate then exit')
    ap.add_argument('--causal_gate_keep', type=float, default=0.5, help='(--causal_gate) fraction of TFs kept per goal by SCM-reachability; 0.5=~140/280, 0.2=~56 (smaller pool => smaller needle-haystack). Works for both PPO and GFlowNet now.')
    ap.add_argument('--phi_hinge', type=float, default=0.0, help='(--hstep) DE-SHALLOW reward: transform potential Φ -> max(0, Φ-T) before shaping, zeroing the weakly-aligned shallow region so only above-threshold (near-driver) cocktails earn reward. Needs exploration help (causal_gate/small pool) or it starves gradient.')
    ap.add_argument('--phi_pow', type=float, default=0.0, help='(--hstep) DE-SHALLOW reward: transform Φ -> sign(Φ)|Φ|^p before shaping; p>1 steepens toward the high-Φ driver peak while keeping gradient everywhere (softer than --phi_hinge).')
    ap.add_argument('--mask_stop', action='store_true', help='(--hstep) MASK the STOP token at every slot => the policy is forced to emit EXACTLY kmax_active TFs (per-source cap, or per-fate answer-k with --force_k_mode answer). Decouples "how many" (fixed budget) from "which" (content), removing the STOP-too-early collapse; the matched-k null keeps the Δ honest.')
    ap.add_argument('--force_k_mode', default='source', choices=['source', 'answer'], help="(--mask_stop) 'source' => forced k = per-source kmax cap (Joung=kmax_joung, Briggs/HSPC=kmax). 'answer' => forced k = each fate's MEDIAN answer-combo breadth (clamped [1,kmax]); fates with no combo answer keep the per-source cap. 'answer' is the clean content-selection diagnostic (oracle breadth).")
    ap.add_argument('--var_weight', action='store_true', help='PRECISION-weighted reward: weight the per-gene match by 1/(per-gene variance across the fate members) so the reward focuses on CONSISTENT fate-defining genes and ignores member-idiosyncratic noise (models the fate VARIANCE, not just the centroid). Complementary to --oe_remove_target.')
    ap.add_argument('--hstep_sil', type=float, default=0.0, help='(--hstep) >0 => SELF-IMITATION (Oh et al. 2018): auxiliary MLE loss L_SIL=E[−logπ(a|s)·(R−V)_+]+½(R−V)_+² on a replay buffer of past trajectories, weighted by this coeff. PPO-side analogue of TB off-policy validity; R is the EXACT terminal Φ (frozen WM => no bootstrap staleness).')
    ap.add_argument('--hstep_sil_conservative', action='store_true', help='(--hstep_sil) key the SIL buffer on the CONSERVATIVE return R=min_k Φ^(k) over the DAG ensemble (not raw Φ) => imitate ROBUST high-reward cocktails, pull away from single-DAG gamers (synergistic with the reward-hacking fix).')
    ap.add_argument('--gflownet', action='store_true', help='train the set-planner as a GFlowNet (trajectory-balance) instead of PPO H-step: learns π(cocktail)∝exp(β·Φ) => samples PROPORTIONAL to reward (covers modes, no argmax collapse) + off-policy replay buffer (memory of past cocktails). Reuses the same autoregressive SetPolicy + WM reward.')
    ap.add_argument('--gflownet_beta', type=float, default=6.0, help='(--gflownet) reward exponent: R=exp(beta·Φ); higher => sharper/greedier')
    ap.add_argument('--gflownet_gamma', type=float, default=0.5, help='(--gflownet) fraction of each training batch drawn from the replay buffer (off-policy memory); 0.5 was the GFlowNet-AL sweet spot')
    ap.add_argument('--gflownet_prioritized', action='store_true', help='(--gflownet) reward-PRIORITIZED replay: Φ-weighted (softmax τ=0.1) sampling from the buffer + keep-best eviction (drop lowest-Φ), instead of recency-FIFO + uniform. Retains discovered high-Φ modes (mode-seeking); uses the Φ already stored per trajectory.')
    ap.add_argument('--gflownet_logz', default='cond', choices=['cond', 'shared'], help="(--gflownet) logZ partition-fn: 'cond' MLP on z_goal (can absorb goal-dependence => low specificity); 'shared' single scalar => forces P_F to carry the goal signal (diverse AND precise)")
    ap.add_argument('--gflownet_logz_wd', type=float, default=0.0, help='(--gflownet) weight decay on the logZ head (soft version of forcing P_F to condition)')
    ap.add_argument('--gflownet_pess', type=float, default=0.0, help='(--gflownet) >0 => DAG-pessimistic reward Φ=mean−pess·std over --n_dag sampled DAGs (matches the H-step robustness penalty)')
    ap.add_argument('--conservative_bon', action='store_true', help='(--eval_only) deploy = argmax over N samples of a CONSERVATIVE score (WCO=min_k Φ over the DAG ensemble; UWO=mean−α·std) instead of greedy — report recall@1 of each rule (no retraining)')
    ap.add_argument('--mpc_ksweep', action='store_true', help='(--eval_only) VALUE-PROP: time MPC beam at k=2..6 (growing/until intractable) vs the O(1) amortized policy — the cost-crossover experiment')
    ap.add_argument('--compute_frontier', action='store_true', help="(--eval_only) COMPUTE-MATCHED FRONTIER (Snell2024/AlphaZero Fig-2): sweep MPC beam over --frontier_beams and record held-out BRIGGS drv_recall@20 (freq-ranked top-K shortlist, SAME formula as rl_eval_suite.drv_recall_at => reproduces the paired-CSV MPC_exact), resid_match, reward Phi, recall@{1,5,10}, the analytic WM-queries/goal = C*(1+(k-1)*beam), and measured ms/goal; dumps results/rl_eval_condz/compute_frontier_<run>.csv. Policy-INDEPENDENT (beam only); the 7 policy-arm points are overlaid downstream from saved dumps. Early-returns after the dump; does NOT run the default recall ladder or gen_eval (default OFF => existing behavior unchanged).")
    ap.add_argument('--frontier_beams', default='1,2,4,8,16,32', help='(--compute_frontier) comma list of MPC beam widths to sweep (the compute axis)')
    ap.add_argument('--frontier_all_briggs', action='store_true', help='(--compute_frontier) score EVERY source==briggs fate (fold/split-independent: the default-frame MPC result per fate does not depend on the train/held-out split), so ONE job covers the SAME ~65 Briggs fates the policy points aggregate over — no fold double-counting. Default OFF => only this fold`s held-out Briggs fates.')
    ap.add_argument('--wm_ensemble', default=None, help='comma-separated ADDITIONAL frozen-WM checkpoints (different split seeds) beyond --checkpoint; enables cross-WM conservative scoring min_k Φ^(k) (multi-WM analogue of DAG pessimism)')
    ap.add_argument('--multi_wm_bon', action='store_true', help='(--eval_only + --wm_ensemble) deploy = argmax over N samples of a CROSS-WM conservative score (WCO=min_k Φ over the WM ensemble; UWO=mean−α·std) — report recall@1 of each rule (no retraining)')
    ap.add_argument('--mpc_honest', action='store_true', help='(--eval_only) (a) HELD-OUT-HONEST MPC fairness test: search MPC against the cosine-weighted centroid of the k NEAREST TRAINING fates (zero-shot neighbor target) instead of the exact held-out centroid; recall measured vs TRUTH. Isolates search-advantage vs exact-goal-access.')
    ap.add_argument('--mpc_honest_k', type=int, default=5, help='(--mpc_honest) # nearest training fates to average into the honest target')
    ap.add_argument('--pg_mpc', action='store_true', help='(--eval_only) (b) POLICY-GUIDED MPC (AlphaZero-style): at each beam step expand only the policy\'s top-M proposals (learned prior), WM scores + prunes. Reports recall + ms/goal + candidate-evals vs plain MPC (the cost win).')
    ap.add_argument('--pg_mpc_m', type=int, default=16, help='(--pg_mpc) # top policy candidates expanded per beam member per step (M<<|cand| is the cost win)')
    ap.add_argument('--headtohead', action='store_true', help='(--eval_only) run MPC-exact + ADDITIVE-GREEDY (summed observed drift-removed single-TF Δ, NO WM composition) baselines head-to-head with the policy: recall ladder + resid + drv_hit on held-out FDR fates, per-fate TF dump for the SOFT eval, and HSPC picks (P1/P2). The additive baseline is the "do you need the WM\'s nonlinear composition" control.')
    ap.add_argument('--mpc_shortlist_k', type=int, default=20, help='(--headtohead, EVAL-ONLY) dump MPC/additive/random as a top-K freq-ranked SHORTLIST (descending int weights => freq-rank==model-score rank) so freq-ranked driver-recall gives every baseline the SAME breadth budget as the policy best-of-N (fixes the ≤k-cocktail vs 20-slot mismatch); the ≤k best set is still kept in a `cocktail` field for reach + diversity.')
    ap.add_argument('--null_full_pool', action='store_true', help='(--eval_only, gated arms) draw the random null from the FULL 280-candidate pool even when the policy is gated to a per-goal shortlist. Gives the pipeline-vs-UNGATED-baseline Δ (consistent across ALL sources incl. HSPC), instead of the strict pool-matched null (policy-vs-its-own-shortlist).')
    ap.add_argument('--random_null', type=int, default=0, help='(--eval_only) >0 => run a MATCHED-BREADTH random-cocktail null with this many permutations: for each held-out fate draw N random cocktails matched to the policy\'s per-sample k, recompute drv_cov/drv_recall@20/resid_best/pass@N/distinct + HSPC P1/P2 union, and report policy vs null mean±std. Tests whether the best-of-N distribution beats breadth-matched chance.')
    ap.add_argument('--briggs_frame', action='store_true', help='(--eval_only) FRAME test: build a Briggs-frame control latent (encode Briggs pluripotent controls through the WM) and compare MPC reach to each held-out goal starting from the Joung-frame vs the Briggs-frame control. Tests whether the Joung-start-vs-Briggs-target frame mismatch costs reach on the current WM (bounds the payoff of a conditioned WM).')
    ap.add_argument('--frame_route', action='store_true', help='(TRAINING) FRAME ROUTING: plan Briggs-fate + HSPC goals from a Briggs-frame control (encode Briggs pluripotent controls) and Joung fates from the Joung control — makes the do-operator start + reward frame-consistent per goal source. The cheap (no-WM-retrain) fix for the Joung-start-vs-Briggs-target mismatch.')
    ap.add_argument('--kmax_joung', type=int, default=0, help='(per-source cocktail cap) >0 => cap Joung-Cao + Joung-combo fate cocktails at this k (the WM only saw Joung at k<=3, so k=6 Joung proposals are extrapolation). Briggs/HSPC keep --kmax. 0 => use global --kmax for all.')
    ap.add_argument('--unified_joung_combo', action='store_true', help='ALSO add HONEST Joung-combo-derived Cao-fate goals (combo-Δ reward target from joung_combo_cao_annotation, parallel to Briggs) — de-circularizes the Joung side (36 combos reach 24 Cao fates, 33 combo-specific). Joung frame; combo composition WITHOUT the frame-transfer confound.')
    ap.add_argument('--joung_combo_ann', default='/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs/joung_combo_cao_annotation.csv')
    ap.add_argument('--joung_combo_delta', default='/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs/joung_combo_delta.npz')
    ap.add_argument('--joung_combo_min_combos', type=int, default=1, help='min Joung combos per fate to admit it (Joung has few combos/fate, so default 1 — unlike Briggs min_combos=8 denoising threshold)')
    ap.add_argument('--recall_depths', default='20,50,100,200', help='(--random_null) comma list of top-k depths for the freq-ranked driver recall sweep (diagnoses whether Joung-Cao drivers are recovered deeper than @20).')
    ap.add_argument('--null_pareto', default='', help='(--random_null) comma list of best-of-N budgets for a Pareto sweep (e.g. 64,128,256,512,1024,2048,4096,11309). Samples N_max once/fate and subsamples prefixes; reports drv_recall@20 + SOFT-1 + SOFT-1-disc vs matched-breadth null at each N, by source.')
    ap.add_argument('--soft_disc', action='store_true', help='(--random_null) compute DISCRIMINATION SOFT-1 = SOFT1(policy picks vs OWN fate drivers) − mean SOFT1(vs OTHER held-out fates drivers), + an ORACLE ceiling (own drivers as picks) + additive-greedy reference. The honest cross-fate version of SOFT-1 (SOFT-1 alone is only proximity).')
    ap.add_argument('--soft2_ks', default='', help='(--random_null) comma list of KMeans cluster counts for a SOFT-2 sensitivity sweep (default hardcodes 12). e.g. 6,12,24,48.')
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, meta, op, data = build_stack(a, dev, n_dag=a.n_dag); data.set_pool(a.pool)
    if a.s_int == 'off':                                                     # Axis 3: attribution — zero the interaction head
        model.interaction_U = None; print("[s_int] interaction head DISABLED (additive-only do-op)", flush=True)
    if a.marker_targets:                                                # marker goals built after drift; init empty containers so keys/K below are valid
        data.de_lineage, data.z_target_lineage, data.lineage_drivers, data.lineage_keys = {}, {}, {}, []
    elif a.goal_source == 'cao':
        jl = None
        if a.joung_labels:
            _j = json.load(open(a.joung_labels)); _rows = _j['compare']
            _fates = [r['fate'] for r in _rows if r.get('joung_drivers')
                      and (r.get('obs_joung') if r.get('obs_joung') is not None else -9.0) >= a.joung_reach_thr]
            jl = {'fates': _fates, 'drivers': {r['fate']: r['joung_drivers'] for r in _rows}}
            print(f"[joung-labels] {len(_fates)} Joung fates at obs_joung>={a.joung_reach_thr} (from {os.path.basename(a.joung_labels)})", flush=True)
        build_cao_goals(data, op, model, dev, n_goals=a.n_cao_goals, goal_rep=a.goal_rep,
                        holdout=() if jl else ('Liver-Hematopoietic stem cells', 'Spleen-Erythroblasts'),
                        joung_labels=jl, goal_de=a.goal_de, ztarget_raw=a.ztarget_raw, blood_only=a.companion_blood,
                        fate_select=a.fate_select, fate_fdr_q=a.fate_fdr_q, fate_fdr_min_drv=a.fate_fdr_min_drv)
    else:
        data.build_lineage_targets(a.lineage_csv, qmax=a.qmax, min_drivers=a.min_drivers)
    keys = data.lineage_keys; names = [k.name for k in keys]; K = len(keys)
    g2i = data.gene_to_idx; thr = data.thresh
    # REACHABILITY: the '<pool>' set omits many answer-key TFs (e.g. TAL1, GATA2, ERG, SPI1), so the
    # policy could not propose them and recovery would be 0 by construction. Union in every P1/P2/driver
    # TF that exists in the WM gene space so recovery is achievable in principle.
    _cand = list(map(int, data.candidate_idx))
    _added = [int(g2i[g]) for g in sorted((P1 | P2 | DRIVERS) & set(g2i)) if int(g2i[g]) not in set(_cand)]
    if _added:
        _cand = _cand + _added
        data.candidate_idx = np.array(_cand)
        data.candidate_genes = [data.gene_names[i] for i in _cand]
    cand = data.candidate_idx
    briggs_lib = _briggs_library()
    print(f"[pool] '{a.pool}' + {len(_added)} answer-key TFs reachable -> |cand|={len(cand)}", flush=True)
    if a.hspc_modality >= 0 and a.train_goal_hspc and hasattr(data, '_ctrl_X'):   # Briggs-frame control (encoder-side modality token)
        with torch.no_grad():
            _mid = torch.full((data._ctrl_X.shape[0],), a.hspc_modality, dtype=torch.long, device=dev)
            _, _, _epsb = model.encoder(data._ctrl_X, modality_id=_mid)
        _dshift = float((_epsb.mean(0) - data.eps_ctrl_mean).norm())
        data.eps_ctrl = _epsb.detach(); data.eps_ctrl_mean = _epsb.mean(0)
        data.z0 = op.scm_forward(data.eps_ctrl, op.mean_A()).mean(0)
        print(f"[hspc_modality] re-encoded control in modality={a.hspc_modality} | ||Δeps_mean||={_dshift:.3f} (Briggs-frame rollout start)", flush=True)
    A = op.mean_A(); ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]
    drift = torch.tensor(np.mean([data.de_single[g]['lfc'] for g in data.single_split['train']], 0),
                         dtype=torch.float32, device=dev)
    if a.marker_targets:                                                    # PRODUCTION marker-signature fate goals (replaces build_cao_goals + unified)
        build_marker_goals(data, dev, op, a.marker_targets, drift, goal_rep=a.goal_rep)
        keys = data.lineage_keys; names = [k.name for k in keys]; K = len(keys)
    if a.unified_fates and not a.marker_targets:                            # inject Briggs-manifold Cao-fate goals BEFORE cent/TR/HO
        add_briggs_fate_goals(data, dev, a.briggs_ann, a.briggs_allk, drift, dual_reach=a.dual_reach,
                              goal_rep=a.goal_rep, min_combos=a.briggs_min_combos,
                              min_spec=a.briggs_min_spec, max_fates=a.briggs_max_fates,
                              fate_fdr_q=a.fate_fdr_q)
        keys = data.lineage_keys; names = [k.name for k in keys]; K = len(keys)
        print(f"[unified_fates] fate space now K={K} (Joung/driversig + Briggs-manifold)", flush=True)
    if getattr(a, 'unified_joung_combo', False) and not a.marker_targets:   # inject HONEST Joung-combo Cao-fate goals (combo-Δ target, de-circularizes the Joung side)
        add_briggs_fate_goals(data, dev, a.joung_combo_ann, a.joung_combo_delta, drift, dual_reach=a.dual_reach,
                              goal_rep=a.goal_rep, min_combos=a.joung_combo_min_combos,
                              min_spec=a.briggs_min_spec, max_fates=a.briggs_max_fates,
                              fate_fdr_q=a.fate_fdr_q, source='joung_combo', skip_sources=('briggs',))
        keys = data.lineage_keys; names = [k.name for k in keys]; K = len(keys)
        print(f"[unified_fates] fate space now K={K} (+ Joung-combo honest targets)", flush=True)
    cent = torch.stack([torch.tensor(data.de_lineage[n]['lfc'], dtype=torch.float32, device=dev) - drift
                        for n in names])                                     # [K,G] lineage centroids
    if int(getattr(a, 'oe_remove_target', 0)) > 0:                          # OE/BULK-AXIS REMOVAL: strip the top-K shared PCs (the bulk residual that makes fate targets collinear) => fate-SPECIFIC reward (now allowed WITH --marker_targets: the bulk-removed-target experiment)
        _K = int(a.oe_remove_target)
        with torch.no_grad():
            _U, _S, _Vt = torch.linalg.svd(cent, full_matrices=False)        # UNCENTERED: shared OE axis lives in the MEAN, so PC1 of the uncentered targets IS the OE axis (centering removes it before SVD => no-op)
            _P = _Vt[:_K]                                                    # [K, G] top-K shared directions incl. OE
            cent = cent - (cent @ _P.T) @ _P                                 # project each fate target off the shared OE subspace => fate-specific residual
        for i, n in enumerate(names):                                        # write back so env (reward) + recall metrics use the OE-removed target
            data.de_lineage[n]['lfc'] = (cent[i] + drift).detach().cpu().numpy()
        print(f"[oe_remove_target] projected fate targets off top-{_K} shared PCs (was {float(_S[0]/_S.sum()):.2f} PC1 var-frac) => fate-specific reward space", flush=True)
    if getattr(a, 'var_weight', False) and getattr(data, 'fate_std', None):  # PRECISION-weight the target by 1/(per-gene member spread): focus reward on CONSISTENT fate genes, down-weight noisy ones
        _nvw = 0
        for i, n in enumerate(names):
            _std = data.fate_std.get(n)
            if _std is None: continue
            _w = 1.0 / (torch.tensor(np.asarray(_std, np.float32), device=dev) + 1e-3)
            _w = _w / (_w.mean() + 1e-9)                                     # normalize so overall scale unchanged
            cent[i] = cent[i] * _w; data.de_lineage[n]['lfc'] = (cent[i] + drift).detach().cpu().numpy(); _nvw += 1
        print(f"[var_weight] precision-weighted {_nvw} fate targets by 1/(per-gene member std) => rewards consistent fate-defining genes", flush=True)
    centn = cent / (cent.norm(dim=1, keepdim=True) + 1e-9)
    rng = np.random.default_rng(0)

    if a.marker_targets and a.fold > -99:                                # FOLD-based leave-fates-out (production); fold -1 (incl. Liver-HSC) always held out
        HO = sorted(i for i, n in enumerate(names) if data.fate_fold.get(n, -1) in (a.fold, -1))
        print(f"[fold-heldout] fold={a.fold}: {len(HO)}/{K} fates held out (fold {a.fold} + fold -1) | Tier-A held: "
              f"{sum(data.fate_tier.get(names[i]) == 'A' for i in HO)}", flush=True)
    elif a.heldout_frac > 0:                                             # random multi-goal held-out (n>1 generalization)
        _rho = np.random.default_rng(1).permutation(K)
        HO = sorted(int(i) for i in _rho[:max(1, int(round(a.heldout_frac * K)))])
        print(f"[multigoal] random held-out {len(HO)}/{K} goals: {[names[i] for i in HO]}", flush=True)
    else:
        # comma-separated substrings → hold out EVERY matching lineage (e.g. the whole blood/immune branch,
        # so a neighbouring blood lineage in training can't give the HSPC answer away)
        ho_sub = [s.strip().lower() for s in a.heldout_substr.split(',') if s.strip()] \
                 + (['hematopoietic'] if a.goal_source == 'cao' else [])
        HO = [i for i, n in enumerate(names) if any(s in n.lower() for s in ho_sub)]
    if not HO:
        print(f"[WARN] no lineage matches '{a.heldout_substr}'; holding out the LAST lineage instead", flush=True)
        HO = [K - 1]
    TR = [i for i in range(K) if i not in HO]
    train_keys = [keys[i] for i in TR]
    print(f"[multigoal] K={K} fates | TRAIN {len(TR)} | HELD-OUT (zero-shot eval): {[names[i] for i in HO]}", flush=True)
    print(f"[multigoal] train fates: {[names[i].split('-')[-1][:20] for i in TR]}", flush=True)

    # ============ claim 2: build the ZERO-SHOT HSPC goal (Briggs, cross-dataset) ============
    # The HSPC direction in LATENT space = gradient of the Briggs fate-readout wrt z = coef/scale (the
    # readout was fit on Briggs fate LABELS, never trained on by the policy). We condition the trained
    # Joung-only policy on z_target_hspc = z0 + c*ghat and check whether its proposals recover P1/P2.
    rd = np.load(a.readout_briggs)
    ghat = (rd['coef'] / rd['scale']).astype(np.float32); ghat = ghat / (np.linalg.norm(ghat) + 1e-9)
    ghat_t = torch.tensor(ghat, device=dev)
    z0 = data.z0.to(dev)
    c = a.hspc_scale or float(torch.stack([(data.z_target_lineage[names[i]].to(dev) - z0).norm() for i in TR]).mean())
    z_hspc = z0 + c * ghat_t
    axis_gene = (op.decode(z_hspc.unsqueeze(0))[0] - op.decode(z0.unsqueeze(0))[0])          # gene-space HSPC delta
    data.z_target_lineage['HSPC'] = z_hspc
    data.de_lineage['HSPC'] = data._de_entry((axis_gene.cpu().numpy() - drift.cpu().numpy()))
    data.lineage_drivers['HSPC'] = {'train': [], 'test': sorted(P1 | P2)}
    hspc_key = Lineage('HSPC'); o_hspc = axis_gene - drift
    if a.hspc_answerkey:                                     # ANSWER-KEY-GROUNDED: mean OBSERVED Δ of the P1/P2 answer-key combos (full non-gameable HSPC transcriptome)
        _ak = pd.read_csv(a.briggs_ann); _akm = _ak['is_answerkey'].values.astype(bool)
        _dd = np.load(a.briggs_allk, allow_pickle=True); _obs = _dd['obs_delta'].astype(np.float32)
        _od = _obs[_akm].mean(0)                             # [G] mean observed Δ of the answer-key combos (all-k)
        o_hspc = torch.tensor(_od, device=dev) - drift
        print(f"[hspc_answerkey] o_hspc <- mean observed Δ of {int(_akm.sum())} answer-key combos − drift | ||o_hspc||={float(o_hspc.norm()):.3f} "
              f"| cos(answerkey, all-pairs-mean)={float((o_hspc@ (torch.tensor(_obs.mean(0),device=dev)-drift))/((o_hspc.norm()+1e-9)*((torch.tensor(_obs.mean(0),device=dev)-drift).norm()+1e-9))):+.3f}", flush=True)
    elif a.hspc_obs_de and os.path.exists(a.hspc_obs_de):     # REAL observed Briggs endpoint Δ (mean over hit combos) instead of the readout gradient
        _od = np.load(a.hspc_obs_de)['obs_delta'].astype(np.float32).mean(0)   # [G] mean observed Δ of Briggs hit combos (day 14/28, vs Briggs control)
        o_hspc = torch.tensor(_od, device=dev) - drift
        print(f"[hspc_obs_de] o_hspc <- observed Briggs hit-combo mean Δ − drift | ||o_hspc||={float(o_hspc.norm()):.3f} | cos(obs,readout-axis)={float((o_hspc@ (axis_gene-drift))/((o_hspc.norm()+1e-9)*((axis_gene-drift).norm()+1e-9))):+.3f}", flush=True)
    if a.hspc_ortho_drift:                      # LESS-DRIFT goal: project the drift direction OUT of o_hspc so the reward
        dn = drift / (drift.norm() + 1e-9)      # rewards HSPC-SPECIFIC genes, not the generic OE response (control no longer ~optimal)
        o_hspc = o_hspc - (o_hspc @ dn) * dn
        print(f"[hspc_ortho_drift] projected drift out of o_hspc | cos(o_hspc,drift)->~0 | ||o_hspc||={float(o_hspc.norm()):.3f}", flush=True)
    FR = FateReward(op, [a.readout_briggs], data.eps_ctrl_mean.to(dev), lam=0.0, n_dag=a.n_dag)
    if a.support_feats and os.path.exists(a.support_feats):
        FR.set_support(a.support_feats, support_gamma=0.0)          # gamma=0: fate unchanged; report maha to flag off-manifold picks
    A_eval = op.mean_A()
    print(f"[HSPC-goal] readout={os.path.basename(a.readout_briggs)} | ||z_hspc-z0||={c:.2f} | "
          f"support={'on' if FR.sup_L is not None else 'off'} | leak-free P1/P2 answer key", flush=True)

    if a.train_goal_hspc:                       # SINGLE-goal RL: HSPC becomes the sole TRAIN + eval goal (append to the atlas so all machinery works)
        # FIX double-drift: de_lineage['HSPC'] above is already (axis_gene - drift); the env's reset subtracts drift
        # AGAIN for reward_mode=resid -> target = axis_gene - 2*drift. Store the RAW axis_gene so env's single
        # subtraction yields o_hspc = axis_gene - drift (matching cent/eval and the single-drift-removed rv).
        data.de_lineage['HSPC'] = data._de_entry((o_hspc + drift).detach().cpu().numpy())   # env subtracts drift once -> o_hspc (works for ortho too)
        names.append('HSPC'); keys.append(hspc_key)
        cent = torch.cat([cent, o_hspc.unsqueeze(0)], 0)
        centn = cent / (cent.norm(dim=1, keepdim=True) + 1e-9)
        _R = getattr(build_cao_goals, '_R', None)            # HSPC must share the Cao goal channel (same JL proj of the gene target),
        if a.goal_rep == 'gene' and _R is not None:          # else its z_target (a raw latent) is off-space and the policy can't condition on it
            _si = build_cao_goals._sidx
            data.z_target_lineage['HSPC'] = _R @ o_hspc[_si].to(dev)
            print(f"[hspc_multigoal] HSPC goal-channel re-projected through the Cao JL (consistent goal space)", flush=True)
        K = len(keys); hspc_i = K - 1
        if a.hspc_heldout:                       # HSPC ALWAYS HELD OUT: train on intermixed Cao fates (existing TR from heldout_frac), eval zero-shot on the held-out Cao fates + HSPC
            HO = sorted(set(int(i) for i in HO) | {hspc_i}); TR = [i for i in TR if i != hspc_i]
            print(f"[hspc_heldout] TRAIN {len(TR)} intermixed Cao fates | HELD-OUT (zero-shot) {len(HO)} = {len(HO)-1} Cao + HSPC | kmax={a.kmax}", flush=True)
        elif a.hspc_multigoal:                   # MULTI-goal: HSPC joins the atlas TR (V(s,t) goal-dependent => EV>0 => from-scratch set-PPO viable)
            TR = list(TR) + [hspc_i]; HO = [hspc_i]
            print(f"[hspc_multigoal] HSPC added to the {len(TR)-1}-fate atlas as a TRAIN goal | eval=HSPC | kmax={a.kmax}", flush=True)
        else:                                    # SINGLE-goal
            TR = [hspc_i]; HO = [hspc_i]
            print(f"[train_goal_hspc] SINGLE-goal RL on HSPC | K={K} | kmax={a.kmax} | rloo={a.rloo}", flush=True)
        train_keys = [keys[i] for i in TR]

    # NON-GAMEABLE HSPC readout: mean predicted drift-removed Δ on the HSPC program panel, EXCLUDING (per cocktail)
    # the cocktail's OWN TFs — so it measures INDIRECT induction of the broad program (GATA2+ETV2 scored on whether
    # it induces RUNX1/TAL1/LMO2/CD34/..., not on GATA2/ETV2 themselves); a single off-target (ZNF680) can't raise it.
    _panel = sorted(m for m in HSPC_MARKERS if m in g2i)
    print(f"[hspc_marker_readout] {len(_panel)}-gene HSPC program panel (per-cocktail self-exclusion): {_panel}", flush=True)

    def score_cocktail(genes):
        """(1) Briggs ridge readout [GAMEABLE — ZNF680 exploits it]; (2) fate_marker = mean predicted drift-removed Δ
        on the HSPC program panel MINUS the cocktail's own TFs [NON-gameable, biology-grounded]; (3) P1/P2/driver
        overlap; (4) maha off-manifold flag."""
        genes = [int(g) for g in genes if 0 <= int(g) < len(data.gene_names)]
        if not genes:
            return None
        z = op.apply_sets(FR.eps0, [genes], A_eval)
        fate = float(FR._fate_all(z)[0, 0])
        maha = float(FR.mahalanobis(z)[0]) if FR.sup_L is not None else float('nan')
        gset = {data.gene_names[i] for i in genes}
        mk = [g2i[m] for m in _panel if m not in gset]                       # full panel minus the cocktail's OWN TFs (no self-OE)
        fate_marker = float((op.decode(z)[0] - ctrl_recon - drift)[mk].mean()) if mk else float('nan')
        return dict(fate=round(fate, 4), fate_marker=round(fate_marker, 4), maha=round(maha, 2),
                    P1=len(gset & P1), P2=len(gset & P2), drv=len(gset & DRIVERS), k=len(genes), picks=sorted(gset))

    HROWS = {}
    wb = wb_init(a.run_name or f'multigoal_{a.heldout_substr}', vars(a),
                 tags=['briggs', 'multigoal', 'zeroshot']) if a.wandb else None

    # Mahalanobis support to the JOUNG-perturbation manifold (on-manifold = where observed Joung singles land)
    maha_fn = None
    if a.beta_maha > 0:
        _mex = {t.strip() for t in a.maha_exclude_tfs.split(',') if t.strip()}
        _sup_tfs = [g for g in data.z_target_single if g not in _mex]
        if _mex: print(f"[maha] leave-hit-out: excluded {len(data.z_target_single)-len(_sup_tfs)} TFs from support manifold", flush=True)
        Zsup = torch.stack([data.z_target_single[g] for g in _sup_tfs]).to(dev)
        mu_s = Zsup.mean(0); cov_s = torch.cov(Zsup.T) + 1e-3 * torch.eye(op.d, device=dev)
        L_s = torch.linalg.cholesky(cov_s)

        def maha_fn(z):                                                     # z [K,d] -> [K]
            sol = torch.linalg.solve_triangular(L_s, (z - mu_s).T, upper=False)
            return ((sol * sol).sum(0) / op.d).sqrt()
        print(f"[maha] Joung-manifold support from {Zsup.shape[0]} single latents | beta={a.beta_maha} tau={a.maha_tau}", flush=True)
    if getattr(a, 'driver_pool', False):                                    # VALID smaller pool = GLOBAL union of all fates' answer drivers (contains every answer; same pool for all goals => non-leaky). Replaces the broken causal_gate (which excluded drivers).
        _du = set()
        for _nm, _dd in data.lineage_drivers.items():
            _du |= set(_dd.get('full', [])) | set(_dd.get('train', [])) | set(_dd.get('test', []))
        for _cc in getattr(data, 'answer_combos', {}).values():
            for _c in _cc: _du |= set(_c)
        # NOTE: deterministic driver-union only (72 TFs) so train/eval pools always match. P1/P2 already enter via HSPC lineage_drivers; the 1 arrayed-only TF that this drops is immaterial (verified 72 vs 73).
        _keep = [int(i) for i in data.candidate_idx if data.gene_names[int(i)] in _du]
        data.candidate_idx = np.array(_keep); data.candidate_genes = [data.gene_names[i] for i in _keep]; cand = data.candidate_idx
        _kn = set(data.candidate_genes)
        print(f"[driver_pool] cand = driver-union + HSPC(P1/P2+arrayed) key: {len(cand)}/280 | P1P2 in-pool {len((P1|P2)&_kn)}/{len(P1|P2)} | all fate-drivers+answer-combos retained; global => non-leaky", flush=True)
    env = CausalLatentEnv(op, data, max_steps=1, reward_mode=a.reward_mode, n_dag_samples=a.n_dag,
                          alpha_pess=a.alpha, reward_scale=a.reward_scale,
                          maha_fn=maha_fn, beta_maha=a.beta_maha, maha_tau=a.maha_tau,
                          reward_match=a.reward_match, cvar_reward_alpha=a.cvar_reward_alpha)
    d = op.d; cpos = {g: i for i, g in enumerate(cand)}

    # ---- GRAPH slot-head operators (FIXED, from the frozen WM): message-passing policy head over the learned causal graph ----
    _mp_op = None; _decode_W = None
    if getattr(a, 'slot_head', 'mlp') != 'mlp':
        with torch.no_grad():
            _A = op.mean_A().detach().to(dev)                                  # [d,d] learned latent DAG (regulator->target)
            _Adir = _A.t()                                                     # A^T: propagate fate-relevance UPSTREAM to regulators
            _dm = getattr(op.model.decoder, 'dec_mask', None)                  # effective SENA decoder = D * dec_mask
            _D = (op.model.decoder.D * _dm) if _dm is not None else op.model.decoder.D
            _D = _D.detach().to(dev)                                           # [G,d]
            _cand_idx = torch.tensor([int(c) for c in cand], device=dev, dtype=torch.long)
            _rownorm = lambda M: M / (M.abs().sum(1, keepdim=True) + 1e-8)
            if a.slot_head == 'grn_latent':
                _mp_op = _rownorm(_Adir + torch.eye(_Adir.shape[0], device=dev))   # [d,d] row-normalized A^T + self-loops
                _decode_W = _D[_cand_idx, :]                                        # [C,d] SENA decoder rows for candidates
            else:                                                              # grn_gene: candidate-gene GRN = D_c A^T D_c^T
                _Dc = _D[_cand_idx, :]                                              # [C,d]
                _GRN = _Dc @ _Adir @ _Dc.t()                                        # [C,C]
                _mp_op = _rownorm(_GRN + torch.eye(_Dc.shape[0], device=dev))       # [C,C]
        print(f"[slot_head] {a.slot_head} | mp_hops={a.mp_hops} gate={a.mp_gate} dir=A^T | mp_op {tuple(_mp_op.shape)}", flush=True)
    _gh = dict(slot_head=getattr(a, 'slot_head', 'mlp'), mp_hops=getattr(a, 'mp_hops', 0),
               mp_gate=getattr(a, 'mp_gate', 'gated'), mp_op=_mp_op, decode_W=_decode_W)

    if getattr(a, 'frame_route', False):                                 # FRAME ROUTING (installed for BOTH eval_only and training): Briggs/HSPC goals plan from a Briggs-frame control; Joung goals keep the Joung frame
        import anndata as _ad, scipy.sparse as _sp
        _PL = ['POU5F1', 'NANOG', 'SOX2', 'LIN28A', 'DPPA4', 'ZFP42']
        _BH = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/_decompressed/all_cells_final_v2.h5ad"
        def _ln(X):
            X = (X.toarray() if _sp.issparse(X) else np.asarray(X)).astype(np.float32); return np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)
        _B = _ad.read_h5ad(_BH); _bg = list(map(str, _B.var_names)); _gn = list(map(str, data.gene_names)); _gi = {g: i for i, g in enumerate(_gn)}
        _sh = [g for g in _bg if g in _gi]; _si = np.array([_gi[g] for g in _sh]); _X = _ln(_B[:, _sh].X)
        _pm = [_sh.index(g) for g in _PL if g in _sh]; _ps = _X[:, _pm].mean(1); _isc = _ps >= np.quantile(_ps, 0.90)
        _Xb = np.zeros((int(_isc.sum()), len(_gn)), np.float32); _Xb[:, _si] = _X[_isc]
        with torch.no_grad():
            _o = op.model.encoder(torch.tensor(_Xb, device=dev)); _epsB = (_o[2] if isinstance(_o, (tuple, list)) else _o).detach()
        _e0 = _epsB.mean(0); _z0 = op.scm_forward(_epsB, A).mean(0); _cr = op.decode(_z0.unsqueeze(0))[0]
        _n2s = {names[i]: data.lineage_drivers[names[i]].get('source', 'joung') for i in range(len(names))}
        def _srcfn(tk):
            nm = getattr(tk, 'name', str(tk)); return 'hspc' if nm == 'HSPC' else _n2s.get(nm, 'joung')
        env.set_frames(_e0, _z0, _cr, _srcfn)
        print(f"[frame_route] Briggs-frame control installed ({int(_isc.sum())} pluri ctrls; "
              f"||e0_B-e0_J||={float((_e0 - data.eps_ctrl_mean).norm()):.1f}) | "
              f"{sum(1 for v in _n2s.values() if v == 'briggs')} Briggs goals routed to Briggs frame", flush=True)
        if int(getattr(a, 'kmax_joung', 0) or 0) > 0:                    # PER-SOURCE cocktail cap (needs the frame-route source map)
            _kj = int(a.kmax_joung)
            env.set_kmax(a.kmax, {'joung': _kj, 'joung_combo': _kj, 'briggs': a.kmax, 'hspc': a.kmax})
            print(f"[kmax] per-source cap: joung/joung_combo={_kj} briggs={a.kmax} hspc={a.kmax} (WM Joung support k<=3)", flush=True)
        if getattr(a, 'mask_stop', False) and getattr(a, 'force_k_mode', 'source') == 'answer':  # PER-FATE forced-k = each fate's median answer-combo breadth (clamped [1,kmax])
            _fk = {}
            for _nm, _acs in getattr(data, 'answer_combos', {}).items():
                _sz = [len(s) for s in _acs if s]
                if _sz: _fk[_nm] = int(min(a.kmax, max(1, int(round(float(np.median(_sz)))))))
            env.set_kmax_fate(_fk)
            print(f"[mask_stop] force_k_mode=answer: {len(_fk)} fates forced to median answer-combo k "
                  f"(dist {dict(sorted(__import__('collections').Counter(_fk.values()).items()))}); STOP masked; fates w/o combo answer keep per-source cap", flush=True)
        elif getattr(a, 'mask_stop', False):
            print(f"[mask_stop] force_k_mode=source: STOP masked at all slots => forced to per-source cap (joung={a.kmax_joung} briggs/hspc={a.kmax})", flush=True)
    elif int(getattr(a, 'kmax_joung', 0) or 0) > 0:
        print("[kmax] WARNING: --kmax_joung needs --frame_route (source routing) to apply; ignoring, using global --kmax", flush=True)

    # ------------------------------------------------------------------ eval on the held-out goal(s)
    def eval_goal(planner_idx, li):
        gidx = [g for g in planner_idx(li) if 0 <= g < len(data.gene_names)]
        if not gidx:
            return None
        rv = pred_resid(op, data, gidx, drift, ctrl_recon, A)
        order = torch.argsort(-(centn @ (rv / (rv.norm() + 1e-9)))).tolist()  # nearest lineage over ALL K
        o = cent[li]; sel = torch.where(o.abs() > thr)[0]
        rm = float(torch_match(rv.unsqueeze(0), o[sel], sel)[0])
        gset = {data.gene_names[i] for i in gidx}; drv = data.lineage_drivers[names[li]]
        return dict(recall1=int(order[0] == li), recall3=int(li in order[:3]),
                    recall5=int(li in order[:5]), recall10=int(li in order[:10]), resid=rm,
                    drv_hit=int(len(gset & set(drv['train'] + drv['test'])) > 0),
                    held=int(len(gset & set(drv['test'])) > 0),
                    P1=len(gset & P1), P2=len(gset & P2), drv=len(gset & DRIVERS), picks=sorted(gset))

    def evalp(planner_idx, n_rep=1):
        accs = [r for _ in range(n_rep) for li in HO if (r := eval_goal(planner_idx, li))]
        if not accs:
            return {}
        agg = {k: float(np.mean([r[k] for r in accs]))
               for k in ['recall1', 'recall3', 'recall5', 'recall10', 'resid', 'drv_hit', 'held', 'P1', 'P2', 'drv']}
        agg['picks'] = accs[0]['picks']
        return agg

    perfate_tf_dump = {}   # tag -> {fate -> {freq: {TF:count over N samples}, drivers: [experimental answer-key TFs]}} for the TF-recovery eval
    def gen_eval(agent, tag, N=64, fates=None):
        """GENERATIVE, distribution-aware eval (fair to a sampling policy, unlike greedy recall@1):
        sample the policy N times per held-out goal ->
          pass@N     : ANY of N samples' nearest-lineage == goal (best-of-N recall)
          resid_best : best resid-match over N (can the policy REACH the goal at all)
          drv_cov    : |union drivers proposed over N| / |goal drivers| (coverage)
          distinct   : # distinct cocktails with resid>0 (diversity)
          specificity: resid-to-goal when conditioned on the CORRECT goal minus when conditioned on a
                       MISMATCHED goal (same z_ctrl) -> is the goal channel steering to the RIGHT fate?"""
        import numpy as _np
        rng_g = _np.random.default_rng(0)
        out = {k: [] for k in ['passN', 'resid_best', 'drv_cov', 'distinct', 'specificity']}
        out['drv_rank'] = []; out['drv_recall20'] = []
        out['drv_recall5'] = []; out['prec5'] = []                                  # ordinary (all-N) readout @5 + precision@5
        for _k in ['e25_recall5','e25_recall20','e25_prec5','e10_recall5','e10_recall20','e10_prec5','wm_extra_per_fate']:
            out[_k] = []                                                            # ELITE-SET readout (top q% of the N sampled cocktails by resid-match Φ) + added WM-query cost
        # BIAS CONTROL: RANDOM+ELITE null (uniform-random k-matched cocktails, same Φ-filter) + LEAVE-ONE-OUT ΔΦ readout
        for _k in ['rand_all_r5','rand_all_r20','rand_all_p5','rand_e25_r5','rand_e25_r20','rand_e25_p5',
                   'rand_e10_r5','rand_e10_r20','rand_e10_p5','loo_e10_r5','loo_e10_r20','loo_e10_p5',
                   'wm_extra_random','wm_extra_loo']:
            out[_k] = []
        for _k in ['bon_r1','bon_r10','bon_r1_null','bon_r10_null']: out[_k] = []   # best-of-N recall@1/@10 (policy) + matched random best-of-N null
        _perfate_top = {}   # per-held-out-fate: top-8 TFs by SAMPLED frequency (is the SAMPLING diverse per fate, unlike the greedy mode?)
        cand_genes = {data.gene_names[int(i)] for i in cand}
        eval_fates = HO if fates is None else fates
        _GG = getattr(env, '_goal_gate', None)
        for li in eval_fates:
            obs = env.reset(keys[li]); s_emb, t_emb = obs[:d], obs[d:]
            if _GG is not None and li in _GG:                        # RETRIEVER/cosine gate: constrain the policy to this goal's shortlist at eval too
                _blk = torch.ones(len(cand), dtype=torch.bool, device=dev); _blk[list(_GG[li])] = False; agent.policy._allowed_block = _blk
            else:
                agent.policy._allowed_block = None
            drvset = set(data.lineage_drivers[names[li]]['train'] + data.lineage_drivers[names[li]]['test'])
            resids, hit_any, cov, cock, freq = [], 0, set(), set(), {}
            _samp = []                                              # per-sample (resid-match Φ, gene-set) for the ELITE-SET readout — reuses rm below (0 extra WM calls)
            _brank = 999                                            # best (min) rank of the true fate over the N policy samples
            _cgp = [int(cand[i]) for i in _GG[li]] if (_GG is not None and li in _GG) else [int(c) for c in cand]  # pool-/full-matched
            _kc = int(getattr(env, 'kmax_active', None) or a.kmax)
            for _ in range(N):
                _, genes, _, _ = agent.select(s_emb, t_emb)
                gi = [cand[g] for g in genes]
                if not gi:
                    continue
                rv = pred_resid(op, data, gi, drift, ctrl_recon, A)
                _ca = centn @ (rv / (rv.norm() + 1e-9)); _rk = int((_ca > _ca[li]).sum()); _brank = min(_brank, _rk)
                if _rk == 0:
                    hit_any = 1
                o = cent[li]; sl = torch.where(o.abs() > thr)[0]
                rm = float(torch_match(rv.unsqueeze(0), o[sl], sl)[0]); resids.append(rm)
                gs = {data.gene_names[i] for i in gi}; cov |= (gs & drvset)
                for g in gs:
                    freq[g] = freq.get(g, 0) + 1
                _samp.append((rm, gs, list(gi)))                    # store for the elite-set readout (rm = resid-match Φ, already computed => free) + gene idx for LOO ΔΦ
                if rm > 0:
                    cock.add(frozenset(gs))
            # ZERO-SHOT RANKING: rank all candidate TFs by selection frequency; where do true drivers land?
            ranked = sorted(cand_genes, key=lambda g: -freq.get(g, 0))
            drv_in = [g for g in drvset if g in cand_genes]
            ranks = [ranked.index(g) + 1 for g in drv_in]
            out['drv_rank'].append(float(np.median(ranks)) if ranks else float('nan'))
            out['drv_recall20'].append(len(set(ranked[:20]) & set(drv_in)) / max(1, len(drv_in)))
            # ---------- ELITE-SET READOUT: re-rank TFs by frequency within the top q% highest-Φ cocktails ----------
            # A diverse (cond) policy dilutes each TF's marginal freq over many high-reward cocktails; ranking TFs
            # only within the ELITE (best-resid-match) cocktails should un-dilute drivers WITHOUT retraining.
            _ndrv = max(1, len(drv_in))
            def _rank_fq(fq): return sorted(cand_genes, key=lambda g: -fq.get(g, 0))
            def _rec(fq, k): return len(set(_rank_fq(fq)[:k]) & set(drv_in)) / _ndrv
            def _prc(fq, k): return len(set(_rank_fq(fq)[:k]) & set(drv_in)) / k
            _ssort = sorted(_samp, key=lambda x: -x[0])              # cocktails ranked by resid-match Φ (desc)
            def _elite_fq(q):
                n_e = max(1, int(np.ceil(q / 100.0 * len(_ssort))))
                fq = {}
                for _, gsE, _giE in _ssort[:n_e]:
                    for g in gsE: fq[g] = fq.get(g, 0) + 1
                return fq, n_e
            freq_e25, _ne25 = _elite_fq(25); freq_e10, _ne10 = _elite_fq(10)
            out['drv_recall5'].append(_rec(freq, 5)); out['prec5'].append(_prc(freq, 5))
            out['e25_recall5'].append(_rec(freq_e25, 5)); out['e25_recall20'].append(_rec(freq_e25, 20)); out['e25_prec5'].append(_prc(freq_e25, 5))
            out['e10_recall5'].append(_rec(freq_e10, 5)); out['e10_recall20'].append(_rec(freq_e10, 20)); out['e10_prec5'].append(_prc(freq_e10, 5))
            out['wm_extra_per_fate'].append(0)                      # elite readout re-ranks the SAME N samples by rm (already computed) => 0 extra WM Φ decode calls
            _perfate_top[names[li]] = [(g, freq.get(g,0)) for g in ranked[:8]]
            # ====================== BIAS CONTROL 1: RANDOM + ELITE (Φ-filter null) ======================
            # Draw N uniform-random cocktails from the SAME pool, matched to the policy's per-sample k, score each
            # with the SAME WM Φ (resid-match to goal), apply the SAME top-q% elite filter -> what does the Φ-filter
            # ALONE recover, with NO learned policy? (costs N NEW WM Φ calls / fate; policy+elite reused, cost 0.)
            o = cent[li]; sl = torch.where(o.abs() > thr)[0]        # fate-level DEG target (re-affirm; used by the random+elite & LOO Φ scoring below)
            _ks_pol = [len(gsE) for _, gsE, _ in _samp] or [min(_kc, len(_cgp))]
            _rsamp = []; _wm_rand = 0
            for _j in range(N):
                _kk = min(int(_ks_pol[_j % len(_ks_pol)]), len(_cgp))
                if _kk < 1: continue
                _rc = [int(x) for x in rng_g.choice(_cgp, size=_kk, replace=False)]
                _rvr = pred_resid(op, data, _rc, drift, ctrl_recon, A); _wm_rand += 1
                _rmr = float(torch_match(_rvr.unsqueeze(0), o[sl], sl)[0])
                _rsamp.append((_rmr, {data.gene_names[i] for i in _rc}, _rc))
            _rfreq = {}
            for _, gsE, _ in _rsamp:
                for g in gsE: _rfreq[g] = _rfreq.get(g, 0) + 1
            _rsort = sorted(_rsamp, key=lambda x: -x[0])
            def _relite_fq(q):
                n_e = max(1, int(np.ceil(q / 100.0 * len(_rsort)))); fq = {}
                for _, gsE, _ in _rsort[:n_e]:
                    for g in gsE: fq[g] = fq.get(g, 0) + 1
                return fq
            _rfreq_e25 = _relite_fq(25); _rfreq_e10 = _relite_fq(10)
            out['rand_all_r5'].append(_rec(_rfreq, 5)); out['rand_all_r20'].append(_rec(_rfreq, 20)); out['rand_all_p5'].append(_prc(_rfreq, 5))
            out['rand_e25_r5'].append(_rec(_rfreq_e25, 5)); out['rand_e25_r20'].append(_rec(_rfreq_e25, 20)); out['rand_e25_p5'].append(_prc(_rfreq_e25, 5))
            out['rand_e10_r5'].append(_rec(_rfreq_e10, 5)); out['rand_e10_r20'].append(_rec(_rfreq_e10, 20)); out['rand_e10_p5'].append(_prc(_rfreq_e10, 5))
            out['wm_extra_random'].append(_wm_rand)
            # ====================== BIAS CONTROL 2: LEAVE-ONE-OUT ΔΦ readout ======================
            # Rank TF a by mean over the POLICY's elite-10% cocktails containing a of [Φ(x) − Φ(x\{a})] — marginal
            # contribution, not raw membership frequency (theory-critique: less biased toward high-degree TFs).
            # Costs sum_x |x| extra WM Φ calls (only over the elite subset).
            _loo_sum = {}; _loo_cnt = {}; _wm_loo = 0
            for _rmE, _gsE, _giE in _ssort[:_ne10]:
                _phi_full = _rmE
                for _idx in _giE:
                    _rest = [j for j in _giE if j != _idx]
                    if _rest:
                        _rvL = pred_resid(op, data, _rest, drift, ctrl_recon, A); _wm_loo += 1
                        _phi_wo = float(torch_match(_rvL.unsqueeze(0), o[sl], sl)[0])
                    else:
                        _phi_wo = 0.0
                    _g = data.gene_names[_idx]; _loo_sum[_g] = _loo_sum.get(_g, 0.0) + (_phi_full - _phi_wo); _loo_cnt[_g] = _loo_cnt.get(_g, 0) + 1
            _loo_score = {g: _loo_sum[g] / _loo_cnt[g] for g in _loo_sum}
            def _loo_rank(): return sorted(cand_genes, key=lambda g: -_loo_score.get(g, -1e9))
            def _lrec(k): return len(set(_loo_rank()[:k]) & set(drv_in)) / _ndrv
            def _lprc(k): return len(set(_loo_rank()[:k]) & set(drv_in)) / k
            out['loo_e10_r5'].append(_lrec(5)); out['loo_e10_r20'].append(_lrec(20)); out['loo_e10_p5'].append(_lprc(5))
            out['wm_extra_loo'].append(_wm_loo)
            _full_drv = [g for g in data.lineage_drivers[names[li]].get('full', []) if g in cand_genes]
            perfate_tf_dump.setdefault(tag, {})[names[li]] = {'freq': {g: int(c) for g, c in freq.items()},
                                                              'freq_elite25': {g: int(c) for g, c in freq_e25.items()},  # ELITE readout: freq within top-25% highest-Φ cocktails
                                                              'freq_elite10': {g: int(c) for g, c in freq_e10.items()},  # ELITE readout: freq within top-10% highest-Φ cocktails
                                                              'n_samples': int(len(_samp)), 'n_elite25': int(_ne25), 'n_elite10': int(_ne10),
                                                              'recall': {'all': {'r5': _rec(freq, 5), 'r20': _rec(freq, 20), 'p5': _prc(freq, 5)},
                                                                         'e25': {'r5': _rec(freq_e25, 5), 'r20': _rec(freq_e25, 20), 'p5': _prc(freq_e25, 5)},
                                                                         'e10': {'r5': _rec(freq_e10, 5), 'r20': _rec(freq_e10, 20), 'p5': _prc(freq_e10, 5)}},
                                                              'recall_rand': {'all': {'r5': _rec(_rfreq, 5), 'r20': _rec(_rfreq, 20), 'p5': _prc(_rfreq, 5)},
                                                                              'e25': {'r5': _rec(_rfreq_e25, 5), 'r20': _rec(_rfreq_e25, 20), 'p5': _prc(_rfreq_e25, 5)},
                                                                              'e10': {'r5': _rec(_rfreq_e10, 5), 'r20': _rec(_rfreq_e10, 20), 'p5': _prc(_rfreq_e10, 5)}},
                                                              'recall_loo_e10': {'r5': _lrec(5), 'r20': _lrec(20), 'p5': _lprc(5)},
                                                              'wm_extra': {'policy_elite': 0, 'random_elite': int(_wm_rand), 'loo_e10': int(_wm_loo)},
                                                              'n_drivers_in_cand': int(len(drv_in)),
                                                              'drivers': [g for g in drvset if g in cand_genes],       # disjoint odd-rank (tiny)
                                                              'drivers_full': _full_drv or [g for g in drvset if g in cand_genes],  # ALL experimental drivers (richer)
                                                              'source': data.lineage_drivers[names[li]].get('source', 'joung')}  # 'briggs' if Briggs-manifold fate, else 'joung'
            # specificity: condition on a MISMATCHED goal's z_target, same start state
            _mis = [x for x in TR if x != li] or [x for x in range(K) if x != li]   # single-goal: draw the mismatched goal from all fates
            lj = int(rng_g.choice(_mis))
            t_wrong = data.target_for(keys[lj]).to(dev)
            r_wrong = []
            for _ in range(N // 2):
                _, genes, _, _ = agent.select(s_emb, t_wrong)
                gi = [cand[g] for g in genes]
                if gi:
                    rv = pred_resid(op, data, gi, drift, ctrl_recon, A)
                    o = cent[li]; sl = torch.where(o.abs() > thr)[0]
                    r_wrong.append(float(torch_match(rv.unsqueeze(0), o[sl], sl)[0]))
            _brn = 999                                              # matched random best-of-N null (same pool/k as the policy)
            for _ in range(N):
                _rc = list(rng_g.choice(_cgp, size=min(_kc, len(_cgp)), replace=False))
                _rvn = pred_resid(op, data, _rc, drift, ctrl_recon, A); _can = centn @ (_rvn / (_rvn.norm() + 1e-9))
                _brn = min(_brn, int((_can > _can[li]).sum()))
                if _brn == 0: break
            out['bon_r1'].append(int(_brank == 0)); out['bon_r10'].append(int(_brank < 10))
            out['bon_r1_null'].append(int(_brn == 0)); out['bon_r10_null'].append(int(_brn < 10))
            out['passN'].append(hit_any)
            out['resid_best'].append(max(resids) if resids else float('nan'))
            out['drv_cov'].append(len(cov) / max(1, len(drvset)))
            out['distinct'].append(len(cock))
            out['specificity'].append((float(np.mean(resids)) if resids else 0.0) - (float(np.mean(r_wrong)) if r_wrong else 0.0))
        agg = {k: float(np.nanmean(v)) for k, v in out.items()}
        print(f"[gen_eval {tag}] N={N} | pass@N {agg['passN']:.2f} | resid_best {agg['resid_best']:.3f} | "
              f"drv_cov {agg['drv_cov']:.2f} | distinct {agg['distinct']:.1f} | specificity {agg['specificity']:+.3f} | "
              f"ZERO-SHOT drv_rank(med/{len(cand_genes)}) {agg['drv_rank']:.0f} | drv_recall@20 {agg['drv_recall20']:.2f}", flush=True)
        print(f"[gen_eval {tag}] ELITE-SET readout (rank TFs within top-q% highest-Φ cocktails; +{agg['wm_extra_per_fate']:.0f} WM-queries/fate): "
              f"all-N  r@5 {agg['drv_recall5']:.3f} r@20 {agg['drv_recall20']:.3f} p@5 {agg['prec5']:.3f} | "
              f"elite-25% r@5 {agg['e25_recall5']:.3f} r@20 {agg['e25_recall20']:.3f} p@5 {agg['e25_prec5']:.3f} | "
              f"elite-10% r@5 {agg['e10_recall5']:.3f} r@20 {agg['e10_recall20']:.3f} p@5 {agg['e10_prec5']:.3f}", flush=True)
        print(f"[gen_eval {tag}] BIAS-CONTROL RANDOM+elite (Φ-filter null, +{agg['wm_extra_random']:.0f} WM-q/fate): "
              f"all-N r@5 {agg['rand_all_r5']:.3f} r@20 {agg['rand_all_r20']:.3f} | elite-25% r@5 {agg['rand_e25_r5']:.3f} r@20 {agg['rand_e25_r20']:.3f} | "
              f"elite-10% r@5 {agg['rand_e10_r5']:.3f} r@20 {agg['rand_e10_r20']:.3f} p@5 {agg['rand_e10_p5']:.3f}", flush=True)
        print(f"[gen_eval {tag}] LEAVE-ONE-OUT ΔΦ readout on policy elite-10% (+{agg['wm_extra_loo']:.0f} WM-q/fate): "
              f"r@5 {agg['loo_e10_r5']:.3f} r@20 {agg['loo_e10_r20']:.3f} p@5 {agg['loo_e10_p5']:.3f}  "
              f"(vs raw-freq elite-10% r@5 {agg['e10_recall5']:.3f} r@20 {agg['e10_recall20']:.3f})", flush=True)
        print(f"[gen_eval {tag}] BEST-of-{N} recall@k (goal-specificity over {len(centn)} fates): "
              f"@1 {agg['bon_r1']:.3f} @10 {agg['bon_r10']:.3f}  |  matched-random best-of-{N} null: @1 {agg['bon_r1_null']:.3f} @10 {agg['bon_r10_null']:.3f}", flush=True)
        print(f"[gen_eval {tag}] PER-FATE top-sampled sets (TF:count over N): "+str({k:v for k,v in _perfate_top.items()}), flush=True)
        if wb is not None:
            wb_summary(wb, {f'gen/{tag}/{k}': agg[k] for k in agg})
        return agg

    def diagnostics(planner_idx, agent, tag):
        try:
            return _diagnostics(planner_idx, agent, tag)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[diag {tag}] FAILED: {e}", flush=True); return {}

    def _diagnostics(planner_idx, agent, tag):
        """The multi-goal health panel (logged to wandb): goal-channel load-bearing, σ_within/σ_between SNR,
        action-space coverage. (EV / approx_kl / clip_frac / kl_bc are logged per-update in train_ppo;
        top-k seed stability is aggregated across the sweep's --seed replicates.)"""
        # deterministic proposal set + resid per TRAIN goal
        det = {}
        for li in TR:
            g = [x for x in planner_idx(li) if 0 <= x < len(data.gene_names)]
            det[li] = (set(g), pred_resid(op, data, g, drift, ctrl_recon, A) if g else None)
        # goal-channel load-bearing: mean pairwise Jaccard of proposal SETS across goals (LOW => goal-dependent)
        jac = []
        tl = list(TR)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                a_, b_ = det[tl[i]][0], det[tl[j]][0]
                if a_ or b_:
                    jac.append(len(a_ & b_) / max(1, len(a_ | b_)))
        goal_jac = float(np.mean(jac)) if jac else float('nan')
        # SNR: σ_between (resid across goals) / σ_within (resid across STOCHASTIC samples for one held-out goal)
        rvs = torch.stack([det[li][1] for li in TR if det[li][1] is not None])
        sig_between = float(rvs.std(0).mean())
        li0 = HO[0] if HO else TR[0]; obs0 = env.reset(keys[li0]); within = []
        for _ in range(16):
            _, genes, _, _ = agent.select(obs0[:d], obs0[d:])
            gi = [cand[gp] for gp in genes]
            within.append(pred_resid(op, data, gi, drift, ctrl_recon, A) if gi else torch.zeros_like(rvs[0]))
        sig_within = float(torch.stack(within).std(0).mean())
        snr = sig_between / (sig_within + 1e-9)
        blib = briggs_lib & set(g2i)
        cov = len(set(int(x) for x in cand) & {g2i[g] for g in blib}) / max(1, len(blib))
        topk = {names[li]: sorted(data.gene_names[x] for x in planner_idx(li)
                                  if 0 <= x < len(data.gene_names)) for li in HO}   # held-out proposals (det has only TR)
        print(f"[diag {tag}] goal-load Jaccard={goal_jac:.3f} (low=load-bearing) | SNR σ_btw/σ_wthn={snr:.2f} "
              f"| action-coverage={cov:.2f} | held-out picks={topk}", flush=True)
        if wb is not None:
            wb_summary(wb, {f'diag/{tag}/goal_load_jaccard': goal_jac, f'diag/{tag}/snr_between_within': snr,
                            f'diag/{tag}/action_coverage': cov})
        return dict(goal_load_jaccard=goal_jac, snr=snr, action_coverage=cov, heldout_picks=topk)

    def amortization_report(agent, tag):
        """Goal-1 packaging: ONE goal-conditioned policy for ALL atlas fates at O(1) inference, vs per-goal
        beam-MPC. Reports recall@{1,3,5,10} across ALL K fates + per-goal wall-clock (policy vs MPC) + speedup."""
        import time
        pol = planner_set(agent, op, data, env); beam = mpc_p(a.beam, 2)
        t = time.perf_counter()
        for li in range(K):
            pol(keys[li])
        t_pol = (time.perf_counter() - t) / K
        t = time.perf_counter()
        for li in range(K):
            beam(li)
        t_mpc = (time.perf_counter() - t) / K
        def rk(planner_idx):
            rs = [eval_goal(planner_idx, li) for li in range(K)]; rs = [r for r in rs if r]
            return {k: float(np.mean([r[f'recall{k}'] for r in rs])) for k in (1, 3, 5, 10)}
        pol_rk = rk(lambda li: pol(keys[li])); mpc_rk = rk(beam)
        print(f"[amortization {tag}] ALL {K} atlas fates | inference: policy {t_pol*1e3:.1f} ms/goal vs "
              f"MPC {t_mpc*1e3:.0f} ms/goal ({t_mpc/max(t_pol,1e-6):.0f}x faster)", flush=True)
        print(f"  recall@k  policy: {[round(pol_rk[k],2) for k in (1,3,5,10)]} | MPC: {[round(mpc_rk[k],2) for k in (1,3,5,10)]}", flush=True)
        if wb is not None:
            wb_summary(wb, {f'amort/{tag}/policy_ms': t_pol*1e3, f'amort/{tag}/mpc_ms': t_mpc*1e3, f'amort/{tag}/speedup': t_mpc/max(t_pol,1e-6),
                            **{f'amort/{tag}/policy_recall{k}': pol_rk[k] for k in (1,3,5,10)},
                            **{f'amort/{tag}/mpc_recall{k}': mpc_rk[k] for k in (1,3,5,10)}})
        if a.save_agent:
            torch.save(agent.policy.state_dict(), a.save_agent)
            print(f"  saved atlas policy artifact -> {a.save_agent}", flush=True)

    def mpc_p(beam, maxset):
        def p(li):
            o = cent[li]; sel = torch.where(o.abs() > thr)[0]
            chosen, _ = fate_mpc(op, data, cand, o[sel], sel, drift, ctrl_recon, A, beam=beam, max_set=maxset)
            return chosen
        return p

    def build_demos(idxs):
        """BC demos = per-goal fate-MPC picks (at the policy's kmax => high-k targets), only for the given (train) goals."""
        Kd = int(a.kmax); beam_p = mpc_p(a.beam, Kd); Ss, Ts, TOKs = [], [], []
        for li in idxs:
            obs = env.reset(keys[li]); Ss.append(obs[:d]); Ts.append(obs[d:])
            picks = [cpos[g] for g in beam_p(li) if g in cpos][:Kd]
            TOKs.append([p + 1 for p in picks] + [0] * (Kd - len(picks)))
        return torch.stack(Ss), torch.stack(Ts), torch.tensor(TOKs, dtype=torch.long, device=dev)

    def bc_init(agent, epochs):
        S, T, TOK = build_demos(TR)                                          # TRAIN goals only
        opt = torch.optim.Adam(agent.policy.parameters(), 1e-3)
        for _ in range(epochs):
            logp, ent = agent.policy.evaluate(S, T, TOK); loss = -logp.mean() - 0.001 * ent.mean()
            opt.zero_grad(); loss.backward(); opt.step()

    def train_ppo(agent, iters, tag):
        from collections import Counter
        rng2 = np.random.default_rng(1); r0 = []
        _win, _cum, _tlog = Counter(), Counter(), []       # DEBUG: picked-TF window/cumulative + per-update curve
        N = max(1, int(a.best_of_n)); it = 0; visit = 0; _greward = {}     # best-of-N + per-goal reward tracker (prioritized sampling)
        _ckpt = 0                                                          # periodic-checkpoint marker (wall-out safety + mid-run eval)
        while it < iters:
            if a.prioritized_goals and len(_greward) == len(TR):        # regret-weighted: low running reward => sampled more
                _mu = np.array([np.mean(_greward[g][-32:]) for g in TR]); _w = np.exp(-(_mu - _mu.max()) / 0.5); _w /= _w.sum()
                li = int(rng2.choice(TR, p=_w))
            else:
                li = TR[int(rng2.integers(len(TR)))]
            obs = env.reset(keys[li])  # sample a DIFFERENT goal each visit
            visit += 1                                                      # per-visit group id => quantile taken within THIS visit's N samples
            for _ in range(N):
                it += 1
                tokens, genes, logp, v = agent.select(obs[:d], obs[d:])
                gi = [cand[g] for g in genes]; _, rw, _, _ = env.step(gi)
                for g in gi:                                                # DEBUG: tally picked TFs over training
                    nm = data.gene_names[g]; _win[nm] += 1; _cum[nm] += 1
                if a.beta_contrast > 0 and gi:                              # (b) CONTRASTIVE: reward reaching THIS goal specifically
                    rv = pred_resid(op, data, gi, drift, ctrl_recon, A)
                    cos_all = centn @ (rv / (rv.norm() + 1e-9))             # [K] cosine to every goal centroid
                    if a.contrast_mode == 'infonce':                        # InfoNCE: hard-negative (nearest wrong fate) via the full-fate denominator => rank-1 pressure
                        lse = float(torch.logsumexp(cos_all / a.contrast_tau, 0))
                        rw = rw + a.beta_contrast * (float(cos_all[li]) / a.contrast_tau - lse)
                    elif a.contrast_mode == 'max':                         # margin to the TOP competitor
                        comp = cos_all.clone(); comp[li] = -1e9
                        rw = rw + a.beta_contrast * float(cos_all[li] - comp.max())
                    else:                                                  # mean competitor (soft)
                        rw = rw + a.beta_contrast * float(cos_all[li] - cos_all.mean())
                agent.store(obs[:d], obs[d:], tokens, rw, logp, v, goal_id=li); r0.append(rw); _greward.setdefault(li, []).append(rw)
            if len(agent._buf) >= 128:
                st = agent.update()
                wb_log(wb, {f'{tag}/reward': float(np.mean(r0[-128:])), f'{tag}/explained_variance': st.get('explained_variance', 0.0),
                            f'{tag}/approx_kl': st.get('approx_kl', 0.0), f'{tag}/clip_frac': st.get('clip_frac', 0.0),
                            f'{tag}/entropy': st.get('entropy', 0.0), f'{tag}/reward_std': st.get('reward_std', 0.0),
                            f'{tag}/kl_bc': st.get('kl_bc', 0.0), f'{tag}/lr_actor': st.get('lr_actor', 0.0)}, step=it)
                _top = _win.most_common(8)                                  # DEBUG: dense per-update curve row
                _tlog.append(dict(it=it, reward=float(np.mean(r0[-128:])), EV=float(st.get('explained_variance', 0.0)),
                                  approx_kl=float(st.get('approx_kl', 0.0)), clip_frac=float(st.get('clip_frac', 0.0)),
                                  entropy=float(st.get('entropy', 0.0)), reward_std=float(st.get('reward_std', 0.0)),
                                  n_distinct_win=len(_win), top_picks='|'.join(f'{k}:{v}' for k, v in _top)))
                _win = Counter()
                if it % 1280 == 0:
                    print(f"  [{tag}] it {it:5d} reward {np.mean(r0[-500:]):+.3f} EV {st.get('explained_variance',0):+.2f} "
                          f"kl {st.get('approx_kl',0):.3f} clip {st.get('clip_frac',0):.2f}", flush=True)
                if a.save_scratch and tag == 'setppo' and it - _ckpt >= 100000:  # periodic checkpoint: survive wall-outs + enable mid-run eval
                    torch.save(agent.policy.state_dict(), a.save_scratch); _ckpt = it
                    print(f"  [{tag}] checkpoint @ it {it} -> {a.save_scratch}", flush=True)
        _sfx_t = (a.run_name or a.heldout_substr).replace('/', '_')          # DEBUG: dump curve + cumulative picks
        pd.DataFrame(_tlog).to_csv(f'{EXP20}/results/briggs/rl_trainlog_{tag}_{_sfx_t}.csv', index=False)
        _cc = _cum.most_common()
        pd.DataFrame(_cc, columns=['TF', 'count']).to_csv(f'{EXP20}/results/briggs/rl_pickcum_{tag}_{_sfx_t}.csv', index=False)
        print(f"  [{tag}] TOP-15 cumulative picks over training: {_cc[:15]}", flush=True)
        print(f"  [{tag}] saved training log + cumulative picks (tag={tag}, sfx={_sfx_t})", flush=True)
        return agent

    # ================= RETRIEVER / cosine gate: build a per-goal answer-agnostic shortlist -> env._goal_gate =================
    # (applies at BOTH training and eval; the trainers read env._goal_gate, eval sets policy._allowed_block per goal)
    if getattr(a, 'cosine_gate', False):
        from analysis.hstep_mdp import cosine_cand_mask
        _gg = {}; _zs = None; _kf = float(a.cosine_gate_keep)
        for _li in range(K):
            env.reset(keys[_li]); _al, _zs = cosine_cand_mask(op, data, env, cand, A, keep_frac=_kf, zsingle=_zs)
            _gg[_li] = _al
        env._goal_gate = _gg
        _n2l = {data.gene_names[int(cand[i])]: i for i in range(len(cand))}
        _hr = []
        for _li in HO:
            _drv = set(data.lineage_drivers[names[_li]].get('train', []) + data.lineage_drivers[names[_li]].get('test', []) + data.lineage_drivers[names[_li]].get('full', []))
            _dl = {_n2l[g] for g in _drv if g in _n2l}
            if _dl: _hr.append(len(_dl & _gg[_li]) / len(_dl))
        print(f"[cosine_gate] per-goal shortlist k={max(1,int(_kf*len(cand)))}/{len(cand)} | HO driver-recall(circular)={np.mean(_hr) if _hr else float('nan'):.3f} | answer-agnostic single-TF-Φ shortlist", flush=True)
    elif getattr(a, 'learned_retriever', False):
        from analysis.retriever import build_retriever_gate
        _gg, _rstats = build_retriever_gate(op, data, cand, cent, TR, HO, names, dev,
                                            epochs=int(a.retriever_epochs), keep=float(a.retriever_keep), hops=int(a.retriever_hops))
        env._goal_gate = _gg

    # ================= EVAL-ONLY: load a saved policy, run policy/MPC/greedy ladder + timing on BOTH fate sets =================
    if getattr(a, 'eval_only', False):
        import time as _time
        assert a.load_policy, "--eval_only requires --load_policy PATH"
        ag_e = SetPPO(state_dim=d, n_candidates=len(cand), k_max=a.kmax, device=dev, **_gh)
        sd = torch.load(a.load_policy, map_location=dev)
        if isinstance(sd, dict) and 'policy' in sd: sd = sd['policy']
        ag_e.policy.load_state_dict(sd); ag_e.policy.eval()
        print(f"[eval_only] loaded policy <- {a.load_policy} | in-domain(train) {len(TR)} fates | held-out {len(HO)} fates", flush=True)
        if getattr(a, 'compute_frontier', False):                    # ===== COMPUTE-MATCHED FRONTIER: MPC beam-budget sweep on held-out BRIGGS (policy-independent) =====
            import time as _tf, os as _osf
            _K_short = int(getattr(a, 'mpc_shortlist_k', 20))
            cand_genes = {data.gene_names[int(i)] for i in cand}; C = len(cand)
            def _drv_recall_at(freq, drivers, k):                    # EXACT rl_eval_suite.drv_recall_at (freq-ranked top-k ∩ drivers / |drivers|)
                drv = set(drivers)
                if not drv or not freq: return float('nan')
                ranked = sorted(freq, key=lambda g: -freq[g]); return len(set(ranked[:k]) & drv) / len(drv)
            def _cap_for(li):                                        # EXACT per-fate cocktail cap the env applies to the policy (matches --headtohead's _mpc_core)
                try: env.reset(keys[li])
                except Exception: pass
                return int(getattr(env, 'kmax_active', None) or a.kmax)
            _br_src = lambda li: data.lineage_drivers[names[li]].get('source', 'joung') == 'briggs'
            briggs_HO = ([li for li in range(len(names)) if _br_src(li)] if getattr(a, 'frontier_all_briggs', False)
                         else [li for li in HO if _br_src(li)])
            beams = [int(b) for b in str(a.frontier_beams).split(',') if b.strip()]
            print(f"[compute_frontier] C(pool)={C} | {'ALL' if getattr(a,'frontier_all_briggs',False) else 'held-out'} BRIGGS fates={len(briggs_HO)} | beams={beams} | shortlist_k={_K_short} "
                  f"| NOTE MPC uses the default (Joung-frame) fate_mpc, matching the --headtohead baseline (reproduces paired MPC_exact)", flush=True)
            fr_rows = []
            for b in beams:
                per = {k: [] for k in ['drv_recall20', 'resid_match', 'phi', 'recall@1', 'recall@5', 'recall@10', 'wm_queries', 'cap']}
                t0 = _tf.perf_counter()
                for li in briggs_HO:
                    cap = _cap_for(li); o = cent[li]; sel = torch.where(o.abs() > thr)[0]
                    if len(sel) == 0: continue
                    chosen, mscore, ranked = fate_mpc(op, data, cand, o[sel], sel, drift, ctrl_recon, A,
                                                      beam=b, max_set=cap, return_ranked=True)   # default Joung frame (no eps0), == headtohead MPC
                    r = eval_goal(lambda _li, _c=chosen: _c, li)
                    if r is None: continue
                    freq = {}                                        # descending int weights => freq-rank == model-score rank (same as headtohead dump)
                    for _r, g in enumerate(ranked[:_K_short]):
                        nm2 = data.gene_names[int(g)]
                        if nm2 not in freq: freq[nm2] = _K_short - _r
                    drvset = set(data.lineage_drivers[names[li]]['train'] + data.lineage_drivers[names[li]]['test'])
                    drivers_incand = [g for g in drvset if g in cand_genes]
                    per['drv_recall20'].append(_drv_recall_at(freq, drivers_incand, 20))
                    per['resid_match'].append(r['resid']); per['phi'].append(float(mscore))
                    per['recall@1'].append(r['recall1']); per['recall@5'].append(r['recall5']); per['recall@10'].append(r['recall10'])
                    per['wm_queries'].append(C * (1 + (cap - 1) * b)); per['cap'].append(cap)
                ms = (_tf.perf_counter() - t0) / max(1, len(briggs_HO)) * 1e3
                row = dict(run='MPC_beam', planner='MPC_beam', fold=a.fold, seed=a.seed, source='briggs', beam=b,
                           k_mean=round(float(np.mean(per['cap'])), 2) if per['cap'] else float('nan'),
                           n_fates=len(per['drv_recall20']), ms_per_goal=round(ms, 2),
                           wm_queries=round(float(np.mean(per['wm_queries'])), 1) if per['wm_queries'] else float('nan'),
                           **{m: (round(float(np.nanmean(per[m])), 4) if per[m] else float('nan'))
                              for m in ['drv_recall20', 'resid_match', 'phi', 'recall@1', 'recall@5', 'recall@10']})
                fr_rows.append(row)
                print(f"[frontier] beam={b:>3d} k~{row['k_mean']} | WMq/goal~{row['wm_queries']:.0f} | {row['ms_per_goal']:.0f} ms/goal | "
                      f"drv_recall@20 {row['drv_recall20']:.3f} | resid {row['resid_match']:.3f} | recall@10 {row['recall@10']:.3f} (n={row['n_fates']})", flush=True)
            _fdir = f'{EXP20}/results/rl_eval_condz'; _osf.makedirs(_fdir, exist_ok=True)
            fout = f'{_fdir}/compute_frontier_{(a.run_name or "policy").replace("/", "_")}.csv'
            pd.DataFrame(fr_rows).to_csv(fout, index=False); print(f"\n[compute_frontier] saved -> {fout}", flush=True)
            return
        sp = planner_set(ag_e, op, data, env)
        if getattr(a, 'distrib_check', False):                       # OPTIONAL GFlowNet distributional + diversity analysis (default off; reuses this exact stack; writes its own CSVs; skips the rest of eval)
            sys.path.insert(0, EXP20)
            from evaluation.rl_gflownet_distribution import run_from_stack as _distrib_run
            _btn = None
            if int(getattr(a, 'distrib_beam_topn', 0)) > 0:                # BEAM-TOP-N: search's N-set frontier as the fair diversity baseline
                def _btn(li, _N=int(a.distrib_beam_topn)):
                    o = cent[li]; sel = torch.where(o.abs() > thr)[0]
                    _, _, _sets = fate_mpc(op, data, cand, o[sel], sel, drift, ctrl_recon, A, beam=_N, max_set=int(a.kmax), return_sets=True)
                    return [s for s, _sc in _sets]                         # list of gene-index cocktails (score-ranked)
            _distrib_run(a, ag_e, op, data, env, cand, keys, names, cent, centn, thr, drift, ctrl_recon, A, d, HO, TR, dev, pred_resid, mpc_p, beam_topn_planner=_btn)
            wb_finish(wb) if wb is not None else None
            return
        planners = [('policy (amortized)', 'pol', lambda li: sp(keys[li])),   # O(1) forward pass
                    ('MPC (beam)',         'mpc', mpc_p(a.beam, 2)),           # beam search over the WM (reference)
                    ('greedy (beam=1)',    'grd', mpc_p(1, 2))]                # 1-step greedy forward selection (same k, beam=1)
        def _rk_set(planner_idx, fates):
            rs = [r for li in fates if (r := eval_goal(planner_idx, li))]
            if not rs: return {}
            return {**{f'recall@{k}': float(np.mean([r[f'recall{k}'] for r in rs])) for k in (1, 3, 5, 10)},
                    'resid_match': float(np.mean([r['resid'] for r in rs])),
                    'driver_hit': float(np.mean([r['drv_hit'] for r in rs]))}
        def _time_set(planner_idx, fates, reps):
            t = _time.perf_counter()
            for _ in range(reps):
                for li in fates: planner_idx(li)
            return (_time.perf_counter() - t) / (reps * len(fates))
        rows = []
        for domain, fates in [('in-domain(train)', list(TR)), ('held-out', list(HO))]:
            for name, short, pidx in planners:
                rk = _rk_set(pidx, fates)
                ms = _time_set(pidx, fates, 20 if short == 'pol' else 1) * 1e3   # policy is ~ms => average more reps
                ge = gen_eval(ag_e, f'{domain}:{name}', N=a.gen_eval_n, fates=fates) if (short == 'pol' and a.gen_eval_n > 0) else {}
                rows.append(dict(domain=domain, planner=name, n_fates=len(fates), ms_per_goal=round(ms, 2),
                                 **{k: round(v, 4) for k, v in rk.items()},
                                 passN=round(ge.get('passN', float('nan')), 4), resid_best=round(ge.get('resid_best', float('nan')), 4),
                                 specificity=round(ge.get('specificity', float('nan')), 4), drv_recall20=round(ge.get('drv_recall20', float('nan')), 4)))
                print(f"  [{domain}] {name:20s} {ms:9.2f} ms/goal | {rk}", flush=True)
        edf = pd.DataFrame(rows)
        eout = f'{EXP20}/results/briggs/eval_only_{(a.run_name or "policy").replace("/", "_")}.csv'
        edf.to_csv(eout, index=False); print(f"\n{edf.to_string()}\nsaved -> {eout}", flush=True)
        if getattr(a, 'conservative_bon', False):                    # (a) CONSERVATIVE BEST-OF-N reranking (NO retraining): deploy = argmax_n of a robust score over the DAG ensemble, not greedy
            eps0 = data.eps_ctrl_mean.unsqueeze(0).to(dev); Kd = a.n_dag; lam = a.alpha or 1.0
            dags = op.sample_dags(Kd)
            def _cbon(fates, N=256):
                acc = {k: [] for k in ['mean_k(vanillaBoN)', 'WCO(min_k)', f'UWO(mean-{lam}std)']}
                for li in fates:
                    obs = env.reset(keys[li]); s, t = obs[:d], obs[d:]
                    Ss, PK = [], []                                  # per-sample: gene set, Φ over K DAGs
                    for _ in range(N):
                        _, gg, _, _ = ag_e.select(s, t); gi = [cand[x] for x in gg]
                        if not gi: continue
                        zk, _, _ = op.apply_ensemble(eps0.repeat(Kd, 1), gi, dags)   # [K,d]
                        Ss.append(gi); PK.append(env._phi(zk))       # [K]
                    if not Ss: continue
                    PK = torch.stack(PK)                             # [n_samp, K]
                    picks = {'mean_k(vanillaBoN)': int(PK.mean(1).argmax()),        # best-of-N by mean Φ
                             'WCO(min_k)': int(PK.min(1).values.argmax()),          # worst-case-over-ensemble (robust)
                             f'UWO(mean-{lam}std)': int((PK.mean(1) - lam * PK.std(1)).argmax())}   # uncertainty-weighted
                    for rule, ni in picks.items():
                        rv = pred_resid(op, data, Ss[ni], drift, ctrl_recon, A)
                        acc[rule].append(int(float((centn @ (rv / (rv.norm() + 1e-9))).argmax()) == li))   # recall@1
                return {r: float(np.mean(v)) if v else float('nan') for r, v in acc.items()}
            for dom, fs in [('held-out', list(HO)), ('in-domain', list(TR))]:
                r1 = _cbon(fs, N=a.gen_eval_n)
                print(f"[conservative-BoN {dom}] recall@1 by deploy rule (N={a.gen_eval_n}, K={Kd} DAGs): {r1}", flush=True)
        if getattr(a, 'multi_wm_bon', False) and a.wm_ensemble:       # (b) CROSS-WM conservative BEST-OF-N: rerank the policy's samples by min_k Φ over K WMs (different split seeds), not DAGs
            from core.wm_ensemble import WMEnsemble
            ens = WMEnsemble(op, data, [c for c in a.wm_ensemble.split(',') if c], a.wm_root, dev, n_dag=a.n_dag)
            lam = a.alpha or 1.0; Kw = ens.K
            def _mwbon(fates, N=256):
                acc = {'mean_k(vanillaBoN)': [], 'WCO(min_k-WM)': [], f'UWO(mean-{lam}std-WM)': [], 'greedy(baseWM)': []}
                agree = []                                            # cross-WM agreement of the base-WM argmax's Φ rank (diagnostic)
                for li in fates:
                    obs = env.reset(keys[li]); s, t = obs[:d], obs[d:]
                    o_res, sel, dr = env._o_res, env._sel_res, env._drift
                    Ss, PK = [], []
                    for _ in range(N):
                        _, gg, _, _ = ag_e.select(s, t); gi = [cand[x] for x in gg]
                        if not gi: continue
                        Ss.append(gi); PK.append(ens.phi_all(gi, o_res, sel, dr))    # [Kw]
                    if not Ss: continue
                    PK = torch.stack(PK)                              # [n_samp, Kw]
                    picks = {'mean_k(vanillaBoN)': int(PK.mean(1).argmax()),
                             'WCO(min_k-WM)': int(PK.min(1).values.argmax()),         # robust: best worst-case-over-WMs
                             f'UWO(mean-{lam}std-WM)': int((PK.mean(1) - lam * PK.std(1)).argmax()),
                             'greedy(baseWM)': int(PK[:, 0].argmax())}                # rerank by base WM only (control)
                    for rule, ni in picks.items():
                        rv = pred_resid(op, data, Ss[ni], dr, ctrl_recon, A)
                        acc[rule].append(int(float((centn @ (rv / (rv.norm() + 1e-9))).argmax()) == li))   # recall@1
                    agree.append(float(PK.std(1).mean()))             # mean cross-WM Φ std over samples
                out = {r: float(np.mean(v)) if v else float('nan') for r, v in acc.items()}
                out['xWM_phi_std'] = float(np.mean(agree)) if agree else float('nan')
                return out
            for dom, fs in [('held-out', list(HO)), ('in-domain', list(TR))]:
                print(f"[multi-WM BoN {dom}] recall@1 by deploy rule (N={a.gen_eval_n}, K={Kw} WMs): {_mwbon(fs, N=a.gen_eval_n)}", flush=True)
        if getattr(a, 'mpc_ksweep', False):                          # VALUE-PROP: MPC cost grows with k; the amortized policy is O(1) — find the crossover
            import time as _t
            fates = list(HO); rows_k = []
            t = _t.perf_counter()
            for _ in range(20):
                for li in fates: sp(keys[li])
            pol_ms = (_t.perf_counter() - t) / (20 * len(fates)) * 1e3; pk = _rk_set(lambda li: sp(keys[li]), fates)
            print(f"[k-sweep] policy (O(1) amortized): {pol_ms:.1f} ms/goal | recall@10 {pk.get('recall@10', 0):.3f} resid {pk.get('resid_match', 0):.3f}", flush=True)
            rows_k.append(dict(planner='policy', k=0, ms_per_goal=round(pol_ms, 2), **{kk: round(v, 4) for kk, v in pk.items()}))
            for k in [2, 3, 4, 5, 6]:
                mp = mpc_p(a.beam, k)
                t = _t.perf_counter()
                for li in fates: mp(li)
                mms = (_t.perf_counter() - t) / len(fates) * 1e3; rk = _rk_set(mp, fates)
                print(f"[k-sweep] MPC beam={a.beam} k={k}: {mms:.0f} ms/goal ({mms/max(pol_ms,1e-6):.0f}× policy) | recall@10 {rk.get('recall@10', 0):.3f} resid {rk.get('resid_match', 0):.3f}", flush=True)
                rows_k.append(dict(planner=f'MPC_k{k}', k=k, ms_per_goal=round(mms, 2), **{kk: round(v, 4) for kk, v in rk.items()}))
                if mms > 5000:
                    print(f"[k-sweep] MPC k={k} exceeds 5s/goal — intractable beyond here; stopping sweep", flush=True); break
            pd.DataFrame(rows_k).to_csv(f'{EXP20}/results/briggs/ksweep_{(a.run_name or "policy").replace("/", "_")}.csv', index=False)
        if getattr(a, 'mpc_honest', False):                          # (a) HELD-OUT-HONEST MPC: search vs the nearest TRAINING fates' centroid, not the exact held-out one; recall vs TRUTH
            knb = int(a.mpc_honest_k); TRl = list(TR)
            centn_TR = centn[torch.tensor(TRl, device=dev)]          # [|TR|,G] unit training centroids
            def _honest_target(li):
                cs = centn_TR @ centn[li]                            # cosine(held-out fate, each training fate) — the neighbor query
                nb = torch.topk(cs, min(knb, len(TRl))).indices
                w = torch.softmax(cs[nb], 0)                         # cosine-weighted neighbor average of RAW training centroids
                return (w[:, None] * cent[torch.tensor([TRl[int(j)] for j in nb], device=dev)]).sum(0)
            def _mpc_honest(li):
                o = _honest_target(li); sel = torch.where(o.abs() > thr)[0]
                if len(sel) == 0: return []
                chosen, _ = fate_mpc(op, data, cand, o[sel], sel, drift, ctrl_recon, A, beam=a.beam, max_set=a.kmax)
                return chosen
            rows_h = []
            for nm, pl in [(f'MPC-exact(k{a.kmax})', mpc_p(a.beam, a.kmax)),
                           (f'MPC-honest(nbr{knb})', _mpc_honest),
                           ('policy(amortized)', lambda li: sp(keys[li]))]:
                rk = _rk_set(pl, list(HO))
                print(f"[held-out-honest {nm:20s}] recall@1 {rk.get('recall@1',0):.3f} @5 {rk.get('recall@5',0):.3f} "
                      f"@10 {rk.get('recall@10',0):.3f} resid {rk.get('resid_match',0):.3f} drv_hit {rk.get('driver_hit',0):.3f}", flush=True)
                rows_h.append(dict(planner=nm, **{k: round(v, 4) for k, v in rk.items()}))
            pd.DataFrame(rows_h).to_csv(f'{EXP20}/results/briggs/mpc_honest_{(a.run_name or "policy").replace("/", "_")}.csv', index=False)
        if getattr(a, 'pg_mpc', False):                              # (b) POLICY-GUIDED MPC: expand only the policy's top-M proposals per beam step (learned prior), WM scores/prunes
            import time as _tp
            M = int(a.pg_mpc_m); ed = ag_e.policy.emb_dim; beam = a.beam; maxset = int(a.kmax)
            eps0 = data.eps_ctrl_mean; S_all = op.shift_for(cand)     # [C,d] per-candidate latent shift
            def _pg_mpc(li, count=None):
                o = cent[li]; sel = torch.where(o.abs() > thr)[0]; o_t = o[sel]
                if len(sel) == 0: return []
                zg = env.reset(keys[li])[d:]
                z0 = op.apply(eps0.unsqueeze(0), [], A)[0].squeeze(0)
                phi0 = float(torch_match((op.decode(z0.unsqueeze(0)) - ctrl_recon) - drift, o_t, sel)[0])
                beams = [([], torch.zeros(d, device=dev), z0, phi0)]; best = beams[0]
                for _step in range(maxset):
                    exps = []
                    for gl, shift, zt, _sc in beams:
                        h = ag_e.policy.encode(zt.unsqueeze(0), zg.unsqueeze(0))
                        cmask = torch.zeros(1, len(cand), dtype=torch.bool, device=dev)
                        if gl: cmask[0, torch.tensor(gl, device=dev)] = True
                        logits = ag_e.policy._slot_logits(h, torch.zeros(1, ed, device=dev), len(gl), cmask)
                        topM = torch.topk(logits[0, 1:], min(M, len(cand))).indices.tolist()   # policy's top-M candidates (STOP excluded)
                        topM = [m for m in topM if m not in gl]
                        if not topM: continue
                        Sb = shift.unsqueeze(0) + S_all[torch.tensor(topM, device=dev)]         # [m,d] cumulative shift per expansion
                        zb = op.apply_many(eps0, Sb, A)                                         # [m,d] batched WM solve (the scoring)
                        scr = torch_match((op.decode(zb) - ctrl_recon) - drift, o_t, sel)       # [m]
                        if count is not None: count[0] += len(topM)
                        for j, m in enumerate(topM):
                            exps.append((gl + [m], Sb[j], zb[j], float(scr[j])))
                    if not exps: break
                    exps.sort(key=lambda b: -b[3]); beams = exps[:beam]
                    if beams[0][3] > best[3]: best = beams[0]
                return [cand[g] for g in best[0]]
            fates = list(HO); qc = [0]
            t0 = _tp.perf_counter()
            rk_pg = _rk_set(lambda li: _pg_mpc(li, qc), fates)
            pg_ms = (_tp.perf_counter() - t0) / len(fates) * 1e3; pg_q = qc[0] / len(fates)
            t0 = _tp.perf_counter(); rk_mpc = _rk_set(mpc_p(a.beam, maxset), fates)
            mpc_ms = (_tp.perf_counter() - t0) / len(fates) * 1e3; mpc_q = maxset * a.beam * len(cand)
            t0 = _tp.perf_counter(); rk_pol = _rk_set(lambda li: sp(keys[li]), fates)
            pol_ms = (_tp.perf_counter() - t0) / len(fates) * 1e3
            print(f"[pg-MPC] policy-guided (M={M}): {pg_ms:.0f} ms/goal | ~{pg_q:.0f} cand-evals/goal | recall@1 {rk_pg.get('recall@1',0):.3f} @10 {rk_pg.get('recall@10',0):.3f} resid {rk_pg.get('resid_match',0):.3f}", flush=True)
            print(f"[pg-MPC] plain MPC (k{maxset}) : {mpc_ms:.0f} ms/goal | ~{mpc_q} cand-evals/goal | recall@1 {rk_mpc.get('recall@1',0):.3f} @10 {rk_mpc.get('recall@10',0):.3f} resid {rk_mpc.get('resid_match',0):.3f}", flush=True)
            print(f"[pg-MPC] policy (O(1))      : {pol_ms:.0f} ms/goal | recall@1 {rk_pol.get('recall@1',0):.3f} @10 {rk_pol.get('recall@10',0):.3f} resid {rk_pol.get('resid_match',0):.3f}", flush=True)
            print(f"[pg-MPC] SPEEDUP vs plain MPC: {mpc_ms/max(pg_ms,1e-6):.1f}× wall / {mpc_q/max(pg_q,1e-6):.1f}× fewer cand-evals; quality Δrecall@10 {rk_pg.get('recall@10',0)-rk_mpc.get('recall@10',0):+.3f}", flush=True)
            pd.DataFrame([dict(planner='pg_mpc', ms=round(pg_ms,1), cand_evals=round(pg_q), **{k:round(v,4) for k,v in rk_pg.items()}),
                          dict(planner='plain_mpc', ms=round(mpc_ms,1), cand_evals=mpc_q, **{k:round(v,4) for k,v in rk_mpc.items()}),
                          dict(planner='policy', ms=round(pol_ms,1), cand_evals=0, **{k:round(v,4) for k,v in rk_pol.items()})]
                         ).to_csv(f'{EXP20}/results/briggs/pg_mpc_{(a.run_name or "policy").replace("/", "_")}.csv', index=False)
            # ---- ADDITIVE (eval-only, guarded by --pg_mpc): drv_recall@20 frontier point for pg_mpc on held-out BRIGGS ----
            # emits a top-K freq-ranked shortlist of pg_mpc's SCORED candidates (best Φ each), scored by the EXACT
            # rl_eval_suite.drv_recall_at formula (matches compute_frontier), so it overlays on compute_frontier.csv.
            import os as _ospg
            _K_short_pg = int(getattr(a, 'mpc_shortlist_k', 20))
            _cand_genes_pg = {data.gene_names[int(i)] for i in cand}
            def _drv_recall_at_pg(freq, drivers, k):                      # EXACT compute_frontier / rl_eval_suite formula
                drv = set(drivers)
                if not drv or not freq: return float('nan')
                rk = sorted(freq, key=lambda g: -freq[g]); return len(set(rk[:k]) & drv) / len(drv)
            def _pg_mpc_ranked(li, cnt):                                  # same search as _pg_mpc; also returns score-ranked shortlist of all scored cands
                o = cent[li]; sel = torch.where(o.abs() > thr)[0]; o_t = o[sel]
                if len(sel) == 0: return None
                zg = env.reset(keys[li])[d:]
                z0 = op.apply(eps0.unsqueeze(0), [], A)[0].squeeze(0)
                phi0 = float(torch_match((op.decode(z0.unsqueeze(0)) - ctrl_recon) - drift, o_t, sel)[0])
                beams = [([], torch.zeros(d, device=dev), z0, phi0)]; best = beams[0]; cand_best = {}
                for _step in range(maxset):
                    exps = []
                    for gl, shift, zt, _sc in beams:
                        h = ag_e.policy.encode(zt.unsqueeze(0), zg.unsqueeze(0))
                        cmask = torch.zeros(1, len(cand), dtype=torch.bool, device=dev)
                        if gl: cmask[0, torch.tensor(gl, device=dev)] = True
                        logits = ag_e.policy._slot_logits(h, torch.zeros(1, ed, device=dev), len(gl), cmask)
                        topM = torch.topk(logits[0, 1:], min(M, len(cand))).indices.tolist()
                        topM = [m for m in topM if m not in gl]
                        if not topM: continue
                        Sb = shift.unsqueeze(0) + S_all[torch.tensor(topM, device=dev)]
                        zb = op.apply_many(eps0, Sb, A)
                        scr = torch_match((op.decode(zb) - ctrl_recon) - drift, o_t, sel)
                        cnt[0] += len(topM)                               # WM candidate-evaluations (== compute_frontier wm_queries semantics)
                        for j, m in enumerate(topM):
                            sc = float(scr[j])
                            if m not in cand_best or sc > cand_best[m]: cand_best[m] = sc
                            exps.append((gl + [m], Sb[j], zb[j], sc))
                    if not exps: break
                    exps.sort(key=lambda b: -b[3]); beams = exps[:beam]
                    if beams[0][3] > best[3]: best = beams[0]
                return sorted(cand_best, key=lambda m: -cand_best[m])     # cand-idx list, best-Φ descending
            _br_src_pg = lambda li: data.lineage_drivers[names[li]].get('source', 'joung') == 'briggs'
            briggs_HO_pg = [li for li in HO if _br_src_pg(li)]
            drv_list = []; cntp = [0]; t0 = _tp.perf_counter()
            for li in briggs_HO_pg:
                ranked = _pg_mpc_ranked(li, cntp)
                if ranked is None: continue
                freq = {}
                for _r, g in enumerate(ranked[:_K_short_pg]):
                    nm2 = data.gene_names[int(g)]
                    if nm2 not in freq: freq[nm2] = _K_short_pg - _r      # descending int weights => freq-rank == score-rank
                drvset = set(data.lineage_drivers[names[li]]['train'] + data.lineage_drivers[names[li]]['test'])
                drivers_incand = [g for g in drvset if g in _cand_genes_pg]
                drv_list.append(_drv_recall_at_pg(freq, drivers_incand, 20))
            n_br = max(1, len(briggs_HO_pg)); drv_ms = (_tp.perf_counter() - t0) / n_br * 1e3; drv_q = cntp[0] / n_br
            drv20 = float(np.nanmean(drv_list)) if drv_list else float('nan')
            print(f"[pg-MPC drv] M={M} | held-out BRIGGS fates={len(drv_list)} | drv_recall@20 {drv20:.4f} | WMq/goal ~{drv_q:.0f} | {drv_ms:.0f} ms/goal", flush=True)
            _frdir = f'{EXP20}/results/rl_eval_condz'; _ospg.makedirs(_frdir, exist_ok=True)
            pd.DataFrame([dict(run=(a.run_name or 'policy'), fold=a.fold, seed=a.seed, M=M, drv_recall20=round(drv20, 4),
                               wm_queries_per_goal=round(drv_q, 1), ms_per_goal=round(drv_ms, 1), n_fates=len(drv_list))]
                        ).to_csv(f'{_frdir}/pg_mpc_drvfrontier_{(a.run_name or "policy").replace("/", "_")}.csv', index=False)
        if getattr(a, 'headtohead', False):                          # MPC-exact + ADDITIVE-GREEDY baselines head-to-head with the policy (hard ladder + per-fate dump for SOFT + HSPC)
            cand_genes = {data.gene_names[int(i)] for i in cand}
            _sel_of = lambda o: torch.where(o.abs() > thr)[0]
            # additive-greedy: greedily sum OBSERVED drift-removed single-TF Δ (de_single − drift) — NO WM interaction head / SCM
            avail, Xrows = [], []
            for c in cand:
                nm_ = data.gene_names[int(c)]
                if nm_ in data.de_single:
                    avail.append(int(c)); Xrows.append(torch.tensor(np.asarray(data.de_single[nm_]['lfc'], np.float32), device=dev) - drift)
            Xmat = torch.stack(Xrows) if Xrows else None            # [C',G] observed residual singles
            _availpos = {int(c): j for j, c in enumerate(avail)}
            # WITHIN-SHORTLIST head-to-head: restrict every baseline to the SAME per-goal retriever shortlist as the policy,
            # so the comparison isolates SELECTION-within-pool (is the RL policy doing anything the retriever+search/random isn't?)
            _GGh = getattr(env, '_goal_gate', None)
            _rngh = np.random.default_rng(0)
            def _subcand(li):                                        # global cand indices allowed for this goal (shortlist or full)
                if _GGh is not None and li in _GGh: return [int(cand[i]) for i in sorted(_GGh[li])]
                return [int(c) for c in cand]
            def _cap_for(li):                                        # EXACT per-source/per-fate cocktail cap the ENV applies to the policy (==a.kmax unless set_kmax/set_kmax_fate installed via --frame_route/--kmax_joung/--mask_stop)
                try: env.reset(hspc_key if li is None else keys[li])  # li None => the dedicated HSPC goal
                except Exception: pass
                return int(getattr(env, 'kmax_active', None) or a.kmax)
            _K_short = int(getattr(a, 'mpc_shortlist_k', 20))        # breadth-match shortlist depth (freq-ranked recall budget)
            print(f"[headtohead] additive pool = {len(avail)}/{len(cand)} | gated-to-shortlist = {_GGh is not None} | shortlist_k={_K_short} | caps honor env.kmax_active", flush=True)
            def _add_core(o_t, sel, li, maxset=None, beam=a.beam):
                if maxset is None: maxset = _cap_for(li)             # per-source/per-fate cap (matches the policy), not the flat a.kmax
                if Xmat is None or len(sel) == 0: return []
                sc = set(_subcand(li)); rows = [_availpos[c] for c in sc if c in _availpos]  # additive over shortlist∩observed-single
                if not rows: return []
                Xs = Xmat[rows]; av = [avail[j] for j in rows]
                m1 = torch_match(Xs, o_t, sel); top = torch.topk(m1, min(beam, len(av))).indices.tolist()
                beams = [([av[i]], Xs[i].clone(), float(m1[i])) for i in top]; best = max(beams, key=lambda b: b[2])
                for _ in range(2, maxset + 1):
                    ext = []
                    for idxs, cur, _ in beams:
                        ms = torch_match(cur.unsqueeze(0) + Xs, o_t, sel)
                        for p_, c_ in enumerate(av):
                            if c_ in idxs: ms[p_] = -1e9
                        for j in torch.topk(ms, min(beam, len(av))).indices.tolist():
                            ext.append((idxs + [av[j]], cur + Xs[j], float(ms[j])))
                    if not ext: break
                    ext.sort(key=lambda b: -b[2]); beams = ext[:beam]
                    if beams[0][2] > best[2]: best = beams[0]
                return best[0]
            def _mpc_core(o_t, sel, li):
                if len(sel) == 0: return []
                chosen, _ = fate_mpc(op, data, _subcand(li), o_t, sel, drift, ctrl_recon, A, beam=a.beam, max_set=_cap_for(li)); return chosen
            def _rand_core(o_t, sel, li):                            # random-within-shortlist reference (matched k)
                sc = _subcand(li); k = min(_cap_for(li), len(sc))
                return list(_rngh.choice(sc, size=k, replace=False)) if sc else []
            # ── ranked ACCESSORS: (best_set, top-K freq-ranked shortlist) — breadth-matched to the policy best-of-N ──
            def _mpc_ranked(o_t, sel, li):
                if len(sel) == 0: return [], []
                chosen, _, ranked = fate_mpc(op, data, _subcand(li), o_t, sel, drift, ctrl_recon, A,
                                             beam=a.beam, max_set=_cap_for(li), return_ranked=True)
                return chosen, ranked[:_K_short]
            def _add_ranked(o_t, sel, li):                           # additive: rank the shortlist candidates by single-TF additive match
                best = _add_core(o_t, sel, li)
                if Xmat is None or len(sel) == 0: return best, list(best)
                sc = set(_subcand(li)); rows = [_availpos[c] for c in sc if c in _availpos]
                if not rows: return best, list(best)
                Xs = Xmat[rows]; av = [avail[j] for j in rows]
                order = [av[i] for i in torch.argsort(torch_match(Xs, o_t, sel), descending=True).tolist()]
                return best, list(dict.fromkeys([int(g) for g in list(best) + order]))[:_K_short]
            def _rand_ranked(o_t, sel, li):                          # random: matched-breadth random-K draw from the same pool
                best = _rand_core(o_t, sel, li); sc = _subcand(li)
                draw = list(_rngh.choice(sc, size=min(_K_short, len(sc)), replace=False)) if sc else []
                return best, list(dict.fromkeys([int(g) for g in list(best) + draw]))[:_K_short]
            cores = {'MPC-exact': _mpc_core, 'additive-greedy': _add_core, 'random-shortlist': _rand_core}
            ranked_of = {'MPC-exact': _mpc_ranked, 'additive-greedy': _add_ranked, 'random-shortlist': _rand_ranked}
            for nm_, core in cores.items():
                pl = lambda li, _c=core: _c(cent[li][_sel_of(cent[li])], _sel_of(cent[li]), li)   # PLAIN best set: reach ladder + HSPC branch stay on this
                racc = ranked_of[nm_]
                rk = _rk_set(pl, list(HO))
                print(f"[headtohead {nm_:16s}] recall@1 {rk.get('recall@1',0):.3f} @5 {rk.get('recall@5',0):.3f} "
                      f"@10 {rk.get('recall@10',0):.3f} resid {rk.get('resid_match',0):.3f} drv_hit {rk.get('driver_hit',0):.3f}", flush=True)
                dump = {}                                            # per-fate top-K freq-ranked shortlist (weighted freq) + ≤k best `cocktail` for the SOFT/diversity eval
                for domlabel, fates in [('in-domain(train):policy (amortized)', list(TR)), ('held-out:policy (amortized)', list(HO))]:
                    dd = {}
                    for li in fates:
                        o = cent[li]; sel = _sel_of(o)
                        best_set, ranked = racc(o[sel], sel, li)
                        gs_best = [data.gene_names[int(i)] for i in best_set if 0 <= int(i) < len(data.gene_names)]
                        freq = {}
                        for r, g in enumerate(ranked):               # descending int weights => freq-rank == model-score rank
                            if 0 <= int(g) < len(data.gene_names):
                                nm2 = data.gene_names[int(g)]
                                if nm2 not in freq: freq[nm2] = _K_short - r
                        drv = data.lineage_drivers[names[li]]; drvset = set(drv['train'] + drv['test'])
                        dd[names[li]] = {'freq': freq, 'cocktail': gs_best, 'drivers': [g for g in drvset if g in cand_genes],
                                         'drivers_full': [g for g in drv.get('full', []) if g in cand_genes] or [g for g in drvset if g in cand_genes],
                                         'source': drv.get('source', 'joung')}
                    dump[domlabel] = dd
                jp = f'{EXP20}/results/briggs/eval_only_perfate_tfs_{(a.run_name or "policy").replace("/","_")}_{nm_.replace("-","_")}.json'
                json.dump(dump, open(jp, 'w'), indent=2); print(f"  saved perfate dump -> {jp}", flush=True)
                try:                                                 # HSPC held-out pick for this baseline (PLAIN best set; cap via the HSPC key)
                    hp = core(o_hspc[_sel_of(o_hspc)], _sel_of(o_hspc), None); sc = score_cocktail(hp)
                    if sc: print(f"  [HSPC {nm_}] picks {sc['picks']} | P1 {sc['P1']}/{len(P1)} P2 {sc['P2']}/{len(P2)}", flush=True)
                except Exception as e:
                    print(f"  [HSPC {nm_}] skipped: {e}", flush=True)
        if getattr(a, 'briggs_frame', False):                        # FRAME test: MPC reach from Joung-frame vs Briggs-frame control (does the start frame cost reach?)
            import anndata as _ad, scipy.sparse as _sp
            _PLURI = ['POU5F1', 'NANOG', 'SOX2', 'LIN28A', 'DPPA4', 'ZFP42']
            _BH = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/_decompressed/all_cells_final_v2.h5ad"
            def _lognorm(X):
                X = (X.toarray() if _sp.issparse(X) else np.asarray(X)).astype(np.float32)
                return np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)
            _B = _ad.read_h5ad(_BH); _bg = list(map(str, _B.var_names)); _gn = list(map(str, data.gene_names)); _g2i = {g: i for i, g in enumerate(_gn)}
            _sh = [g for g in _bg if g in _g2i]; _si = np.array([_g2i[g] for g in _sh]); _Xln = _lognorm(_B[:, _sh].X)
            _pmk = [_sh.index(g) for g in _PLURI if g in _sh]; _ps = _Xln[:, _pmk].mean(1); _isc = _ps >= np.quantile(_ps, 0.90)
            _Xb = np.zeros((int(_isc.sum()), len(_gn)), np.float32); _Xb[:, _si] = _Xln[_isc]
            with torch.no_grad():
                _o = op.model.encoder(torch.tensor(_Xb, device=dev)); _epsB = (_o[2] if isinstance(_o, (tuple, list)) else _o).detach()
            eps0_B = _epsB.mean(0); z0_B = op.scm_forward(_epsB, A).mean(0); ctrl_recon_B = op.decode(z0_B.unsqueeze(0))[0]
            print(f"[briggs-frame] {int(_isc.sum())} Briggs pluripotent controls | ||eps0_B - eps0_J|| = {float((eps0_B - data.eps_ctrl_mean).norm()):.2f}", flush=True)
            rows_bf = []
            for li in HO:                                            # MPC reach under each frame, per held-out goal
                src = data.lineage_drivers[names[li]].get('source', 'joung'); o = cent[li]; sel = torch.where(o.abs() > thr)[0]
                if len(sel) == 0: continue
                _, mJ = fate_mpc(op, data, cand, o[sel], sel, drift, ctrl_recon, A, beam=a.beam, max_set=a.kmax)                       # Joung-frame start
                _, mB = fate_mpc(op, data, cand, o[sel], sel, drift, ctrl_recon_B, A, beam=a.beam, max_set=a.kmax, eps0=eps0_B)        # Briggs-frame start
                rows_bf.append((src, mJ, mB))
            _selh = torch.where(o_hspc.abs() > thr)[0]
            _, mJh = fate_mpc(op, data, cand, o_hspc[_selh], _selh, drift, ctrl_recon, A, beam=a.beam, max_set=a.kmax)
            _, mBh = fate_mpc(op, data, cand, o_hspc[_selh], _selh, drift, ctrl_recon_B, A, beam=a.beam, max_set=a.kmax, eps0=eps0_B)
            for grp in ['joung', 'briggs']:
                r = [x for x in rows_bf if x[0] == grp]
                if r: print(f"[briggs-frame {grp:6s}] n={len(r)} MPC reach: Joung-frame {np.mean([x[1] for x in r]):+.3f} | "
                            f"Briggs-frame {np.mean([x[2] for x in r]):+.3f} | Δ(B−J) {np.mean([x[2]-x[1] for x in r]):+.3f}", flush=True)
            print(f"[briggs-frame HSPC  ] MPC reach: Joung-frame {mJh:+.3f} | Briggs-frame {mBh:+.3f} | Δ(B−J) {mBh-mJh:+.3f}", flush=True)
        if getattr(a, 'random_null', 0) > 0:                         # MATCHED-BREADTH random-cocktail null, BY SOURCE, incl. freq-weighted SOFT
            import collections as _co
            R = int(a.random_null); N = a.gen_eval_n or 256; rng_n = np.random.default_rng(0)
            cand_genes = {data.gene_names[int(i)] for i in cand}; cg_list = [int(c) for c in cand]
            eps0 = data.eps_ctrl_mean; S_all = op.shift_for(cand); pos = {int(c): i for i, c in enumerate(cand)}
            _tfs = [t for t in data.de_single]; _dr = drift.detach().cpu().numpy()   # SOFT functional space: drift-removed single-TF Δ, unit-normed, 12 KMeans modules
            _Xr = np.stack([np.asarray(data.de_single[t]['lfc'], np.float32) for t in _tfs]) - _dr
            _Xn = _Xr / (np.linalg.norm(_Xr, axis=1, keepdims=True) + 1e-9); _ti = {t: i for i, t in enumerate(_tfs)}
            try:
                from sklearn.cluster import KMeans; _lab = KMeans(12, n_init=4, random_state=0).fit_predict(_Xn)
            except Exception: _lab = np.zeros(len(_tfs), int)
            try:  # dump the SOFT functional space ONCE so the eval suite can compute SOFT-1/2 for ANY planner (MPC/additive/random/oracle) offline, WM-free
                np.savez(f'{EXP20}/results/briggs/eval_funcspace.npz', tfs=np.array(_tfs), Xn=_Xn.astype('float32'), lab=_lab.astype('int32'))
            except Exception: pass
            # ── SOFT-RECALL (functional near-miss) — background-calibrated τ = q-th pctile of random-TF→driver cosine, so the null floor is low ──
            _A_bg = set()
            for _li in HO:
                _drvb = data.lineage_drivers[names[_li]]; _dsetb = set(_drvb['train'] + _drvb['test']) & cand_genes
                _A_bg |= set([g for g in _drvb.get('full', []) if g in cand_genes] or list(_dsetb))
            def _set_tau(driver_names, q=0.90):
                _dd = sorted({x for x in driver_names if x in _ti})
                if not _dd: return 1.0
                _di = [_ti[x] for x in _dd]; _Sd = _Xn @ _Xn[_di].T
                for _j, _i in enumerate(_di): _Sd[_i, _j] = np.nan   # mask self-similarity (driver→itself=1) so background stays honest
                return float(np.nanpercentile(_Sd, 100.0 * q))
            _TAU = _set_tau(_A_bg)
            def _softrec(freq, A, topk=20, tau=None, mode='thr'):    # driver-centric graded recall: fraction of drivers whose best top-K pick cosine >= τ (near-miss credit); τ→1 => exact drv_recall
                A = [x for x in A if x in _ti]; items = [(t, c) for t, c in freq.items() if t in _ti]
                if not A or not items: return float('nan')
                items = sorted(items, key=lambda tc: -tc[1])[:topk]
                P = _Xn[[_ti[t] for t, _ in items]]; Am = _Xn[[_ti[x] for x in A]]
                cov = (Am @ P.T).max(1); tau = _TAU if tau is None else tau
                return float(np.mean(cov >= tau)) if mode == 'thr' else float(np.mean(cov))
            print(f"[soft-recall] τ(q=0.90 random-TF→driver cos)={_TAU:.3f} over {len(_A_bg)} drivers w/ single-TF Δ", flush=True)
            def _soft(freq, A):                                      # RAW freq-weighted SOFT-1 (nearest-answer cos) + SOFT-2 (same module)
                A = [x for x in A if x in _ti]; items = [(t, c) for t, c in freq.items() if t in _ti]
                if not A or not items: return float('nan'), float('nan')
                Am = _Xn[[_ti[x] for x in A]]; acl = set(int(_lab[_ti[x]]) for x in A)
                w = np.array([c for _, c in items], float); w /= w.sum(); xi = np.array([_ti[t] for t, _ in items])
                near = (_Xn[xi] @ Am.T).max(1); inmod = np.array([1.0 if _lab[i] in acl else 0.0 for i in xi])
                return float((w * near).sum()), float((w * inmod).sum())
            def _mets(cocktails, li):
                o = cent[li]; sel = torch.where(o.abs() > thr)[0]; o_t = o[sel]
                drv = data.lineage_drivers[names[li]]; drvset = set(drv['train'] + drv['test']) & cand_genes
                A_full = [g for g in drv.get('full', []) if g in cand_genes] or list(drvset)
                freq = {}; Sb = []
                for gl in cocktails:
                    if not gl: continue
                    for g in gl: freq[data.gene_names[int(g)]] = freq.get(data.gene_names[int(g)], 0) + 1
                    Sb.append(torch.stack([S_all[pos[int(g)]] for g in gl]).sum(0))
                best = -9.0
                if Sb:
                    z = op.apply_many(eps0, torch.stack(Sb), A); best = float(torch_match(op.decode(z) - ctrl_recon - drift, o_t, sel).max())
                ranked = sorted(freq, key=lambda g: -freq[g]); s1, s2 = _soft(freq, A_full)
                return dict(drv_recall20=len(set(ranked[:20]) & drvset) / max(1, len(drvset)),
                            soft1=s1, soft2=s2, drv_cov=len(set(freq) & drvset) / max(1, len(drvset)), resid_best=best,
                            soft_recall20=_softrec(freq, list(drvset), topk=20))   # τ-graded superset of drv_recall20 (near-miss credit)
            keys_m = ['drv_recall20', 'soft1', 'soft2', 'drv_cov', 'resid_best', 'soft_recall20']
            src_of = lambda li: data.lineage_drivers[names[li]].get('source', 'joung')
            POL = _co.defaultdict(lambda: _co.defaultdict(list)); NUL = _co.defaultdict(lambda: _co.defaultdict(list))
            _Ns = sorted({int(x) for x in a.null_pareto.split(',') if x.strip()}) if getattr(a, 'null_pareto', '') else []
            _Nmax = max([N] + _Ns)                                       # sample N_max once/fate; existing metrics use first N (unchanged)
            pol_by_fate = {}
            _GGn = getattr(env, '_goal_gate', None)                      # RETRIEVER/cosine gate: pool-matched null per goal
            perfate_soft = {}                                            # per-fate policy vs null soft metrics (harvest)
            for li in HO:
                obs = env.reset(keys[li]); s_, t_ = obs[:d], obs[d:]; pol = []
                if _GGn is not None and li in _GGn:                      # gate policy proposals AND draw the null from the SAME per-goal shortlist (fair, pool-matched)
                    _blk = torch.ones(len(cand), dtype=torch.bool, device=dev); _blk[list(_GGn[li])] = False; ag_e.policy._allowed_block = _blk
                    cg_list_li = cg_list if getattr(a, 'null_full_pool', False) else [int(cand[i]) for i in _GGn[li]]   # full-280 null (pipeline vs ungated) or pool-matched (policy vs shortlist)
                else:
                    ag_e.policy._allowed_block = None; cg_list_li = cg_list
                _kc = int(getattr(env, 'kmax_active', None) or _Nmax)    # per-source cap (no-op unless set_kmax installed)
                for _ in range(_Nmax):
                    _, g, _, _ = ag_e.select(s_, t_); pol.append([cand[x] for x in g][:_kc])
                pol_by_fate[li] = pol
                polN = pol[:N]; ks = [max(1, len(c)) for c in polN]; pm = _mets(polN, li); src = src_of(li)
                acc = {k: [] for k in keys_m}
                for _ in range(R):
                    rc = [list(rng_n.choice(cg_list_li, size=min(k, len(cg_list_li)), replace=False)) for k in ks]; m = _mets(rc, li)
                    for k in keys_m:
                        if m[k] == m[k]: acc[k].append(m[k])
                for k in keys_m:
                    if pm[k] == pm[k]:
                        for grp in (src, 'ALL'): POL[grp][k].append(pm[k])
                    if acc[k]:
                        for grp in (src, 'ALL'): NUL[grp][k].append(float(np.mean(acc[k])))
                perfate_soft[names[li]] = dict(source=src, k_med=float(np.median(ks)), n_samples=len(polN),
                    **{f'pol_{k}': (float(pm[k]) if pm[k] == pm[k] else None) for k in keys_m},
                    **{f'null_{k}': (float(np.mean(acc[k])) if acc[k] else None) for k in keys_m},
                    # rl_eval_suite: null STD + quantiles per fate => enables z-score / empirical-percentile
                    **{f'nullstd_{k}': (float(np.std(acc[k])) if acc[k] else None) for k in keys_m},
                    **{f'nullq_{k}': ([float(np.quantile(acc[k], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
                                      if acc[k] else None) for k in keys_m})
            import json as _jsps
            _pfp = f'{EXP20}/results/briggs/eval_only_perfate_soft_{(a.run_name or "policy").replace("/","_")}.json'
            _jsps.dump(perfate_soft, open(_pfp, 'w'), indent=2); print(f"  saved per-fate soft dump ({len(perfate_soft)} fates) -> {_pfp}", flush=True)
            print(f"[random-null] policy/null(Δ) — MATCHED-k random-{N} ({R} perms), BY SOURCE:", flush=True)
            for grp in ['ALL', 'joung', 'joung_combo', 'briggs']:
                if not POL[grp]: continue
                nf = len(POL[grp]['drv_recall20'])
                seg = "  ".join(f"{k}={np.mean(POL[grp][k]):+.3f}/{np.mean(NUL[grp][k]):+.3f}(Δ{np.mean(POL[grp][k])-np.mean(NUL[grp][k]):+.3f})" for k in keys_m)
                print(f"   {grp:6s} n={nf:2d} | {seg}", flush=True)
            print("   [SOFT-1/2 = RAW freq-weighted; Δ vs breadth-null is the honest signal | drv_cov/resid_best breadth if Δ~0]", flush=True)

            # ═══ COMBO-SET recovery (joung_combo + briggs): set-Jaccard(cocktail, answer combo) + FATE-STATE dump (cell/Δ-space UMAP) ═══
            _HOset = set(HO); CJ = _co.defaultdict(list); CN = _co.defaultdict(list); _fatestates = {}
            for li in range(len(names)):
                src = src_of(li); nm_ = names[li]
                acs = [(s & cand_genes) for s in getattr(data, 'answer_combos', {}).get(nm_, []) if len(s & cand_genes) >= 2]
                pol_delta = None; jac = None
                if li in _HOset:
                    pol = pol_by_fate[li][:N]
                    if acs:                                            # SET-Jaccard: best cocktail-vs-answer-combo overlap (WM-independent, set-level) vs matched-breadth null
                        def _mj(cocks):
                            b = 0.0
                            for gl in cocks:
                                gs = set(data.gene_names[int(g)] for g in gl)
                                for am in acs:
                                    u = gs | am; b = max(b, (len(gs & am) / len(u)) if u else 0.0)
                            return b
                        ksj = [max(1, len(c)) for c in pol]; jac = _mj(pol)
                        _cgl = cg_list if getattr(a, 'null_full_pool', False) else ([int(cand[i]) for i in _GGn[li]] if (_GGn is not None and li in _GGn) else cg_list)   # full-280 or pool-matched null under a gate
                        njc = np.mean([_mj([list(rng_n.choice(_cgl, size=min(k, len(_cgl)), replace=False)) for k in ksj]) for _ in range(R)])
                        for grp in (src, 'combo_ALL'): CJ[grp].append(jac); CN[grp].append(float(njc))
                    env.reset(keys[li])                                # policy endpoint = WM-predicted Δ of the modal cocktail (frame-routed control)
                    modal = _co.Counter(frozenset(int(g) for g in c) for c in pol if c).most_common(1)
                    if modal:
                        S = [int(g) for g in modal[0][0]]
                        with torch.no_grad():
                            _zt, _, _ = op.apply(env.eps0_active.unsqueeze(0), S, A)
                            pol_delta = (op.decode(_zt)[0] - env._ctrl_recon_active - drift).detach().cpu().numpy().astype('float32').tolist()
                _cocktails = [sorted(data.gene_names[int(g)] for g in c) for c in pol_by_fate[li][:N]] if li in _HOset else []
                _fatestates[nm_] = dict(source=src, heldout=(li in _HOset),
                    target=(cent[li].detach().cpu().numpy().astype('float32').tolist()),
                    policy=pol_delta, answer_tfs=sorted(set().union(*acs)) if acs else [],
                    answer_combos=[sorted(s) for s in getattr(data, 'answer_combos', {}).get(nm_, [])],
                    cocktails=_cocktails)                              # raw N sampled cocktails (TF names) for diversity / by-k analysis
            if CJ:
                print("[combo-set Jaccard] best cocktail-vs-answer-combo set-Jaccard, policy/null(Δ), by source (WM-independent):", flush=True)
                for grp in ['combo_ALL', 'joung_combo', 'briggs']:
                    if CJ.get(grp): print(f"   {grp:10s} n={len(CJ[grp]):2d} | policy {np.mean(CJ[grp]):.3f} / null {np.mean(CN[grp]):.3f} (Δ{np.mean(CJ[grp])-np.mean(CN[grp]):+.3f})", flush=True)
            _fsp = f'{EXP20}/results/briggs/eval_only_fatestates_{(a.run_name or "policy").replace("/", "_")}.json'
            json.dump({'genes': list(map(str, data.gene_names)), 'fates': _fatestates}, open(_fsp, 'w'))
            print(f"  saved fate-state dump ({len(_fatestates)} fates) -> {_fsp}", flush=True)

            # ═══ UPGRADED EVAL (WM-independent): recall-depth sweep · SOFT-1 discrimination · best-of-N Pareto · SOFT-2 k-sweep ═══
            def _drvset(li):
                drv = data.lineage_drivers[names[li]]; return set(drv['train'] + drv['test']) & cand_genes
            def _Afull(li):
                drv = data.lineage_drivers[names[li]]; return [g for g in drv.get('full', []) if g in cand_genes] or list(_drvset(li))
            def _freqof(cocks):
                fq = {}
                for gl in cocks:
                    for g in gl: fq[data.gene_names[int(g)]] = fq.get(data.gene_names[int(g)], 0) + 1
                return fq
            # (1) RECALL-DEPTH SWEEP — freq-ranked driver recall @ several depths, policy vs matched-breadth null, by source
            depths = [int(x) for x in a.recall_depths.split(',') if x.strip()]
            RP = _co.defaultdict(lambda: _co.defaultdict(list)); RN = _co.defaultdict(lambda: _co.defaultdict(list))
            for li in HO:
                polN = pol_by_fate[li][:N]; ds = _drvset(li)
                if not ds: continue
                fq = _freqof(polN); ranked = sorted(fq, key=lambda g: -fq[g]); src = src_of(li); ks = [max(1, len(c)) for c in polN]
                for dep in depths:
                    pr = len(set(ranked[:dep]) & ds) / len(ds); nr = []
                    for _ in range(R):
                        rc = [list(rng_n.choice(cg_list, size=k, replace=False)) for k in ks]
                        fqr = _freqof(rc); rr = sorted(fqr, key=lambda g: -fqr[g]); nr.append(len(set(rr[:dep]) & ds) / len(ds))
                    for grp in (src, 'ALL'): RP[grp][dep].append(pr); RN[grp][dep].append(float(np.mean(nr)))
            print(f"[recall-depth sweep] freq-ranked driver recall@{depths}, policy/null(Δ), by source (N={N}, {R} perms):", flush=True)
            for grp in ['ALL', 'joung', 'joung_combo', 'briggs']:
                if not RP[grp]: continue
                seg = "  ".join(f"@{dep}={np.mean(RP[grp][dep]):.3f}/{np.mean(RN[grp][dep]):.3f}(Δ{np.mean(RP[grp][dep])-np.mean(RN[grp][dep]):+.3f})" for dep in depths)
                print(f"   {grp:6s} n={len(RP[grp][depths[0]]):2d} | {seg}", flush=True)
            # (2) SOFT-1 DISCRIMINATION — SOFT1(picks vs OWN drivers) − mean SOFT1(vs OTHER held-out fates); + ORACLE ceiling (drivers-as-picks)
            if getattr(a, 'soft_disc', False):
                Afull = {li: _Afull(li) for li in HO}; pfreq = {li: _freqof(pol_by_fate[li][:N]) for li in HO}
                DP = _co.defaultdict(list); DO = _co.defaultdict(list)
                DPr = _co.defaultdict(list); DOr = _co.defaultdict(list)   # soft-RECALL own−other + oracle (near-miss discrimination)
                for li in HO:
                    own = _soft(pfreq[li], Afull[li])[0]; others = [_soft(pfreq[li], Afull[lj])[0] for lj in HO if lj != li and Afull[lj]]
                    others = [x for x in others if x == x]
                    if own != own or not others: continue
                    grp0 = src_of(li); ofreq = {g: 1 for g in Afull[li]}
                    o_own = _soft(ofreq, Afull[li])[0]; o_oth = [_soft(ofreq, Afull[lj])[0] for lj in HO if lj != li and Afull[lj]]
                    o_oth = [x for x in o_oth if x == x]; odisc = (o_own - float(np.mean(o_oth))) if (o_own == o_own and o_oth) else float('nan')
                    for grp in (grp0, 'ALL'): DP[grp].append(own - float(np.mean(others))); DO[grp].append(odisc)
                    ds_own = list(_drvset(li))                        # soft-recall discrimination on the (train+test) answer set
                    r_own = _softrec(pfreq[li], ds_own); r_oth = [_softrec(pfreq[li], list(_drvset(lj))) for lj in HO if lj != li and _drvset(lj)]
                    r_oth = [x for x in r_oth if x == x]
                    if r_own == r_own and r_oth:
                        orf = {g: 1 for g in ds_own}
                        ro_own = _softrec(orf, ds_own); ro_oth = [_softrec(orf, list(_drvset(lj))) for lj in HO if lj != li and _drvset(lj)]
                        ro_oth = [x for x in ro_oth if x == x]; ro_disc = (ro_own - float(np.mean(ro_oth))) if (ro_own == ro_own and ro_oth) else float('nan')
                        for grp in (grp0, 'ALL'): DPr[grp].append(r_own - float(np.mean(r_oth))); DOr[grp].append(ro_disc)
                print("[SOFT-1 DISCRIMINATION] SOFT1(own)−mean SOFT1(other fates); policy vs ORACLE(drivers-as-picks) ceiling, by source:", flush=True)
                for grp in ['ALL', 'joung', 'joung_combo', 'briggs']:
                    if not DP[grp]: continue
                    _od = [x for x in DO[grp] if x == x]; ora = f" | oracle disc={np.mean(_od):+.3f}" if _od else ""
                    print(f"   {grp:6s} n={len(DP[grp]):2d} | policy disc={np.mean(DP[grp]):+.3f}{ora}", flush=True)
                print("[SOFT-RECALL DISCRIMINATION] softrec(own)−mean softrec(other fates); policy vs ORACLE(drivers-as-picks), by source:", flush=True)
                for grp in ['ALL', 'joung', 'joung_combo', 'briggs']:
                    if not DPr[grp]: continue
                    _odr = [x for x in DOr[grp] if x == x]; orar = f" | oracle disc={np.mean(_odr):+.3f}" if _odr else ""
                    print(f"   {grp:6s} n={len(DPr[grp]):2d} | policy disc={np.mean(DPr[grp]):+.3f}{orar}", flush=True)
            # (3) BEST-OF-N PARETO — drv_recall@20 & SOFT-1 vs matched-breadth null across sampling budgets N, by source
            if _Ns:
                print(f"[best-of-N Pareto] N in {_Ns} — drv_recall@20 & SOFT-1 (policy−null Δ), by source ({R} perms):", flush=True)
                for Ncur in _Ns:
                    PP = _co.defaultdict(lambda: _co.defaultdict(list)); PN = _co.defaultdict(lambda: _co.defaultdict(list))
                    for li in HO:
                        sub = pol_by_fate[li][:Ncur]; ds = _drvset(li); A = _Afull(li)
                        fq = _freqof(sub); ranked = sorted(fq, key=lambda g: -fq[g]); src = src_of(li); ks = [max(1, len(c)) for c in sub]
                        p_dr = len(set(ranked[:20]) & ds) / max(1, len(ds)); p_s1 = _soft(fq, A)[0]; nd, ns = [], []
                        for _ in range(R):
                            rc = [list(rng_n.choice(cg_list, size=k, replace=False)) for k in ks]
                            fqr = _freqof(rc); rr = sorted(fqr, key=lambda g: -fqr[g]); nd.append(len(set(rr[:20]) & ds) / max(1, len(ds)))
                            s1r = _soft(fqr, A)[0]
                            if s1r == s1r: ns.append(s1r)
                        for grp in (src, 'ALL'):
                            if ds: PP[grp]['dr'].append(p_dr); PN[grp]['dr'].append(float(np.mean(nd)))
                            if p_s1 == p_s1 and ns: PP[grp]['s1'].append(p_s1); PN[grp]['s1'].append(float(np.mean(ns)))
                    for grp in ['ALL', 'joung', 'joung_combo', 'briggs']:
                        if not PP[grp]: continue
                        dr = (np.mean(PP[grp]['dr']) - np.mean(PN[grp]['dr'])) if PP[grp]['dr'] else float('nan')
                        s1 = (np.mean(PP[grp]['s1']) - np.mean(PN[grp]['s1'])) if PP[grp]['s1'] else float('nan')
                        print(f"   N={Ncur:5d} {grp:6s} | drv_recall@20 Δ{dr:+.3f} | SOFT-1 Δ{s1:+.3f}", flush=True)
            # (4) SOFT-2 CLUSTER-COUNT SENSITIVITY — functional modules at several KMeans k; policy−null Δ, by source
            if getattr(a, 'soft2_ks', ''):
                from sklearn.cluster import KMeans as _KM
                def _s2(freq, A, lab):
                    A = [x for x in A if x in _ti]; items = [(t, c) for t, c in freq.items() if t in _ti]
                    if not A or not items: return float('nan')
                    acl = set(int(lab[_ti[x]]) for x in A); w = np.array([c for _, c in items], float); w /= w.sum()
                    xi = [_ti[t] for t, _ in items]; return float((w * np.array([1.0 if lab[i] in acl else 0.0 for i in xi])).sum())
                for kc in [int(x) for x in a.soft2_ks.split(',') if x.strip()]:
                    try: lab_k = _KM(kc, n_init=4, random_state=0).fit_predict(_Xn)
                    except Exception: continue
                    SP = _co.defaultdict(list)
                    for li in HO:
                        sub = pol_by_fate[li][:N]; A = _Afull(li); fq = _freqof(sub); ps = _s2(fq, A, lab_k); ks = [max(1, len(c)) for c in sub]
                        nn = [_s2(_freqof([list(rng_n.choice(cg_list, size=k, replace=False)) for k in ks]), A, lab_k) for _ in range(R)]
                        nn = [x for x in nn if x == x]
                        if ps == ps and nn:
                            for grp in (src_of(li), 'ALL'): SP[grp].append(ps - float(np.mean(nn)))
                    seg = "  ".join(f"{grp}={np.mean(SP[grp]):+.3f}" for grp in ['ALL', 'joung', 'joung_combo', 'briggs'] if SP[grp])
                    print(f"[SOFT-2 k={kc:2d}] policy−null Δ by source | {seg}", flush=True)

            _ho = env.reset(hspc_key); polh = []                     # HSPC P1/P2 union vs matched-breadth null (the union-metric stress test)
            for _ in range(N):
                _, g, _, _ = ag_e.select(_ho[:d], _ho[d:]); polh.append([cand[x] for x in g])
            ksh = [max(1, len(c)) for c in polh]
            def _p1p2(cocks):
                u = set()
                for gl in cocks: u |= {data.gene_names[int(x)] for x in gl}
                return len(u & P1), len(u & P2)
            pp1, pp2 = _p1p2(polh); n1, n2 = [], []
            for _ in range(R):
                rc = [list(rng_n.choice(cg_list, size=k, replace=False)) for k in ksh]; a1, a2 = _p1p2(rc); n1.append(a1); n2.append(a2)
            print(f"[random-null HSPC] P1 union: policy {pp1}/{len(P1)} vs null {np.mean(n1):.2f}±{np.std(n1):.2f} | "
                  f"P2 union: policy {pp2}/{len(P2)} vs null {np.mean(n2):.2f}±{np.std(n2):.2f}", flush=True)
            A_hspc = [g for g in (P1 | P2) if g in _ti]              # HSPC SOFT: freq-weighted functional proximity of picks to the P1∪P2 answer key (WM-INDEPENDENT), vs breadth null
            def _freq(cocks):
                fq = {}
                for gl in cocks:
                    for x in gl: fq[data.gene_names[int(x)]] = fq.get(data.gene_names[int(x)], 0) + 1
                return fq
            A_id = (P1 | P2) & cand_genes                            # HSPC HARD (exact TF identity): freq-ranked drv_recall@20 over the P1∪P2 answer key, vs null
            def _dr20(cocks):
                fq = _freq(cocks); ranked = sorted(fq, key=lambda g: -fq[g])
                return len(set(ranked[:20]) & A_id) / max(1, len(A_id))
            pdr = _dr20(polh); ndr = [_dr20([list(rng_n.choice(cg_list, size=k, replace=False)) for k in ksh]) for _ in range(R)]
            print(f"[random-null HSPC HARD] drv_recall@20 (P1∪P2, freq-ranked): policy {pdr:.3f} vs null "
                  f"{np.mean(ndr):.3f}±{np.std(ndr):.3f} (Δ{pdr-np.mean(ndr):+.3f}) | |A_id|={len(A_id)}/12", flush=True)
            if len(A_hspc) >= 2:
                ph1, ph2 = _soft(_freq(polh), A_hspc); ns1, ns2 = [], []
                for _ in range(R):
                    rc = [list(rng_n.choice(cg_list, size=k, replace=False)) for k in ksh]; s1r, s2r = _soft(_freq(rc), A_hspc)
                    if s1r == s1r: ns1.append(s1r); ns2.append(s2r)
                print(f"[random-null HSPC SOFT] |A|={len(A_hspc)}/{len(P1|P2)} answer-key TFs w/ Joung single | "
                      f"SOFT-1 policy {ph1:+.3f} vs null {np.mean(ns1):+.3f} (Δ{ph1-np.mean(ns1):+.3f}) | "
                      f"SOFT-2 policy {ph2:+.3f} vs null {np.mean(ns2):+.3f} (Δ{ph2-np.mean(ns2):+.3f})", flush=True)
            else:
                print(f"[random-null HSPC SOFT] skipped: only {len(A_hspc)} P1/P2 TFs have a Joung single-TF Δ", flush=True)
            try:                                                     # ARRAYED answer key: the 26 validated HSPC cocktails (54 TFs) — a MUCH broader key than P1/P2 (only 11 of 54 in P1∪P2)
                _arr_df = pd.read_excel('/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/Briggs_Supplementary/mmc5.xlsx', 'Arrayed_screens')
                ARR = [frozenset(str(t).split('|')[0].strip() for t in _sub['TF_names']) for _cid, _sub in _arr_df.groupby('Combination_ID')]
                ARR = [s for s in ARR if len(s) >= 2]
            except Exception as e:
                ARR = []; print(f"[random-null HSPC ARRAYED] could not load arrayed combos: {e}", flush=True)
            if ARR:
                A_arr = [g for g in (set().union(*ARR) & cand_genes) if g in _ti]        # arrayed answer-key TFs (in pool + w/ Joung single) — SOFT target
                ARR_mod = [set(s) & cand_genes for s in ARR if len(set(s) & cand_genes) >= 2]   # arrayed combos restricted to modelable TFs — set-recovery target
                if len(A_arr) >= 2:
                    aph1, aph2 = _soft(_freq(polh), A_arr); as1, as2 = [], []
                    for _ in range(R):
                        rc = [list(rng_n.choice(cg_list, size=k, replace=False)) for k in ksh]; s1a, s2a = _soft(_freq(rc), A_arr)
                        if s1a == s1a: as1.append(s1a); as2.append(s2a)
                    print(f"[random-null HSPC ARRAYED-SOFT] |A|={len(A_arr)} arrayed-answer TFs w/ Joung single | "
                          f"SOFT-1 policy {aph1:+.3f} vs null {np.mean(as1):+.3f} (Δ{aph1-np.mean(as1):+.3f}) | "
                          f"SOFT-2 policy {aph2:+.3f} vs null {np.mean(as2):+.3f} (Δ{aph2-np.mean(as2):+.3f})", flush=True)
                def _maxjac(cocks):                                  # best set-Jaccard of any sampled cocktail vs any modelable arrayed combo
                    b = 0.0
                    for gl in cocks:
                        gs = set(data.gene_names[int(x)] for x in gl)
                        for am in ARR_mod:
                            u = gs | am; b = max(b, (len(gs & am) / len(u)) if u else 0.0)
                    return b
                pj = _maxjac(polh); njs = [_maxjac([list(rng_n.choice(cg_list, size=k, replace=False)) for k in ksh]) for _ in range(R)]
                print(f"[random-null HSPC ARRAYED-Jaccard] {len(ARR_mod)} modelable combos | best set-Jaccard "
                      f"policy {pj:.3f} vs null {np.mean(njs):.3f}±{np.std(njs):.3f} (Δ{pj-np.mean(njs):+.3f})", flush=True)
        try:                                                         # HSPC best-of-N (same block as the training path) — apples-to-apples P1/P2 union for any saved policy
            _ho = env.reset(hspc_key); _p1u, _p2u, _best = set(), set(), None
            for _ in range(a.gen_eval_n):
                _t, _g, _, _ = ag_e.select(_ho[:d], _ho[d:]); _gi = [cand[x] for x in _g]
                _sc = score_cocktail(_gi)
                if _sc:
                    _gs = set(_sc['picks']); _p1u |= (_gs & P1); _p2u |= (_gs & P2)
                    if _best is None or _sc.get('fate_marker', -9) > _best.get('fate_marker', -9): _best = _sc
            if _best is not None:
                print(f"[HSPC best-of-N] over N={a.gen_eval_n}: P1_union={len(_p1u)}/{len(P1)} P2_union={len(_p2u)}/{len(P2)} "
                      f"(recovered: {sorted(_p1u|_p2u)}) | best-fate_marker cocktail {_best['picks']}", flush=True)
        except Exception as e:
            print(f"[HSPC best-of-N] skipped: {e}", flush=True)
        if perfate_tf_dump:                                          # per-fate SAMPLED TF frequencies + experimental answer-key drivers (for the TF-recovery hard/soft evals)
            pj = f'{EXP20}/results/briggs/eval_only_perfate_tfs_{(a.run_name or "policy").replace("/", "_")}.json'
            json.dump(perfate_tf_dump, open(pj, 'w'), indent=2); print(f"saved -> {pj}", flush=True)
        if wb is not None: wb_finish(wb)
        return

    ROWS = {}
    # (1) MPC — per-goal planner reference (searches the held-out goal directly)
    ROWS['MPC (beam,k2)'] = evalp(mpc_p(a.beam, 2))
    sel_h = torch.where(o_hspc.abs() > thr)[0]
    chosen_h, _ = fate_mpc(op, data, cand, o_hspc[sel_h], sel_h, drift, ctrl_recon, A, beam=a.beam, max_set=a.kmax)
    HROWS['MPC-HSPC (ceiling)'] = score_cocktail(chosen_h)                     # direct search on the HSPC goal
    print(f"[done] MPC", flush=True)

    # (2) MPC-distill (BC) — SKIPPED when bc_epochs<=0 (no behavior cloning)
    if a.bc_epochs > 0:
      try:
        ag_d = SetPPO(state_dim=d, n_candidates=len(cand), k_max=a.kmax, device=dev)
        bc_init(ag_d, a.bc_epochs); dp = planner_set(ag_d, op, data, env)
        ROWS['MPC-distill (zero-shot)'] = evalp(lambda li: dp(keys[li]))
        gen_eval(ag_d, 'mpc-distill', N=a.gen_eval_n)   # best-of-N gen_eval for the MPC-distill BASELINE (resid_best/drv_recall@20/specificity)
        HROWS['MPC-distill (zero-shot)'] = score_cocktail(dp(hspc_key))
        print(f"[done] MPC-distill", flush=True)
      except Exception as e:
        print(f"[FAIL] MPC-distill: {e}", flush=True)
    else:
        print("[skip] MPC-distill track (bc_epochs=0 => no behavior cloning)", flush=True)
    env.reset(keys[TR[0]])   # init env.z_target/eps0_active for track 3 (was a side-effect of the now-guarded MPC-distill eval)

    # (3) set-PPO from scratch (multi-goal) — eval zero-shot
    try:
        ag_s = SetPPO(state_dim=d, n_candidates=len(cand), k_max=a.kmax, entropy_coeff=a.ent_coeff, device=dev, cvar_tau=a.cvar_tau, cvar_pergoal=a.cvar_pergoal, rloo=a.rloo, cvar_anneal=a.cvar_anneal, cvar_loo=a.cvar_loo, adv_norm_pergoal=a.adv_norm_pergoal, **_gh)
        if a.hstep or a.gflownet:                                            # #1 H-step MDP / GFlowNet-TB (+ optional #2 causal gate) — both reuse the unified manifold penalty
            _uni_maha = None                                                 # unified manifold penalty: min-dist to Joung (maha_fn) OR Briggs (FR) manifold
            if a.beta_maha > 0 and (maha_fn is not None or FR.sup_L is not None):
                def _uni_maha(z):
                    vs = []
                    if maha_fn is not None: vs.append(maha_fn(z))            # Joung-perturbation manifold
                    if FR.sup_L is not None: vs.append(FR.mahalanobis(z))    # observed-Briggs-combo manifold
                    return torch.stack(vs, 0).min(0).values if vs else torch.zeros(z.shape[0], device=z.device)
                print(f"[maha-both] unified penalty = min(Joung {'on' if maha_fn is not None else 'off'}, Briggs {'on' if FR.sup_L is not None else 'off'}) | beta={a.beta_maha}", flush=True)
            # (frame routing is installed earlier, right after env creation, so it applies to both eval_only and training)
            _kw = dict(op=op, data=data, env=env, cand=cand, keys=keys, TR=TR, names=names, cent=cent, drift=drift,
                       ctrl_recon=ctrl_recon, A=A, dev=dev, d=d, a=a, wb=wb, wb_log=wb_log, model=model, EXP20=EXP20, maha_fn=_uni_maha)
            if a.gflownet:
                from analysis.gflownet_hstep import train_gflownet
                train_gflownet(ag_s, a.iters, 'setppo', **_kw)
            else:
                from analysis.hstep_mdp import train_hstep
                train_hstep(ag_s, a.iters, 'setppo', **_kw)
        else:
            train_ppo(ag_s, a.iters, 'setppo')
        sp = planner_set(ag_s, op, data, env)
        ROWS['set-PPO (zero-shot)'] = evalp(lambda li: sp(keys[li]))
        HROWS['set-PPO (zero-shot)'] = score_cocktail(sp(hspc_key))
        # BEST-OF-N HSPC: the greedy argmax games ZNF680, but the SAMPLED distribution reaches the answer key.
        # Sample the policy N times on the HSPC goal; report the best cocktail (by fate_marker) + the P1/P2 UNION over N.
        try:
            _ho = env.reset(hspc_key); _p1u, _p2u, _best = set(), set(), None
            for _ in range(a.gen_eval_n):
                _t, _g, _, _ = ag_s.select(_ho[:d], _ho[d:]); _gi = [cand[x] for x in _g]
                _sc = score_cocktail(_gi)
                if _sc:
                    _gs = set(_sc['picks']); _p1u |= (_gs & P1); _p2u |= (_gs & P2)
                    if _best is None or _sc.get('fate_marker', -9) > _best.get('fate_marker', -9): _best = _sc
            if _best is not None:
                HROWS['set-PPO (best-of-N)'] = {**_best, 'P1': len(_p1u), 'P2': len(_p2u)}   # P1/P2 = UNION over N (distribution coverage)
                print(f"[HSPC best-of-N] over N={a.gen_eval_n}: P1_union={len(_p1u)}/{len(P1)} P2_union={len(_p2u)}/{len(P2)} "
                      f"(recovered: {sorted(_p1u|_p2u)}) | best-fate_marker cocktail {_best['picks']}", flush=True)
        except Exception as e:
            print(f"[HSPC best-of-N] skipped: {e}", flush=True)
        gen_eval(ag_s, 'set-PPO', N=a.gen_eval_n)
        if a.save_scratch:                                                       # save the FROM-SCRATCH set-PPO (ag_s) — the diverse-explorer sibling of the BC-init ag_b
            torch.save(ag_s.policy.state_dict(), a.save_scratch)
            print(f"  saved FROM-SCRATCH set-PPO policy artifact -> {a.save_scratch}", flush=True)
        gen_eval(ag_s, 'set-PPO IN-DOMAIN(train)', fates=TR[:max(len(HO), 8)])   # resid_best on TRAIN fates: did it learn even in-domain, or ignore the goal everywhere?
        diagnostics(lambda li: sp(keys[li]), ag_s, 'set-PPO')       # DEBUG: goal-load Jaccard + held-out picks for set-PPO
        print(f"[done] set-PPO", flush=True)
    except Exception as e:
        print(f"[FAIL] set-PPO: {e}", flush=True)

    # (4) BC-init + PPO ('bcppo' track) — SKIPPED when bc_epochs<=0 (no behavior cloning)
    if a.bc_epochs > 0:
      try:
        ag_b = SetPPO(state_dim=d, n_candidates=len(cand), k_max=a.kmax, entropy_coeff=a.ent_coeff,
                      clip_epsilon=a.clip, lr_actor=a.ft_lr, n_epochs=a.ft_epochs,
                      target_kl=a.target_kl, warmup_iters=a.warmup, kl_bc_coeff=a.kl_bc, device=dev,
                      cvar_tau=a.cvar_tau, cvar_pergoal=a.cvar_pergoal, rloo=a.rloo, cvar_anneal=a.cvar_anneal, cvar_loo=a.cvar_loo, adv_norm_pergoal=a.adv_norm_pergoal)   # BC-init = warm start for RLOO/GRPO (per-group baseline needs a non-uniform policy)
        bc_init(ag_b, a.bc_epochs)
        ag_b.bc_policy = copy.deepcopy(ag_b.policy)                          # (1) frozen BC snapshot for the KL anchor
        for p in ag_b.bc_policy.parameters():
            p.requires_grad_(False)
        ag_b.bc_policy.eval()
        bp = planner_set(ag_b, op, data, env); ROWS['BC-init (pre-PPO)'] = evalp(lambda li: bp(keys[li]))
        HROWS['BC-init (pre-PPO)'] = score_cocktail(bp(hspc_key))
        print(f"[BC-init pre-PPO] {ROWS['BC-init (pre-PPO)']}", flush=True)
        gen_eval(ag_b, 'bc-init-pre', N=a.gen_eval_n)   # BC-init BEFORE PPO: baseline to test whether PPO improves over BC or is inert
        train_ppo(ag_b, a.iters, 'bcppo'); bp2 = planner_set(ag_b, op, data, env)
        ROWS['BC-init+PPO (4-fix, zero-shot)'] = evalp(lambda li: bp2(keys[li]))
        HROWS['BC-init+PPO (4-fix, zero-shot)'] = score_cocktail(bp2(hspc_key))
        diagnostics(lambda li: bp2(keys[li]), ag_b, 'bcppo')            # multi-goal health panel
        gen_eval(ag_b, 'bcppo', N=a.gen_eval_n)                                          # generative distribution-aware eval
        amortization_report(ag_b, 'bcppo')                              # Goal-1: atlas-wide policy vs per-goal MPC + timing
        print(f"[done] BC-init+PPO (4-fix)", flush=True)
      except Exception as e:
        print(f"[FAIL] BC-init+PPO: {e}", flush=True)
    else:
        print("[skip] BC-init+PPO/bcppo track (bc_epochs=0 => no behavior cloning)", flush=True)

    # baselines
    ROWS['random (k2)'] = evalp(lambda li: [int(x) for x in rng.choice(cand, 2, replace=False)], n_rep=200)
    def _oracle1(li):                                                       # 1-driver oracle; HSPC has empty 'train' (held-out) -> fall back to 'test'/skip
        drv = data.lineage_drivers[names[li]]['train'] or data.lineage_drivers[names[li]]['test']
        return [g2i[drv[0]]] if drv and drv[0] in g2i else []
    ROWS['1-driver oracle'] = evalp(_oracle1)
    ROWS['chance'] = dict(recall1=1 / K, recall3=3 / K, resid=0.0, drv_hit=float('nan'),
                          held=float('nan'), P1=float('nan'), P2=float('nan'), drv=float('nan'))

    df = pd.DataFrame({k: {c: v.get(c) for c in ['recall1', 'recall3', 'recall5', 'recall10', 'resid', 'drv_hit', 'held', 'P1', 'P2', 'drv']}
                       for k, v in ROWS.items() if v}).T
    df.columns = ['recall@1', 'recall@3', 'recall@5', 'recall@10', 'resid_match', 'driver_hit', 'held_driver', 'P1', 'P2', 'drivers']
    print(f"\n{'='*100}\nMULTI-GOAL ZERO-SHOT FATE RL — train {len(TR)} fates, eval held-out {[names[i] for i in HO]}", flush=True)
    print(df.round(3).to_string(), flush=True)
    for k, v in ROWS.items():
        if v and v.get('picks'):
            print(f"  {k:32s} picks: {v['picks']}", flush=True)
    _sfx = (a.run_name or a.heldout_substr).replace('/', '_')
    out = f'{EXP20}/results/briggs/multigoal_ladder_{_sfx}.csv'
    df.round(4).to_csv(out); print(f"\nsaved -> {out}", flush=True)

    # ---- DEBUG: per-held-out-fate picks + resid for every planner (for the continuous-metric analysis) ----
    try:
        perfate = {'heldout': [names[i] for i in HO],
                   'drivers': {names[i]: {'train': data.lineage_drivers[names[i]]['train'],
                                          'test': data.lineage_drivers[names[i]]['test']} for i in HO}}
        _pl = {'MPC (beam,k2)': mpc_p(a.beam, 2)}
        if 'dp' in locals():  _pl['MPC-distill (zero-shot)'] = lambda li: dp(keys[li])
        if 'sp' in locals():  _pl['set-PPO (zero-shot)'] = lambda li: sp(keys[li])
        if 'bp2' in locals(): _pl['BC-init+PPO (4-fix)'] = lambda li: bp2(keys[li])
        perfate['picks'], perfate['resid'] = {}, {}
        for pname, pidx in _pl.items():
            perfate['picks'][pname], perfate['resid'][pname] = {}, {}
            for li in HO:
                r = eval_goal(pidx, li)
                perfate['picks'][pname][names[li]] = r['picks'] if r else []
                perfate['resid'][pname][names[li]] = (r['resid'] if r else None)
        pj = f'{EXP20}/results/briggs/multigoal_perfate_{_sfx}.json'
        json.dump(perfate, open(pj, 'w'), indent=2); print(f"saved -> {pj}", flush=True)
    except Exception as e:
        print(f"[perfate dump FAILED] {e}", flush=True)

    # ============ claim 2 output: ZERO-SHOT HSPC recovery (Briggs, cross-dataset) ============
    rand_scores = [score_cocktail([int(x) for x in rng.choice(cand, 2, replace=False)]) for _ in range(200)]
    HROWS['random (k2)'] = {kk: float(np.mean([s[kk] for s in rand_scores])) for kk in ['fate', 'maha', 'P1', 'P2', 'drv']}
    HROWS['GATA2+ETV2 oracle'] = score_cocktail([g2i[t] for t in ('GATA2', 'ETV2') if t in g2i])
    for nm, tfs in LIT_COCKTAILS.items():
        present = [t for t in tfs if t in g2i]
        HROWS[f'lit: {nm}'] = score_cocktail([g2i[t] for t in present])
        if len(present) < len(tfs):
            print(f"  [lit] {nm}: {len(tfs)-len(present)} TFs absent from pool ({set(tfs)-set(present)})", flush=True)
    hdf = pd.DataFrame({k: {c: v.get(c) for c in ['fate', 'fate_marker', 'maha', 'P1', 'P2', 'drv', 'k']}
                        for k, v in HROWS.items() if v}).T.sort_values('fate_marker', ascending=False)
    print(f"\n{'='*100}\nZERO-SHOT HSPC RECOVERY (claim 2) — policy trained on Joung fates ONLY, conditioned on the "
          f"Briggs HSPC goal at eval.\nfate = Briggs-validated readout score (higher=more HSPC); P1/P2/drv = TF overlap "
          f"with Briggs answer key. Beat the lit cocktails?", flush=True)
    print(hdf.round(4).to_string(), flush=True)
    for k, v in HROWS.items():
        if v and v.get('picks'):
            print(f"  {k:28s} picks: {v['picks']}", flush=True)
    hout = f'{EXP20}/results/briggs/multigoal_hspc_{_sfx}.csv'
    hdf.round(4).to_csv(hout); print(f"\nsaved -> {hout}", flush=True)

    if wb is not None:
        wb_summary(wb, {f'{m}/{c}': float(df.loc[m, c]) for m in df.index for c in df.columns
                        if pd.notna(df.loc[m, c])})
        wb_summary(wb, {f'hspc/{m}/{c}': float(hdf.loc[m, c]) for m in hdf.index for c in hdf.columns
                        if pd.notna(hdf.loc[m, c])})
        wb_finish(wb)


if __name__ == '__main__':
    main()
