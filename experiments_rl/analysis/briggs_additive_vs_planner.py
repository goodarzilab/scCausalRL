"""Non-additive composition vs the ADDITIVE baseline for HSPC driver recovery.

FRAMING (per task): the ridge/nnls fit over single-TF Joung signatures is NOT an oracle -- it is
definitionally an ADDITIVE composition model (find w s.t. Sum_i w_i * Delta_i ~= target). Its failure
is the ceiling for additive methods, which the world-model planner claims to beat. We rank TFs by both
methods against the same goal + metric, across multiple target definitions (TEST 2 control).

STAGE 1 (this file, --stage additive): build Cao goals + targets, the additive_baseline ranking, S1
coverage. No training -> fast. STAGE 2 (--stage full) adds the trained planner ranking + full table.

additive_baseline: D = [de_single[t] for t in cand_tfs]  (G x C, single-TF observed Delta in WM gene
space); solve ridge  w = (D^T D + lam I)^-1 D^T y   and nnls; rank TFs by |w|. Both D and target y are
restricted to the goal's DEG mask (marker-relaxed = restrict mask to the HSPC identity genes P1/P2 hit).
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_stack import add_stack_args, build_stack

EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
H17 = "/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/analysis/human_dev_annotation"
BRIGGS_SUP = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/Briggs_Supplementary"

# ---- Briggs answer keys (validated against mmc4 / mmc5 Arrayed_screens) ----
P1 = ['TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2']
P2 = ['DDIT3', 'MZF1', 'HOXB3', 'MIER1', 'SIRT6', 'ZNF85']
LANGE = ['ETV2', 'GATA2', 'LMO2', 'TAL1']          # Lange 2020, contains both epistasis pairs
CANON = ['HHEX', 'GFI1B', 'KLF1', 'MEIS1', 'RUNX3']  # canonical-HSC control set (both methods should retain)
EPISTATIC = ['TAL1', 'LMO2', 'ETV2', 'GATA2']       # per-TF Delta-rank targets
# marker-relaxed target genes (HSPC identity that P1/P2 hit); exclude HLF/MECOM/SPINK2
MARKER_KEEP = ['PTPRC', 'CD34', 'THY1', 'ITGA6', 'PROCR', 'KIT', 'RUNX1', 'MLLT3', 'HOPX', 'MEIS1', 'GATA2', 'SOCS2']
MARKER_EXCL = ['HLF', 'MECOM', 'SPINK2']


def rank_of(ranked_tfs, tf):
    return (ranked_tfs.index(tf) + 1) if tf in ranked_tfs else None


def recall_at_k(ranked_tfs, key, k):
    key_in = [t for t in key if t in ranked_tfs]                 # only TFs that COULD be proposed
    if not key_in:
        return float('nan')
    return len(set(ranked_tfs[:k]) & set(key_in)) / len(key_in)


def build_cao_target(cao_df, gene_names, celltype):
    """target DE for a Cao celltype in WM gene space [G] = centroid - mean-over-celltypes (drift)."""
    shared = [g for g in gene_names if g in cao_df.columns]
    idx = [gene_names.index(g) for g in shared]
    C = cao_df[shared].values.astype(np.float32)                 # [n_celltypes, n_shared]
    drift = C.mean(0)
    row = cao_df.loc[celltype, shared].values.astype(np.float32)
    tgt = np.zeros(len(gene_names), np.float32)
    tgt[idx] = row - drift                                       # drift-removed, WM gene space
    return tgt, np.array(idx)


def additive_fit(D, y, sel, lam=1.0):
    """D [G x C] single-TF signatures, y [G] target, sel = mask gene indices. Returns ridge |w|, nnls w."""
    Dm = D[sel]; ym = y[sel]                                     # restrict to the DEG mask
    Dm = Dm - Dm.mean(0, keepdims=True)                          # deviation signatures (match human_annotate)
    DtD = Dm.T @ Dm
    w_ridge = np.linalg.solve(DtD + lam * np.eye(Dm.shape[1]), Dm.T @ ym)
    w_nnls, _ = nnls(np.clip(Dm, 0, None), np.clip(ym, 0, None))
    cos = (Dm.T @ ym) / (np.linalg.norm(Dm, axis=0) * np.linalg.norm(ym) + 1e-9)  # single-sig cosine
    return np.abs(w_ridge), w_nnls, cos


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--stage', default='additive', choices=['additive', 'full'])
    ap.add_argument('--device', default='cpu'); ap.add_argument('--pool', default='curated')
    ap.add_argument('--lam', type=float, default=1.0)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, meta, op, data = build_stack(a, dev, n_dag=8); data.set_pool(a.pool)
    gene_names = list(map(str, data.gene_names)); g2i = data.gene_to_idx

    # ---- candidate TF set = Briggs MORF library ∩ WM-genes ∩ Joung single signatures (shared by BOTH methods) ----
    # Pool 'TF' entries are ISOFORM-level (SYMBOL|NM_xxxx); the WM is gene-level -> collapse to symbol.
    iso = pd.concat([pd.read_excel(f'{BRIGGS_SUP}/mmc5.xlsx', s)['TF'] for s in ['Pool A', 'Pool B', 'Pool C', 'Pool D']]).astype(str)
    briggs_morf = set(iso.str.split('|').str[0].str.strip())
    print(f"[library] {len(iso)} MORF isoforms -> {len(briggs_morf)} unique gene symbols (WM is gene-level)", flush=True)
    de_tfs = set(data.de_single)
    cand_tfs = sorted(briggs_morf & de_tfs)
    # ---- S1: action-space coverage ----
    in_wm = sorted(briggs_morf & set(gene_names))
    missing_wm = sorted(briggs_morf - set(gene_names))
    missing_sig = sorted((briggs_morf & set(gene_names)) - de_tfs)
    print(f"\n===== S1 · ACTION-SPACE COVERAGE =====", flush=True)
    print(f"Briggs MORF library: {len(briggs_morf)} TFs | in WM gene space: {len(in_wm)} | "
          f"with Joung single-signature (proposable+scorable): {len(cand_tfs)}", flush=True)
    print(f"  {len(missing_wm)} MORFs NOT in WM 4366-gene space (unmodelable in principle): "
          f"{missing_wm[:15]}{'...' if len(missing_wm) > 15 else ''}", flush=True)
    print(f"  {len(missing_sig)} MORFs in WM space but no Joung single-signature: {missing_sig[:15]}", flush=True)
    for nm, key in [('P1', P1), ('P2', P2), ('Lange', LANGE), ('CANON', CANON)]:
        miss = [t for t in key if t not in cand_tfs]
        print(f"  answer-key {nm}: {len([t for t in key if t in cand_tfs])}/{len(key)} proposable"
              f"{' | MISSING ' + str(miss) if miss else ''}", flush=True)

    D = np.stack([np.asarray(data.de_single[t]['lfc'], np.float32) for t in cand_tfs], axis=1)  # [G, C]

    # ---- targets (TEST 2 control): Cao-HSC, marker-relaxed; HFLA unavailable -> flagged ----
    cao = pd.read_csv(f'{H17}/cao_type_mean_logexpr.csv', index_col=0)
    cao.columns = cao.columns.astype(str)
    HSC = 'Liver-Hematopoietic stem cells'
    tgt_hsc, hsc_sel = build_cao_target(cao, gene_names, HSC)
    thr = data.thresh
    targets = {}
    # (1) Cao-HSC: DEG mask = |target|>thresh
    sel_full = np.where(np.abs(tgt_hsc) > thr)[0]
    targets['Cao-HSC'] = (tgt_hsc, sel_full)
    # (2) marker-relaxed: same target, mask restricted to present KEEP markers, excluding EXCL
    keep_idx = [g2i[g] for g in MARKER_KEEP if g in g2i and g not in MARKER_EXCL]
    targets['marker-relaxed'] = (tgt_hsc, np.array(sorted(keep_idx)))
    print(f"\n[targets] Cao-HSC DEGs={len(sel_full)} | marker-relaxed mask={len(keep_idx)} genes "
          f"({[g for g in MARKER_KEEP if g in g2i and g not in MARKER_EXCL]})", flush=True)
    print(f"[targets] HFLA HSC/MPP (TEST 2a): NOT STAGED on disk -> reported as unavailable", flush=True)

    # ---- additive_baseline ranking per target ----
    rows = []
    for tname, (y, sel) in targets.items():
        wabs, wnnls, cos = additive_fit(D, y, sel, lam=a.lam)
        order = np.argsort(-wabs)
        ranked = [cand_tfs[i] for i in order]                    # ridge |w| ranking = additive_baseline
        row = {'method': 'additive_baseline', 'target': tname}
        for t in EPISTATIC:
            row[f'rank_{t}'] = rank_of(ranked, t)
        for nm, key in [('P1', P1), ('P2', P2), ('Lange', LANGE)]:
            row[f'recall20_{nm}'] = round(recall_at_k(ranked, key, 20), 3)
        row['canon_top50'] = f"{len([t for t in CANON if t in ranked[:50]])}/{len([t for t in CANON if t in cand_tfs])}"
        row['top10'] = ranked[:10]
        rows.append(row)
        # also recall@k sweep to file
        for k in (10, 20, 50, 100):
            for nm, key in [('P1', P1), ('P2', P2), ('Lange', LANGE)]:
                pass  # summarized in table; full sweep printed below
        print(f"\n--- additive_baseline · {tname} ---", flush=True)
        print(f"  top-15: {ranked[:15]}", flush=True)
        print(f"  ranks: " + " ".join(f"{t}={rank_of(ranked, t)}" for t in EPISTATIC), flush=True)
        print(f"  recall@k P1/P2/Lange: " + " | ".join(
            f"k{k}: {recall_at_k(ranked, P1, k):.2f}/{recall_at_k(ranked, P2, k):.2f}/{recall_at_k(ranked, LANGE, k):.2f}"
            for k in (10, 20, 50, 100)), flush=True)
        print(f"  canonical-HSC in top-50: {[t for t in CANON if t in ranked[:50]]}", flush=True)

    # ================= STAGE 2: WM-SEARCH PLANNER (non-additivity ceiling) =================
    # Score every single + every pair over the SAME 118-TF pool via the world-model do-op
    # (decode(do(eps0, set)) - ctrl_recon), matched to the target on its DEG mask (Pearson).
    # Rank TFs by selection frequency in the top-K cocktails. The do-op's interaction head makes
    # pairs NON-additive -> if TAL1/LMO2/ETV2 rank up vs additive, non-additive composition helps.
    if a.stage == 'full':
        import itertools
        eps0 = data.eps_ctrl_mean.to(dev)
        A = op.mean_A()
        ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]
        cand_idx = [g2i[t] for t in cand_tfs]                       # WM gene indices for the 118 TFs
        singles = [[i] for i in cand_idx]
        pairs = [[i, j] for i, j in itertools.combinations(cand_idx, 2)]
        allsets = singles + pairs
        print(f"\n===== STAGE 2 · WM-SEARCH ({len(singles)} singles + {len(pairs)} pairs = {len(allsets)} cocktails) =====", flush=True)

        @torch.no_grad()
        def wm_delta(sets):                                          # [K, G] decoded expression change
            out = []
            for s in range(0, len(sets), 512):
                z = op.apply_sets(eps0, sets[s:s + 512], A)          # [b, d]
                out.append((op.decode(z) - ctrl_recon).cpu())
            return torch.cat(out, 0)

        DELTA = wm_delta(allsets)                                    # [n_sets, G]

        planner_rows = []
        for tname, (y, sel) in targets.items():
            yt = torch.tensor(y, dtype=torch.float32)
            sel_t = torch.tensor(sel, dtype=torch.long)
            from core.scoring import torch_match
            match = torch_match(DELTA, yt[sel_t], sel_t).numpy()     # [n_sets] Pearson on the mask
            order = np.argsort(-match)
            topK = order[:100]                                       # top-100 cocktails
            freq = {t: 0 for t in cand_tfs}
            for oi in topK:
                for gi in allsets[oi]:
                    freq[gene_names[gi]] += 1
            # rank TFs by selection frequency (tie-break: best single-set match containing the TF)
            best = {t: -9.0 for t in cand_tfs}
            for si, st in enumerate(allsets):
                for gi in st:
                    g = gene_names[gi]
                    if match[si] > best[g]:
                        best[g] = float(match[si])
            ranked = sorted(cand_tfs, key=lambda t: (-freq[t], -best[t]))
            row = {'method': 'WM-search', 'target': tname}
            for t in EPISTATIC:
                row[f'rank_{t}'] = rank_of(ranked, t)
            for nm, key in [('P1', P1), ('P2', P2), ('Lange', LANGE)]:
                row[f'recall20_{nm}'] = round(recall_at_k(ranked, key, 20), 3)
            row['canon_top50'] = f"{len([t for t in CANON if t in ranked[:50]])}/{len([t for t in CANON if t in cand_tfs])}"
            row['top10'] = ranked[:10]
            planner_rows.append(row)
            print(f"\n--- WM-search · {tname} ---", flush=True)
            print(f"  top-15 (by freq in top-100 cocktails): {ranked[:15]}", flush=True)
            print(f"  ranks: " + " ".join(f"{t}={rank_of(ranked, t)}(freq {freq[t]})" for t in EPISTATIC), flush=True)
            print(f"  recall@k P1/P2/Lange: " + " | ".join(
                f"k{k}: {recall_at_k(ranked, P1, k):.2f}/{recall_at_k(ranked, P2, k):.2f}/{recall_at_k(ranked, LANGE, k):.2f}"
                for k in (10, 20, 50, 100)), flush=True)
            print(f"  canonical-HSC in top-50: {[t for t in CANON if t in ranked[:50]]}", flush=True)
            # best single vs best pair for each epistatic TF (direct non-additivity readout)
            for t in EPISTATIC:
                ti = g2i[t]
                s_solo = match[allsets.index([ti])] if [ti] in allsets else float('nan')
                pair_matches = [(match[si], [gene_names[g] for g in st if g != ti][0])
                                for si, st in enumerate(allsets) if len(st) == 2 and ti in st]
                bp = max(pair_matches) if pair_matches else (float('nan'), '-')
                print(f"    {t}: solo match={s_solo:+.3f} | best pair +{bp[1]} match={bp[0]:+.3f} "
                      f"(synergy {bp[0]-s_solo:+.3f})", flush=True)
        rows = rows + planner_rows

    df = pd.DataFrame(rows)
    out = f'{EXP20}/results/briggs/additive_vs_planner.csv'
    df.to_csv(out, index=False)
    print(f"\n===== ADDITIVE vs WM-SEARCH TABLE =====\n{df.drop(columns=['top10']).to_string(index=False)}", flush=True)
    print(f"\nsaved -> {out}", flush=True)
    print(f"\n[stage={a.stage} done]", flush=True)


if __name__ == '__main__':
    main()
