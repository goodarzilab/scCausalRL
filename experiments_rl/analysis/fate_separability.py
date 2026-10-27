"""CEILING CHECK before reframing Joung RL to fate targets: how separable are the developmental
FATES in the frozen world model? If fates collapse onto the shared "leaving-pluripotency" drift,
a fate-reward RL has no signal; if they diverge in the residual, the reframing has a real ceiling.

Fate label per TF = our paper-validated annotation (joung_human_annotation_celllevel.csv, cosine
cell-level). We measure separability in three spaces, RAW and DRIFT-REMOVED (subtract the mean
rep over TFs = the perturb-mean drift), at coarse-lineage and fine-cell-type granularity:
  latent_target   z_target_single[g]            -- the state the RL targets   (state=[z_t; z_tgt])
  pred_delta      decode(do-op(g)) - decode(z0) -- what the reward Pearson sees (policy output)
  obs_delta       de_single[g].lfc              -- data ceiling (what's achievable in principle)

Metrics (fates with >=3 confident TFs): nearest-fate-centroid leave-one-out accuracy (cosine),
fate-retrieval recall@1/3, silhouette, and drift fraction ||mean||^2 / mean||rep||^2.

Run: python analysis/fate_separability.py --checkpoint <ugate epoch60> --wm_root <exp19>
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import add_stack_args, build_stack
ANN = ('/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/analysis/'
       'human_dev_annotation/joung_human_annotation_celllevel.csv')


def coarse_fate(ct):
    s = str(ct).lower()
    if any(k in s for k in ['trophoblast', 'syncytio', 'cytotrophoblast', 'csh1', 'placenta']): return 'TROPHOBLAST'
    if any(k in s for k in ['photoreceptor', 'amacrine', 'bipolar', 'ganglion', 'horizontal', 'retinal', 'lens', 'muller', 'corneal']): return 'EYE'
    if any(k in s for k in ['neuron', 'neural', 'glia', 'oligodendro', 'astrocyte', 'schwann', 'purkinje', 'granule', 'microglia', 'skor2', 'satb2', 'slc24a4']): return 'NEURAL'
    if any(k in s for k in ['erythro', 'megakaryo', 'myeloid', 'lymphoid', 'thymocyte', 'hematop', 'antigen presenting', 'clc_il5ra']): return 'BLOOD'
    if any(k in s for k in ['endothel', 'vascular', 'endocardial']): return 'ENDOTHELIAL'
    if any(k in s for k in ['hepato', 'pancrea', 'acinar', 'ductal', 'intestin', 'lung', 'alveolar', 'bronchiolar', 'goblet', 'parietal', 'stomach', 'squamous', 'ciliated', 'epithel', 'metanephric', 'ureteric', 'mesangial', 'neuroendocrine', 'islet', 'enteroendocrine', 'pde11a', 'pde1c', 'muc13', 'elf3', 'paep']): return 'EPI_ENDO_KIDNEY'
    if any(k in s for k in ['muscle', 'myocyte', 'cardiac', 'chondro', 'osteo', 'satellite', 'sympathoblast', 'chromaffin']): return 'MUSCLE_MESO'
    if any(k in s for k in ['stromal', 'mesenchym', 'fibroblast', 'mesothel', 'stellate', 'epicardial', 'stc2']): return 'STROMAL'
    return 'OTHER'


def drift_remove(M):
    return M - M.mean(0, keepdims=True)


def metrics(M, labels, name):
    """M [n, dim], labels [n]. Returns dict of separability metrics over fates with >=3 members."""
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels); uniq, cnt = np.unique(labels, return_counts=True)
    keep_f = set(uniq[cnt >= 3]); mask = np.array([l in keep_f for l in labels])
    M = M[mask]; labels = labels[mask]
    if len(set(labels)) < 2: return {'space': name, 'n': int(mask.sum()), 'nfate': len(set(labels))}
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)        # cosine geometry
    fates = sorted(set(labels)); n = len(labels)
    # leave-one-out nearest-fate-centroid accuracy + retrieval recall@k
    cent = {f: Mn[labels == f].sum(0) for f in fates}; csz = {f: (labels == f).sum() for f in fates}
    r1 = r3 = acc = 0
    for i in range(n):
        f0 = labels[i]
        sc = {f: (cent[f] - (Mn[i] if f == f0 else 0)) / (csz[f] - (1 if f == f0 else 0)) for f in fates}
        order = sorted(fates, key=lambda f: -float(Mn[i] @ (sc[f] / (np.linalg.norm(sc[f]) + 1e-9))))
        acc += order[0] == f0; r1 += f0 in order[:1]; r3 += f0 in order[:3]
    drift = float((M.mean(0) ** 2).sum() / (M ** 2).sum(1).mean())
    sil = float(silhouette_score(Mn, labels, metric='cosine'))
    chance = 1.0 / len(fates)
    return {'space': name, 'n': n, 'nfate': len(fates), 'NC_acc': acc / n, 'recall@1': r1 / n,
            'recall@3': r3 / n, 'silhouette': sil, 'drift_frac': drift, 'chance_acc': chance}


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--qmax', type=float, default=0.10, help='fate-confidence FDR q cutoff')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, meta, op, data = build_stack(a, dev, n_dag=8)

    ann = pd.read_csv(ANN); ann['TF'] = ann['TF'].astype(str)
    fate = dict(zip(ann['TF'], ann['top_celltype'])); qv = dict(zip(ann['TF'], ann['fdr_qval']))

    for qmax, tag in [(1.01, 'ALL annotated'), (a.qmax, f'confident q<{a.qmax}')]:
        tfs = [g for g in data.screened_genes if g in fate and qv.get(g, 1) < qmax]
        if len(tfs) < 30:
            print(f"\n### {tag}: only {len(tfs)} TFs — skip"); continue
        idx = torch.as_tensor([data.gene_to_idx[g] for g in tfs], device=dev)
        # representations
        lat = torch.stack([data.z_target_single[g] for g in tfs]).cpu().numpy()          # latent target
        S = op.shift_for(idx.tolist())                                                    # [K,d]
        zp = op.apply_many(data.eps_ctrl_mean, S, data.A_ref)                             # [K,d] gate-aware do-op
        z0 = data.z0
        pred_lat = (zp - z0.unsqueeze(0)).cpu().numpy()
        pred_gene = (op.decode(zp) - op.decode(z0.unsqueeze(0))).cpu().numpy()
        obs_gene = np.stack([data.de_single[g]['lfc'] for g in tfs])
        fc = np.array([coarse_fate(fate[g]) for g in tfs])
        ff = np.array([str(fate[g]) for g in tfs])

        print(f"\n{'='*92}\n### {tag}: {len(tfs)} TFs | coarse fates: "
              f"{dict(zip(*np.unique(fc, return_counts=True)))}")
        for gran, labels in [('coarse', fc), ('fine', ff)]:
            rows = []
            for nm, M in [('latent_target', lat), ('pred_delta_latent', pred_lat),
                          ('pred_delta_gene', pred_gene), ('obs_delta_gene (data ceiling)', obs_gene)]:
                rows.append(metrics(M, labels, f'{nm} RAW'))
                rows.append(metrics(drift_remove(M), labels, f'{nm} RESID'))
            df = pd.DataFrame([r for r in rows if 'NC_acc' in r])
            print(f"\n-- {gran} fate ({df['nfate'].iloc[0]} fates, chance acc {df['chance_acc'].iloc[0]:.2f}) --")
            print(df[['space', 'n', 'nfate', 'NC_acc', 'recall@1', 'recall@3', 'silhouette', 'drift_frac']]
                  .to_string(index=False, float_format=lambda x: f'{x:.3f}'))
            if gran == 'coarse' and 'confident' in tag:
                df.to_csv(f'{HERE}/fate_separability_coarse.csv', index=False)
    print(f"\nsaved -> fate_separability_coarse.csv")


if __name__ == '__main__':
    main()
