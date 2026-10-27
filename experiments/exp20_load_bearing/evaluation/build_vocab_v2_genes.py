"""Build vocab-v2 gene list: top-4000 Seurat-v3 HVG (on raw counts, à la Jain 2026 re-analysis) UNION
curated fate markers. Reports overlap with our current 9923-HVG proc vocab -> tells us whether we can
just subset the existing proc file or need a full rebuild from raw.

Run on a high-mem node (raw h5ad is large; we subsample cells for HVG stability).
"""
import os, sys, argparse, numpy as np
import scanpy as sc, anndata as ad

RAW = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/GSE217460_210322_TFAtlas_raw.h5ad"
PROC = "/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad"
VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n_hvg', type=int, default=4000)
    ap.add_argument('--subsample', type=int, default=200000); a = ap.parse_args()

    print(f"[raw] opening {RAW} (backed)", flush=True)
    A = ad.read_h5ad(RAW, backed='r')
    print(f"[raw] shape {A.shape}; var head: {list(A.var_names[:5])}; obs cols: {list(A.obs.columns)[:8]}", flush=True)
    n = A.shape[0]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, size=min(a.subsample, n), replace=False))
    print(f"[raw] subsampling {len(idx)}/{n} cells for HVG", flush=True)
    sub = A[idx].to_memory()
    X = sub.X
    is_int = np.allclose(X[:100].toarray() if hasattr(X, 'toarray') else X[:100],
                         np.round(X[:100].toarray() if hasattr(X, 'toarray') else X[:100]))
    print(f"[raw] X looks like counts: {is_int}", flush=True)
    if not is_int:
        print("[warn] X not integer counts; seurat_v3 expects counts. Proceeding anyway.", flush=True)
    sc.pp.highly_variable_genes(sub, n_top_genes=a.n_hvg, flavor='seurat_v3')
    hvg = sub.var_names[sub.var['highly_variable'].values].astype(str).tolist()
    print(f"[hvg] top-{a.n_hvg} seurat_v3 HVG computed: {len(hvg)} genes", flush=True)

    # fate markers (curated S4B)
    fm = []
    fmf = os.path.join(VDIR, 'fate_markers_curated.txt')
    if os.path.exists(fmf):
        fm = [g.strip() for g in open(fmf) if g.strip()]
    raw_genes = set(A.var_names.astype(str))
    fm_in = [g for g in fm if g in raw_genes]
    vocab = sorted(set(hvg) | set(fm_in))
    print(f"[fate] curated markers {len(fm)}, in raw {len(fm_in)}, add beyond HVG: {len(set(fm_in)-set(hvg))}", flush=True)

    # overlap with current 9923 proc vocab
    P = ad.read_h5ad(PROC, backed='r'); cur = set(P.var_names.astype(str))
    inboth = set(vocab) & cur
    print(f"\n=== VOCAB-V2 GENES ===", flush=True)
    print(f"  HVG(4000) ∪ fate = {len(vocab)} genes", flush=True)
    print(f"  in current 9923 proc: {len(inboth)} ({100*len(inboth)/len(vocab):.1f}%)", flush=True)
    print(f"  NOT in current proc (need raw rebuild if kept): {len(set(vocab)-cur)}", flush=True)
    print(f"  -> if overlap high, just SUBSET the existing proc; else rebuild from raw.", flush=True)

    open(os.path.join(VDIR, 'vocab_v2_genes.txt'), 'w').write("\n".join(vocab) + "\n")
    open(os.path.join(VDIR, 'vocab_v2_genes_in_current_proc.txt'), 'w').write("\n".join(sorted(inboth)) + "\n")
    np.savez(os.path.join(VDIR, 'vocab_v2_meta.npz'), hvg=np.array(hvg), fate=np.array(fm_in),
             vocab=np.array(vocab), in_proc=np.array(sorted(inboth)))
    print(f"\nwrote vocab_v2_genes.txt ({len(vocab)}) to {VDIR}", flush=True)


if __name__ == '__main__':
    main()
