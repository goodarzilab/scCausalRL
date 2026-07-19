"""Cross-seed model-stability audit. Are the models themselves (esp. the learned DAG A) wildly
different across split seeds, or is the observed combo variance just test-set variance?
Compares mmd1_deg2 at split_seed {1,7,42}:
  (1) held-out SINGLE overlap across seeds (how different are the eval sets)
  (2) latent A structural stats — basis-INDEPENDENT (nnz, spectral radius, ||A||_F, mean|A|); note the
      64 latent programs are NOT aligned across independent runs, so element-wise A is meaningless.
  (3) FUNCTIONAL stability in GENE space: predicted drift-removed Δ for a FIXED set of TFs that are
      TRAIN in all 3 seeds, correlated pairwise across seeds. High corr = stable function; low = unstable.
"""
import os, sys, glob, re, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
torch.set_grad_enabled(False)
SEEDS = [42, 1, 7]
SRC = {42: 'v2b_mmd1_deg2', 1: 'v2b_mmd1_deg2_s1', 7: 'v2b_mmd1_deg2_s7'}

# per-seed splits + a shared control set
splits = {}
for s in SEEDS:
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt", split_seed=s, train_frac=0.9, val_frac=0.0)
    de = dd['de_single']
    splits[s] = dict(train=set(g for g in dd['pert_split']['train'] if g in de),
                     test=set(g for g in dd['pert_split']['test'] if g in de), de=de)
gn = None
Xc = None; ctrl = None
dd0 = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
gn = list(map(str, dd0['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
Xc = dd0['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd0['control_idx']); rng = np.random.default_rng(0)
xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 2000, replace=False)]])

# (1) held-out single overlap
print("=== (1) held-out SINGLE overlap across seeds ===", flush=True)
for i in range(len(SEEDS)):
    for j in range(i + 1, len(SEEDS)):
        a, b = splits[SEEDS[i]]['test'], splits[SEEDS[j]]['test']
        print(f"  s{SEEDS[i]} ({len(a)}) vs s{SEEDS[j]} ({len(b)}): common {len(a&b)}  Jaccard {len(a&b)/len(a|b):.2f}", flush=True)

# fixed TF set: train in ALL seeds (so every model can predict them cleanly, no leakage into any eval)
common_train = set.intersection(*[splits[s]['train'] for s in SEEDS])
fixed = sorted(common_train)[:60]
print(f"\nFixed comparison TF set: {len(fixed)} TFs train in ALL 3 seeds", flush=True)

# load models, A stats, predicted Δ (gene space, drift-removed) on fixed set
preds = {}
print("\n=== (2) latent A structural stats (basis-independent) ===", flush=True)
for s in SEEDS:
    ck = max(glob.glob(f'results/grid/{SRC[s]}/checkpoints/phase1_epoch*.pt'), key=lambda p: int(re.search(r'epoch(\d+)', p).group(1)))
    m, gnm, c = load_shift_model(ck, 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}
    A = m.dag.sample_A(1).squeeze(0)
    An = A.cpu().numpy(); ev = np.linalg.eigvals(An)
    nnz = float((np.abs(An) > 0.01).mean()); rho = float(np.abs(ev).max()); fro = float(np.linalg.norm(An)); mab = float(np.abs(An).mean())
    print(f"  s{s}: nnz(|A|>.01)={nnz:.3f}  spectral_radius={rho:.3f}  ||A||_F={fro:.2f}  mean|A|={mab:.4f}", flush=True)
    _, _, eps = m.encoder(xc); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    um = c.get('config', {}).get('phase1', {}).get('pert_mutilate_u_mode', 'shift'); bt = float(c.get('config', {}).get('phase1', {}).get('pert_mutilate_beta', 1.0))
    drift = np.stack([np.asarray(splits[s]['de'][g], np.float32) for g in splits[s]['train']]).mean(0)
    P = np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode=um, shift_beta=bt)[0]).mean(0).cpu().numpy() - cr) for g in fixed])
    preds[s] = P - drift    # drift-removed predicted Δ [fixed, G]

# (3) functional stability: pairwise per-TF correlation of predicted Δ across seeds
print("\n=== (3) FUNCTIONAL stability: predicted drift-removed Δ correlation across seeds (fixed TFs) ===", flush=True)
def corr_rows(A, B):
    return float(np.nanmean([np.corrcoef(A[i], B[i])[0, 1] for i in range(len(A)) if A[i].std() > 1e-9 and B[i].std() > 1e-9]))
for i in range(len(SEEDS)):
    for j in range(i + 1, len(SEEDS)):
        c = corr_rows(preds[SEEDS[i]], preds[SEEDS[j]])
        print(f"  s{SEEDS[i]} vs s{SEEDS[j]}: mean per-TF pred-Δ corr = {c:+.3f}", flush=True)
# also vs observed (upper bound of what's learnable), using seed-42 observed for the fixed TFs
obs42 = np.stack([np.asarray(splits[42]['de'][g], np.float32) for g in fixed])
drift42 = np.stack([np.asarray(splits[42]['de'][g], np.float32) for g in splits[42]['train']]).mean(0)
for s in SEEDS:
    print(f"  s{s} pred vs OBSERVED (fixed TFs, drift-removed): {corr_rows(preds[s], obs42 - drift42):+.3f}", flush=True)
print("\ndone", flush=True)
