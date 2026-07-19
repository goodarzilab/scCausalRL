"""Briggs fidelity SWEEP — rank many trained WMs by how well their one-step do-op predicts Briggs's
endpoint HSPC fate direction. Shared across all v2b models: answer key, Cao axis, combo->gene mapping,
control data (load once). Per model: re-encode control, predict each combo's rho_pred (do_mean + head),
drift-center, project on Cao axis, Spearman vs rho_obs. Signal lives at low k, so we predict the reliable
low-k combos and report Spearman at k<=2 and k<=3 (the discriminating slices) + all-predicted + P1 pctile.

Runs on a subset of combos (reliable vocab, k<=max_k) that is IDENTICAL across v2b models (data-side),
so rankings are comparable. Reads a newline-list of run tags via --runs_file (grid tag per line).
Writes results/briggs/sweep_fidelity.csv (one row per model, sorted by rel-k<=3 Spearman).
Run: python evaluation/briggs/sweep_fidelity.py --runs_file <tags.txt> --max_k 4 --n_ctrl 128
"""
import os, sys, csv, argparse, glob, re, numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.briggs.hspc_sim import build_cao_hsc_signature
from data.load_joung import load_joung
from scipy.stats import spearmanr
torch.set_grad_enabled(False)
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"


def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def latest(tag):
    cks = glob.glob(f'results/grid/{tag}/checkpoints/phase1_epoch*.pt')
    return max(cks, key=lambda p: int(re.search(r'epoch(\d+)', p).group(1))) if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_file', required=True); ap.add_argument('--answer', default='results/briggs/observed_combo_targets.csv')
    ap.add_argument('--max_k', type=int, default=4); ap.add_argument('--n_ctrl', type=int, default=128); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    tags = [t.strip() for t in open(a.runs_file) if t.strip() and not t.startswith('#')]

    # ---- shared: control data + answer key (loaded ONCE) ----
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None,
                    effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc_np = Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]]
    reliable = set(open(V + "/effective_tf_genes.txt").read().split())
    obs = list(csv.DictReader(open(a.answer)))
    P1 = {'TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2'}
    print(f"[shared] {len(obs)} answer combos | n_ctrl {len(xc_np)} | {len(tags)} models to sweep", flush=True)

    out_rows = []
    for ti, tag in enumerate(tags):
        ck = latest(tag)
        if not ck: print(f"[{ti+1}/{len(tags)}] {tag}: NO checkpoint", flush=True); continue
        try:
            model, gene_names, _ = load_shift_model(ck, a.device)
        except Exception as e:
            print(f"[{ti+1}/{len(tags)}] {tag}: LOAD FAIL {e}", flush=True); continue
        gn = list(map(str, gene_names)); g2i = {g: i for i, g in enumerate(gn)}
        d_cao, cidx = build_cao_hsc_signature(gn)
        _, _, eps = model.encoder(torch.tensor(xc_np, device=a.device)); A = model.dag.sample_A(1).squeeze(0)
        z_prev = model.dag.scm_forward(eps, A); cr = model.decoder(z_prev).mean(0)
        has_head = getattr(model, 'interaction_U', None) is not None

        # eval subset: reliable-vocab, k<=max_k, all TFs on model axis (identical across v2b models)
        rows, deltas = [], []
        for r in obs:
            tfs = r['combo'].split('|'); k = int(r['k'])
            if k > a.max_k: continue
            if any(t not in reliable for t in tfs): continue
            gi = [g2i[g] for g in tfs if g in g2i]
            if len(gi) != len(tfs): continue
            z_post, _ = model.intervene_shift_mutilate(eps, gi, A, z_prev, u_mode='shift', shift_beta=1.0)
            deltas.append((model.decoder(z_post).mean(0) - cr).cpu().numpy())
            rows.append(dict(combo=r['combo'], k=k, rho_obs=float(r['rho_cao']), is_P1=int(r['is_P1'])))
        if len(rows) < 30: print(f"[{ti+1}/{len(tags)}] {tag}: too few combos ({len(rows)})", flush=True); continue
        D = np.stack(deltas); D = D - D.mean(0)
        rp = np.array([cos(D[i][cidx], d_cao) for i in range(len(rows))])
        ro = np.array([r['rho_obs'] for r in rows]); ks = np.array([r['k'] for r in rows])
        sp_all = spearmanr(rp, ro).correlation
        sp2 = spearmanr(rp[ks <= 2], ro[ks <= 2]).correlation if (ks <= 2).sum() > 10 else float('nan')
        sp3 = spearmanr(rp[ks <= 3], ro[ks <= 3]).correlation if (ks <= 3).sum() > 10 else float('nan')
        p1 = [i for i, r in enumerate(rows) if r['is_P1']]
        p1pct = float((rp < rp[p1[0]]).mean() * 100) if p1 else float('nan')
        row = dict(run=tag, epoch=int(re.search(r'epoch(\d+)', ck).group(1)), head=int(has_head),
                   n=len(rows), sp_all=round(sp_all, 4), sp_k2=round(sp2, 4), sp_k3=round(sp3, 4), P1_pct=round(p1pct, 1))
        out_rows.append(row)
        print(f"[{ti+1}/{len(tags)}] {tag:34s} head={int(has_head)} n={len(rows):4d} | "
              f"k<=2 {sp2:+.3f}  k<=3 {sp3:+.3f}  all {sp_all:+.3f}  P1pct {p1pct:.0f}", flush=True)

    out_rows.sort(key=lambda r: (-1 if np.isnan(r['sp_k3']) else -r['sp_k3']))
    od = 'results/briggs'; os.makedirs(od, exist_ok=True); out = f'{od}/sweep_fidelity.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'epoch', 'head', 'n', 'sp_k2', 'sp_k3', 'sp_all', 'P1_pct']); w.writeheader(); w.writerows(out_rows)
    print(f"\n===== TOP 15 by reliable-k<=3 Spearman =====", flush=True)
    for r in out_rows[:15]:
        print(f"  {r['run']:34s} head={r['head']} | k<=2 {r['sp_k2']:+.3f}  k<=3 {r['sp_k3']:+.3f}  all {r['sp_all']:+.3f}  P1pct {r['P1_pct']:.0f}", flush=True)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
