"""P3 + Phase-1a gate — predict each Briggs combo's fate-axis alignment with the exp20 combo WM, then
correlate against the observed answer key (rho_obs).

rho_pred(C): do_mean composition WITH the interaction head via model.intervene_shift_mutilate(u_mode='shift')
  (sums per-gene shifts + Option-C pairwise interaction, per-single-mean u, structural single solve),
  Delta_pred = decode(z_post).mean - decode(z_ctrl).mean, DRIFT-CENTERED across combos (parity with
  rho_obs = combo_mean - screen_bulk), then cosine with the Cao HSC axis in the MODEL's gene space.
We use the Cao axis on BOTH sides (it maps by gene symbol to model AND Briggs space) so rho_pred and
rho_obs are comparable; the Briggs-native axis is a within-Briggs cross-check only.

Tiering: a combo is predictable iff all its TFs are on the model axis (ESM pert_label => zero-shot for any).
n_reliable = # TFs in effective_tf_genes.txt (seen in training); combos with any zero-shot TF flagged.

Reads results/briggs/observed_combo_targets.csv; writes results/briggs/phase1a_fidelity.csv + prints the GATE.
Run: python evaluation/briggs/predict_combo.py --ckpt results/grid/v2b_prod_deg8_all40/checkpoints/phase1_epoch40.pt
"""
import os, sys, csv, argparse, numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.briggs.hspc_sim import build_cao_hsc_signature
from data.load_joung import load_joung
from scipy.stats import spearmanr
torch.set_grad_enabled(False)
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/grid/v2b_prod_deg8_all40/checkpoints/phase1_epoch40.pt')
    ap.add_argument('--answer', default='results/briggs/observed_combo_targets.csv')
    ap.add_argument('--device', default='cpu'); ap.add_argument('--n_ctrl', type=int, default=2000)
    a = ap.parse_args()

    # ---- model + control encoding ----
    model, gene_names, _ = load_shift_model(a.ckpt, a.device)
    gn = list(map(str, gene_names)); g2i = {g: i for i, g in enumerate(gn)}
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None,
                    effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=a.device)
    _, _, eps = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0)
    z_prev = model.dag.scm_forward(eps, A); cr = model.decoder(z_prev).mean(0)     # control recon [G]
    reliable = set(open(V + "/effective_tf_genes.txt").read().split())
    has_head = getattr(model, 'interaction_U', None) is not None
    print(f"[model] genes={len(gn)} interaction_head={has_head} reliable_vocab={len(reliable)}", flush=True)

    # ---- Cao HSC fate axis in MODEL gene space ----
    d_cao, cidx = build_cao_hsc_signature(gn)
    print(f"[axis] Cao HSC axis genes in model space = {len(cidx)}", flush=True)

    # ---- read answer key ----
    obs = list(csv.DictReader(open(a.answer)))
    print(f"[answer] {len(obs)} observed combos", flush=True)

    rows, deltas, deltas_add = [], [], []
    skipped_offaxis = 0
    iu = model.interaction_U                                                       # for the head-OFF additive counterfactual
    for r in obs:
        tfs = r['combo'].split('|'); gi = [g2i[g] for g in tfs if g in g2i]
        if len(gi) != len(tfs): skipped_offaxis += 1; continue                    # some TF not on model axis
        z_post, _ = model.intervene_shift_mutilate(eps, gi, A, z_prev, u_mode='shift', shift_beta=1.0)
        deltas.append((model.decoder(z_post).mean(0) - cr).cpu().numpy())          # do_mean WITH interaction head
        model.interaction_U = None                                                 # identical path, interaction OFF
        z_add, _ = model.intervene_shift_mutilate(eps, gi, A, z_prev, u_mode='shift', shift_beta=1.0)
        model.interaction_U = iu
        deltas_add.append((model.decoder(z_add).mean(0) - cr).cpu().numpy())       # model-additive counterfactual
        rows.append(dict(combo=r['combo'], k=int(r['k']),
                         n_reliable=sum(t in reliable for t in tfs),
                         rho_obs=float(r['rho_cao']), rho_obs_briggs=float(r['rho_briggs']),
                         is_P1=int(r['is_P1']), is_P2=int(r['is_P2'])))
    D = np.stack(deltas); D = D - D.mean(0)                                        # drift-center (parity w/ obs)
    Da = np.stack(deltas_add); Da = Da - Da.mean(0)
    for i, r in enumerate(rows):
        r['rho_pred'] = cos(D[i][cidx], d_cao); r['rho_add'] = cos(Da[i][cidx], d_cao)
    print(f"[predict] {len(rows)} predicted | skipped {skipped_offaxis} (TF off model axis)", flush=True)

    od = 'results/briggs'; os.makedirs(od, exist_ok=True); out = f'{od}/phase1a_fidelity.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['combo', 'k', 'n_reliable', 'rho_pred', 'rho_add', 'rho_obs', 'rho_obs_briggs', 'is_P1', 'is_P2'])
        w.writeheader(); w.writerows(rows)

    # ---- GATE ----
    rp = np.array([r['rho_pred'] for r in rows]); ra = np.array([r['rho_add'] for r in rows])
    ro = np.array([r['rho_obs'] for r in rows]); ks = np.array([r['k'] for r in rows])
    sp_all = spearmanr(rp, ro).correlation; sp_add = spearmanr(ra, ro).correlation
    q = np.quantile(ro, 0.9); pq = np.quantile(rp, 0.9)
    top_obs = ro >= q; top_pred = rp >= pq
    recall = float((top_obs & top_pred).sum() / max(1, top_obs.sum()))
    print(f"\n===== PHASE-1a WM FIDELITY GATE =====", flush=True)
    print(f"  Spearman(rho_pred, rho_obs)  ALL {len(rp)} combos = {sp_all:+.3f}   [interaction head ON]", flush=True)
    print(f"  Spearman(rho_add,  rho_obs)  ALL {len(rp)} combos = {sp_add:+.3f}   [additive, head OFF]", flush=True)
    print(f"  --> interaction head fidelity gain = {sp_all - sp_add:+.3f}", flush=True)
    print(f"  top-decile recall (pred top10% ∩ obs top10%) = {recall:.2f}  (chance 0.10)", flush=True)
    allrel = np.array([r['n_reliable'] == r['k'] for r in rows])
    if allrel.sum() > 10:
        print(f"  RELIABLE-only ({allrel.sum()}): Spearman {spearmanr(rp[allrel], ro[allrel]).correlation:+.3f}", flush=True)
    if (~allrel).sum() > 10:
        print(f"  has-zero-shot ({(~allrel).sum()}): Spearman {spearmanr(rp[~allrel], ro[~allrel]).correlation:+.3f}", flush=True)
    for kk in sorted(set(ks.tolist())):
        m = ks == kk
        if m.sum() >= 15:
            print(f"    k={kk} (n={int(m.sum())}): Spearman {spearmanr(rp[m], ro[m]).correlation:+.3f}", flush=True)
    for nm in ('P1', 'P2'):
        mm = [r for r in rows if r[f'is_{nm}']]
        if mm:
            r = mm[0]; pct = float((rp < r['rho_pred']).mean() * 100)
            print(f"  {nm}: rho_pred {r['rho_pred']:+.3f} (pred pctile {pct:.0f}) vs rho_obs {r['rho_obs']:+.3f}", flush=True)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
