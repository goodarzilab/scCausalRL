"""BOTH-SIDES eval for the lb11 FINE-TUNE arms (warm-started from the best singles model
v2b_mmd1_deg2). For each arm, SINGLES metrics (disc/Pnk/allDEG@0.15, drift-removed, KO-excluded,
real-A, on the 49 held-out TFs) reusing the exact agap_correctness.suite logic so numbers are
comparable to the master singles table. Combos are scored separately by combo_eval_full.py.

Each arm's checkpoint = its LATEST periodic epoch (phase1_epoch{max}.pt) — fine-tune arms use
different n_epochs (frozen=40, replay=25->epoch20, fullcombo/combofrac=60), and periodic saves are
every 10 epochs, so we take the newest available rather than assuming epoch60.

References: mmd1_deg2 (singles-only source) and deg2_Jcombo_ho50 (from-scratch combo-trained) — the
two endpoints of the frontier the fine-tunes try to Pareto-dominate.
Run: python evaluation/eval_ft.py --device cpu
"""
import os, sys, glob, re, argparse, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"

# (tag, note). Checkpoint auto-discovered = latest phase1_epoch*.pt under results/grid/<tag>.
ARMS = [
    ('v2b_mmd1_deg2',              'REF: singles-only source'),
    ('v2b_mmd1_deg2_Jcombo_ho50',  'REF: from-scratch combo-trained (50/50)'),
    ('v2b_ft_ixhead_frozen_fx',    '(1) frozen singles + interaction head only'),
    ('v2b_ft_replay_fx',           '(2) warm-start low-LR singles+combo replay'),
    ('v2b_ft_fullcombo_fx',        '(3) warm-start full combo training'),
    ('v2b_ft_combofrac0p1_fx',     '(4) warm-start combo_frac=0.1'),
]


def latest_ckpt(tag):
    cks = glob.glob(f'results/grid/{tag}/checkpoints/phase1_epoch*.pt')
    if not cks:
        return None
    return max(cks, key=lambda p: int(re.search(r'epoch(\d+)', p).group(1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu')
    ap.add_argument('--arms', default='', help='comma list of tags to override the default ARMS (refs auto-prepended)')
    a = ap.parse_args()
    global ARMS
    if a.arms:
        refs = [r for r in ARMS if r[0].startswith('v2b_mmd1_deg2')][:2]
        ARMS = refs + [(t, '(swept arm)') for t in a.arms.split(',') if t]
    torch.set_grad_enabled(False)
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
    de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
    tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
    obs = np.stack([np.asarray(de[g], np.float32) for g in te])
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 1500, replace=False)]])

    def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

    def suite(P):
        Pr, Or = P - drift, obs - drift
        C = zr(Pr) @ zr(Or).T; d = np.diag(C)[:, None]; T = len(P); disc = float(((d > C).sum(1) / (T - 1)).mean())
        pnk, adeg = [], []
        for i, g in enumerate(te):
            keep = np.ones(P.shape[1], bool); keep[gidx[g]] = False
            x, y = Pr[i, keep], Or[i, keep]; pnk.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 else np.nan)
            deg = keep & (np.abs(Or[i]) > 0.15); xd, yd = Pr[i, deg], Or[i, deg]
            adeg.append(np.corrcoef(xd, yd)[0, 1] if deg.sum() > 2 and xd.std() > 1e-9 else np.nan)
        return disc, float(np.nanmean(pnk)), float(np.nanmean(adeg))

    print(f"{'arm':30}{'epoch':>6}{'disc':>7}{'Pnk':>7}{'allDEG':>8}   note", flush=True)
    for tag, note in ARMS:
        ck = latest_ckpt(tag)
        if ck is None:
            print(f"{tag:30}{'--':>6}{'(no checkpoint yet)':>23}   {note}", flush=True); continue
        ep = int(re.search(r'epoch(\d+)', ck).group(1))
        try:
            m, gnm, c = load_shift_model(ck, a.device); g2 = {g: i for i, g in enumerate(gnm)}
            _, _, eps = m.encoder(xc)
            um = c.get('config', {}).get('phase1', {}).get('pert_mutilate_u_mode', 'shift')
            bt = float(c.get('config', {}).get('phase1', {}).get('pert_mutilate_beta', 1.0))
            m.dag.disabled = False; A = m.dag.sample_A(1).squeeze(0); zp0 = m.dag.scm_forward(eps, A)
            cr = m.decoder(zp0).mean(0).cpu().numpy()
            P = np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp0, u_mode=um, shift_beta=bt)[0]).mean(0).cpu().numpy() - cr) for g in te])
            disc, pnk, adeg = suite(P)
            print(f"{tag:30}{ep:>6}{disc:>7.3f}{pnk:>7.3f}{adeg:>8.3f}   {note}", flush=True)
        except Exception as e:
            print(f"{tag:30}{ep:>6}  FAIL {type(e).__name__}: {e}   {note}", flush=True)


if __name__ == '__main__':
    main()
