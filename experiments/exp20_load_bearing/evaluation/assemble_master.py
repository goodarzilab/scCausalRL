"""Assemble the exp20 master tables from the canonical evals (same scripts as always):
  - heldout_zeroshot_eval.py  -> drift-removed discrimination / pearson / deg-auroc (singles)
  - grn_dag_sweep.py          -> A load-bearing (real/zero/shuf) + GRN biological-ness vs priors
  - component_ablation.py     -> which component carries the signal (shift / decoder / A)
  - agap_probe.py (log)       -> gate_open_frac + probe A_gap (does A change the output)

Writes docs/MASTER_RESULTS.md and results/MASTER_eval_x20.csv.
"""
import os, sys, json, glob, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RLA = '/home/sgoudarzi/scCausalRL/experiments_rl/analysis'
ARMS = ['struct_min_esm', 'struct_sparse_esm', 'struct_sparse_con', 'struct_sparse_resid',
        'struct_sparse_pertonly', 'struct_sparse_residrecon', 'struct_sparse_all']
short = lambda t: t.replace('struct_', '')


def heldout_row(tag):
    p = f'{ROOT}/results/heldout_eval/hz_x20_{tag}/summary.json'
    if not os.path.exists(p):
        return {}
    s = json.load(open(p))
    return {k: s.get(k) for k in ['discrimination_cos_resid', 'pearson_delta_resid', 'deg_auroc_resid',
                                  'discrimination_cos', 'pearson_delta', 'retrieval_top1',
                                  'add_discrimination_cos_resid', 'add_pearson_delta_resid']}


def parse_agap(log):
    """Parse the agap_x20 sbatch log into {tag: (gate_open, A_gap_all, A_gap_nt)}."""
    out = {}
    if not log or not os.path.exists(log):
        return out
    cur = None
    for ln in open(log):
        m = re.match(r'#+\s*(struct_\S+)', ln)
        if m: cur = m.group(1); continue
        if cur is None: continue
        if 'gate_open_frac' in ln:
            g = float(re.search(r'=\s*([0-9.]+)', ln).group(1)); out.setdefault(cur, {})['gate'] = g
        elif 'A_gap (all' in ln:
            g = float(re.search(r'=\s*([\-0-9.]+)', ln).group(1)); out.setdefault(cur, {})['gap'] = g
        elif 'A_gap (non-target' in ln and 'n/a' not in ln:
            g = float(re.search(r'=\s*([\-0-9.]+)', ln).group(1)); out.setdefault(cur, {})['gap_nt'] = g
    return out


def main():
    grn = pd.read_csv(f'{RLA}/grn_dag_sweep_x20.csv').set_index('WM')
    comp = pd.read_csv(f'{RLA}/component_ablation_x20.csv')
    logs = sorted(glob.glob(f'{ROOT}/logs/agap_x20_*.out'))
    ag = parse_agap(logs[-1] if logs else None)

    rows = []
    for tag in ARMS:
        r = {'arm': short(tag)}
        r.update(heldout_row(tag))
        if tag in grn.index:
            g = grn.loc[tag]
            r['S_real'] = g.s_spear_real; r['S_zero'] = g.s_spear_zero; r['S_shuf'] = g.s_spear_shuf
            r['S_Agap'] = g.s_spear_real - max(g.s_spear_zero, g.s_spear_shuf)
            r['C_real'] = g.c_spear_real; r['C_Agap'] = g.c_spear_real - max(g.c_spear_zero, g.c_spear_shuf)
            r['collectri_auroc'] = g.collectri_auroc; r['shuffA_auroc'] = g.shuffA
            r['dorothea_held'] = g.dorothea_held; r['corrGRN'] = g.corrGRN; r['randomD'] = g.randomD
        a = ag.get(tag, {})
        r['gate_open'] = a.get('gate'); r['probe_Agap'] = a.get('gap'); r['probe_Agap_nt'] = a.get('gap_nt')
        cs = comp[(comp.WM == tag) & (comp.split == 'S')]
        if len(cs):
            c = cs.iloc[0]
            r['abl_shift_zero'] = c.spear - c['shift_zero.spear']
            r['abl_D_rand'] = c.spear - c['D_rand.spear']
            r['abl_A_zero'] = c.spear - c['A_zero.spear']
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f'{ROOT}/results/MASTER_eval_x20.csv', index=False)

    def fmt(v, p=3):
        return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{p}f}'

    L = ['# exp20 master results — did the structural+sparse fix make A load-bearing?\n']
    L.append('## 1. Drift-removed prediction (held-out singles) — the metric that matters\n')
    L.append('| arm | disc_cos_resid | pearson_δ_resid | deg_auroc_resid | disc_cos (raw) | retrieval_top1 |')
    L.append('|---|---|---|---|---|---|')
    for r in rows:
        L.append(f"| {r['arm']} | {fmt(r.get('discrimination_cos_resid'))} | {fmt(r.get('pearson_delta_resid'))} "
                 f"| {fmt(r.get('deg_auroc_resid'))} | {fmt(r.get('discrimination_cos'))} | {fmt(r.get('retrieval_top1'))} |")
    L.append('\n_exp19 best for reference: v2_esm_con_epsmix disc_cos_resid=0.647; ceiling ~0.55–0.65._\n')

    L.append('## 2. Is A load-bearing? (drift-removed spearman: real vs zero-A vs shuffled-A)\n')
    L.append('A_gap>0 ⇒ real A **improves** prediction. probe_Agap>0 ⇒ A **changes** the output (mechanistically used).\n')
    L.append('| arm | S_real | S_zero | S_shuf | **S_Agap** | C_Agap | gate_open | probe_Agap | probe_Agap(nt) |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        L.append(f"| {r['arm']} | {fmt(r.get('S_real'))} | {fmt(r.get('S_zero'))} | {fmt(r.get('S_shuf'))} "
                 f"| {fmt(r.get('S_Agap'))} | {fmt(r.get('C_Agap'))} | {fmt(r.get('gate_open'))} "
                 f"| {fmt(r.get('probe_Agap'))} | {fmt(r.get('probe_Agap_nt'))} |")

    L.append('\n## 3. Component ablation (Δspearman when ablated; larger drop = more load-bearing)\n')
    L.append('| arm | shift→0 | D→random | A→0 |')
    L.append('|---|---|---|---|')
    for r in rows:
        L.append(f"| {r['arm']} | {fmt(r.get('abl_shift_zero'))} | {fmt(r.get('abl_D_rand'))} | {fmt(r.get('abl_A_zero'))} |")

    L.append('\n## 4. GRN biological-ness (projected G=D·A·Dᵀ vs priors, AUROC)\n')
    L.append('shuffA≈collectri ⇒ the "biological" signal is decoder-coexpression (D·Dᵀ), not the causal A.\n')
    L.append('| arm | collectri | shuffA | dorothea_held | corrGRN(base) | randomD(chance) |')
    L.append('|---|---|---|---|---|---|')
    for r in rows:
        L.append(f"| {r['arm']} | {fmt(r.get('collectri_auroc'))} | {fmt(r.get('shuffA_auroc'))} "
                 f"| {fmt(r.get('dorothea_held'))} | {fmt(r.get('corrGRN'))} | {fmt(r.get('randomD'))} |")

    open(f'{ROOT}/docs/MASTER_RESULTS.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nsaved -> {ROOT}/results/MASTER_eval_x20.csv and docs/MASTER_RESULTS.md')


if __name__ == '__main__':
    main()
