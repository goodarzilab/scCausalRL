"""Unify ALL exp20 arms (prior-on x20, prior-off x20po, Phase-2 architecture x20arch) into one
master table: drift-removed SINGLES + COMBOS prediction, A-load-bearing (real vs zero vs shuf),
and the mechanism probes (probe_A / probe_Dperm / crossTF_div). Writes docs/MASTER_ALL.md + csv.
"""
import os, sys, json, glob, re
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RLA = '/home/sgoudarzi/scCausalRL/experiments_rl/analysis'

GROUPS = {  # tag -> (grn_csv, group label, prior)
    'x20':     ('grn_dag_sweep_x20.csv',     'prior-ON'),
    'x20po':   ('grn_dag_sweep_x20po.csv',   'prior-off'),
    'x20arch': ('grn_dag_sweep_x20arch.csv', 'prior-off'),
}
# heldout summary dir uses different tag prefixes per batch; try them all
HZ_CANDIDATES = lambda tag: [f'hz_x20_{tag}', f'hz_{tag}', f'hz_x20po_{tag}']


def heldout(tag):
    for name in HZ_CANDIDATES(tag):
        p = f'{ROOT}/results/heldout_eval/{name}/summary.json'
        if os.path.exists(p):
            s = json.load(open(p))
            return {'disc_resid': s.get('discrimination_cos_resid'),
                    'pear_resid': s.get('pearson_delta_resid'),
                    'deg_resid': s.get('deg_auroc_resid')}
    return {}


def parse_probes():
    """Scan all eval logs for component_probe blocks -> {tag: {probe_A, probe_Dperm, crossTF}}."""
    out = {}
    for lg in glob.glob(f'{ROOT}/logs/*.out'):
        try:
            txt = open(lg, errors='ignore').read()
        except Exception:
            continue
        for m in re.finditer(r'COMPONENT PROBE \| (\S+).*?probe_A.*?= ([\d.]+).*?probe_Dperm.*?= ([\d.]+).*?crossTF_div.*?= ([\d.\-]+)', txt, re.S):
            tag = m.group(1)
            out[tag] = {'probe_A': float(m.group(2)), 'probe_Dperm': float(m.group(3)), 'crossTF': float(m.group(4))}
    return out


def main():
    probes = parse_probes()
    rows = []
    for grid, (csvname, prior) in GROUPS.items():
        p = f'{RLA}/{csvname}'
        if not os.path.exists(p):
            print(f'[skip] {csvname} missing'); continue
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            tag = r['WM']
            hz = heldout(tag)
            pb = probes.get(tag, {})
            name = tag.replace('struct_', '').replace('_po', '')
            rows.append({
                'arm': name, 'prior': prior,
                'disc_resid': hz.get('disc_resid'), 'pear_resid': hz.get('pear_resid'),
                'S_real': r['s_spear_real'], 'S_Agap': r['s_spear_real'] - max(r['s_spear_zero'], r['s_spear_shuf']),
                'C_real': r['c_spear_real'], 'C_Agap': r['c_spear_real'] - max(r['c_spear_zero'], r['c_spear_shuf']),
                'C_disc': r['c_disc_real'],
                'probe_A': pb.get('probe_A'), 'probe_Dperm': pb.get('probe_Dperm'), 'crossTF': pb.get('crossTF'),
                'collectri': r['collectri_auroc'], 'shuffA': r['shuffA'],
            })
    df = pd.DataFrame(rows)
    df.to_csv(f'{ROOT}/results/MASTER_ALL_x20.csv', index=False)

    fm = lambda v, p=3: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{p}f}'
    L = ['# exp20 MASTER — all arms: drift-removed prediction (singles+combos) + A-load-bearing\n',
         'disc_resid/pear_resid = held-out SINGLES drift-removed (chance disc 0.5). '
         'S_/C_real = drift-removed spearman singles/combos. **_Agap = real−max(zero,shuf)** (>0 ⇒ A helps). '
         'probe_A = 1−cos(Δ,Δ_{A=0}) (A used), probe_Dperm≈1 ⇒ decoder dominates, crossTF≈0 ⇒ mode collapse.\n',
         '| arm | prior | disc_resid | pear_resid | S_real | **S_Agap** | C_real | **C_Agap** | C_disc | probe_A | probe_Dperm | crossTF |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda x: (x['prior'], -(x['disc_resid'] or 0))):
        L.append('| ' + ' | '.join([
            r['arm'], r['prior'], fm(r['disc_resid']), fm(r['pear_resid']),
            fm(r['S_real']), fm(r['S_Agap']), fm(r['C_real']), fm(r['C_Agap']), fm(r['C_disc']),
            fm(r['probe_A']), fm(r['probe_Dperm']), fm(r['crossTF'])]) + ' |')
    L.append('\n_exp19 best (linear, prior-on epsmix): disc_resid=0.647. Prior-on CollecTRI AUROC ~0.97 = memorization; '
             'prior-off ~0.57 = corr floor._')
    open(f'{ROOT}/docs/MASTER_ALL.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nsaved -> {ROOT}/results/MASTER_ALL_x20.csv + docs/MASTER_ALL.md')


if __name__ == '__main__':
    main()
