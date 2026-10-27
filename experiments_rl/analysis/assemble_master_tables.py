"""Merge every sweep (grnfix + sena + epsd) × both evals (grn_dag_sweep load-bearing +
component_ablation shift/decoder attribution) into ONE master table for SINGLES and ONE
for COMBOS. Reads config knobs per run so each row is self-describing.

Run AFTER the grn_dag_sweep_* and component_ablation_* CSVs exist:
    python analysis/assemble_master_tables.py
"""
import os, glob, yaml
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIRS = [f'/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose/configs/ablations/{g}'
            for g in ('grn3', 'grn2')]
pd.set_option('display.width', 400); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 120)


def load_concat(pattern, dedup_key):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f'{HERE}/{pattern}')) if os.path.exists(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=dedup_key, keep='first')


def cfg_knobs(wm):
    """Defining hyper-params per run (search grn3 then grn2)."""
    p = next((f'{d}/{wm}.yaml' for d in CFG_DIRS if os.path.exists(f'{d}/{wm}.yaml')), None)
    if p is None:
        return dict(enc='?', eps=np.nan, ortho=np.nan, dagutil=np.nan, shiftL1=np.nan, spA=np.nan, prior=np.nan)
    c = yaml.safe_load(open(p)); m = c.get('model', {}); p1 = c.get('phase1', {})
    return dict(enc='sena' if m.get('encoder_type') == 'sena' else 'mlp',
                eps=p1.get('lambda_eps_decorr', 0.0), ortho=p1.get('lambda_ortho'),
                dagutil=p1.get('pert_dag_util_weight', 0.0), shiftL1=p1.get('pert_shift_l1', 0.0),
                spA=p1.get('lambda_sparse'), prior=p1.get('lambda_prior'))


def main():
    sweep = load_concat('grn_dag_sweep_{grnfix,sena,epsd}.csv'.replace('{grnfix,sena,epsd}', '*'), 'WM')
    # only the grid CSVs (not the base grn_dag_sweep.csv)
    sweep = pd.concat([pd.read_csv(f) for f in [f'{HERE}/grn_dag_sweep_{g}.csv' for g in ('grnfix', 'sena', 'epsd')] if os.path.exists(f)],
                      ignore_index=True).drop_duplicates('WM', keep='first')
    sweep['WM'] = sweep['WM'].str.replace('grnfix_', '')
    abl = pd.concat([pd.read_csv(f) for f in [f'{HERE}/component_ablation_{g}.csv' for g in ('grnfix', 'sena', 'epsd')] if os.path.exists(f)],
                    ignore_index=True).drop_duplicates(['WM', 'split'], keep='first')

    for split, name in [('S', 'SINGLES'), ('C', 'COMBOS')]:
        A = abl[abl.split == split].set_index('WM')
        rows = []
        for wm in sweep['WM']:
            k = cfg_knobs(wm)
            sw = sweep[sweep.WM == wm].iloc[0]
            pre = 's_' if split == 'S' else 'c_'
            r = {'run': wm, **k}
            # ---- load-bearing of A (real vs zero vs shuffled) ----
            r['spear_realA'] = sw[f'{pre}spear_real']; r['spear_zeroA'] = sw[f'{pre}spear_zero']; r['spear_shufA'] = sw[f'{pre}spear_shuf']
            r['pear_realA'] = sw[f'{pre}pear_real']; r['pear_zeroA'] = sw[f'{pre}pear_zero']; r['pear_shufA'] = sw[f'{pre}pear_shuf']
            r['A_gap'] = sw[f'{pre}spear_real'] - max(sw[f'{pre}spear_zero'], sw[f'{pre}spear_shuf'])   # >0 ⇒ A load-bearing
            # ---- component attribution (Δ from baseline; larger ⇒ more load-bearing) ----
            if wm in A.index:
                a = A.loc[wm]
                r['disc_base'] = a['disc']; r['spear_base'] = a['spear']; r['pear_base'] = a['pear']
                for comp in ['shift_perm', 'shift_rand', 'D_perm', 'D_rand']:
                    r[f'dSpear_{comp}'] = a['spear'] - a[f'{comp}.spear']
                    r[f'dDisc_{comp}'] = a['disc'] - a[f'{comp}.disc']
            # ---- GRN memorization + A structure ----
            r['ctri_auroc'] = sw['collectri_auroc']; r['ctri_shufA'] = sw['shuffA']
            r['dens_solve'] = sw['dens_solve']; r['spec_solve'] = sw['spec_solve']
            rows.append(r)
        M = pd.DataFrame(rows)
        M.to_csv(f'{HERE}/master_table_{name.lower()}.csv', index=False)
        # curated view
        cols = ['run', 'enc', 'eps', 'ortho', 'dagutil', 'shiftL1', 'spA', 'prior',
                'spear_realA', 'spear_zeroA', 'spear_shufA', 'A_gap',
                'pear_realA', 'pear_zeroA',
                'dSpear_shift_perm', 'dSpear_shift_rand', 'dSpear_D_perm', 'dSpear_D_rand',
                'dDisc_shift_perm', 'dDisc_D_perm',
                'ctri_auroc', 'ctri_shufA', 'dens_solve', 'spec_solve']
        cols = [c for c in cols if c in M.columns]
        print(f"\n{'='*140}\nMASTER TABLE — {name}  (A_gap>0 ⇒ A load-bearing; dSpear/dDisc = drop when component ablated, larger ⇒ that component carries prediction)\n{'='*140}")
        print(M[cols].round(3).to_string(index=False))
        print(f"saved -> master_table_{name.lower()}.csv")


if __name__ == '__main__':
    main()
