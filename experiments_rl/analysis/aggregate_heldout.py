"""Aggregate canonical heldout_zeroshot_eval summaries (hz_grnfix_*) into one table —
directly comparable to the exp17 prior-Joung numbers (same script, same held-out split)."""
import json, glob, os, pandas as pd
R='/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose/results/heldout_eval'
rows=[]
for f in sorted(glob.glob(f'{R}/hz_grnfix_*/summary.json')):
    d=json.load(open(f)); flat={}
    def w(o):
        if isinstance(o,dict):
            for k,v in o.items():
                if isinstance(v,dict): w(v)
                else: flat[k]=v
    w(d)
    tag=os.path.basename(os.path.dirname(f)).replace('hz_grnfix_','')
    keys=['pearson_delta','pearson_delta_resid','discrimination_cos','discrimination_cos_resid',
          'retrieval_top1','retrieval_top1_resid','deg_auroc_resid','effect_spearman_resid','fate_top1','fate_betacos']
    rows.append({'run':tag, **{k:flat.get(k) for k in keys}})
df=pd.DataFrame(rows)
df.to_csv(f'{os.path.dirname(os.path.abspath(__file__))}/heldout_canonical_all.csv',index=False)
pd.set_option('display.width',260); pd.set_option('display.max_rows',60)
print(df.round(3).to_string(index=False))
print("\nchance: discrimination_cos=0.5 | retrieval_top1=1/T≈0.006 | deg_auroc=0.5")
print("saved -> analysis/heldout_canonical_all.csv")
