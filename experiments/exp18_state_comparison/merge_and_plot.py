"""Stage 3: merge per-pert cell-eval results for STATE, exp13, exp16, ContextMean,
PerturbMean on the matched K562 held-out singles → head-to-head table + figure.

exp13 is scored on the 110 perts held out by BOTH exp13 & STATE; exp16 on the 92 held
out by BOTH exp16 & STATE. STATE + baselines are subset to the same perts for each panel,
so every model-vs-STATE comparison is on an identical, fairly-held-out pert set.
"""
import os, json, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = f"{HERE}/results"; FIG = f"{HERE}/results"
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--arcinstitute--ST-SE-Replogle/snapshots/e324967ff4cea5ec199e29bcbb5c1f00e5b9d69c")
_ap = argparse.ArgumentParser()
_ap.add_argument('--regime', default='fewshot', choices=['fewshot', 'zeroshot'])
ARGS = _ap.parse_args()
SUF = '_zs' if ARGS.regime == 'zeroshot' else ''
STATE_RES = f"{SNAP}/{ARGS.regime}/k562/eval_best.ckpt/k562_results.csv"
METRICS = ['pearson_delta', 'de_direction_match', 'de_spearman_lfc_sig',
           'discrimination_score_cosine', 'de_sig_genes_recall', 'overlap_at_100', 'mae_delta']
HUE = '#3C5488'


def load_res(path):
    d = pd.read_csv(path)
    pc = 'perturbation' if 'perturbation' in d.columns else ('gene' if 'gene' in d.columns else d.columns[0])
    return d.rename(columns={pc: 'pert'}).set_index('pert')


def sem(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (float(v.mean()) if len(v) else np.nan, 0.0)


def panel(model_label, model_res_path, fair_perts_file):
    perts = set(json.load(open(f"{R}/{fair_perts_file}")))
    src = {'STATE': load_res(STATE_RES), model_label: load_res(model_res_path),
           'ContextMean': load_res(f"{R}/contextmean{SUF}_celleval/contextmean_results.csv"),
           'PerturbMean': load_res(f"{R}/perturbmean{SUF}_celleval/perturbmean_results.csv")}
    # restrict every method to perts present in the MODEL's results (the actually-scored set)
    scored = perts & set(src[model_label].index)
    rows = []
    for meth, df in src.items():
        sub = df.loc[df.index.isin(scored)]
        row = {'method': meth, 'n': len(sub)}
        for m in METRICS:
            if m in sub.columns:
                mu, se = sem(sub[m]); row[m] = mu; row[m + '_sem'] = se
        rows.append(row)
    return pd.DataFrame(rows), len(scored)


def main():
    out = {}
    for lab, rp, ff in [('exp13', f"{R}/exp13{SUF}_celleval/exp13_results.csv", f'perts_exp13{SUF}.json'),
                        ('exp16', f"{R}/exp16{SUF}_celleval/exp16_results.csv", f'perts_exp16{SUF}.json')]:
        tab, n = panel(lab, rp, ff)
        tab.to_csv(f"{R}/headtohead_{lab}{SUF}.csv", index=False)
        out[lab] = (tab, n)
        print(f"\n===== {lab} vs STATE-{ARGS.regime} (K562 held-out singles, n={n}) =====")
        print(tab[['method', 'n'] + [m for m in METRICS if m in tab.columns]].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # figure: 2 model panels × key metrics, methods grouped
    show = ['pearson_delta', 'de_direction_match', 'discrimination_score_cosine', 'overlap_at_100']
    order = ['STATE', None, 'ContextMean', 'PerturbMean']   # None = the model
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, lab in zip(axes, ['exp13', 'exp16']):
        tab, n = out[lab]
        meths = ['STATE', lab, 'ContextMean', 'PerturbMean']
        x = np.arange(len(show)); w = 0.2
        for i, meth in enumerate(meths):
            r = tab[tab.method == meth]
            if not len(r): continue
            vals = [float(r[m].iloc[0]) if m in r else np.nan for m in show]
            ses = [float(r[m + '_sem'].iloc[0]) if m + '_sem' in r else 0 for m in show]
            hatch = '' if meth in ('STATE', lab) else '//'
            alpha = 1.0 if meth in ('STATE', lab) else 0.6
            ax.bar(x + (i - 1.5) * w, vals, w, yerr=ses, capsize=2, label=meth,
                   color=HUE, alpha=alpha, edgecolor='k', linewidth=0.5, hatch=hatch)
        ax.set_xticks(x); ax.set_xticklabels(show, fontsize=8, rotation=15)
        ax.set_title(f'{lab} vs STATE-{ARGS.regime} — K562 held-out singles (n={n})', fontsize=11)
        ax.set_ylabel('metric'); ax.legend(fontsize=8, frameon=False); ax.set_ylim(0, 1)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=0.25)
    fig.suptitle(f'Replogle K562 single-perturbation prediction ({ARGS.regime}, cell-eval): causal WM vs STATE vs baselines',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIG}/state_comparison{SUF}.png", dpi=200, bbox_inches='tight')
    print(f"\nsaved → {FIG}/state_comparison{SUF}.png")


if __name__ == '__main__':
    main()
