"""Data-grounded validation of a planner.

A planner is any `fn(target_key) -> gene_set`. For each held-out perturbation we
report three complementary, falsifiable measures:

  1. GENE RECOVERY (primary planning metric): did the planner recover the true
     perturbation's gene(s) from its data-grounded induced state? (recall / both-hit)
  2. LATENT DISTANCE: ‖z_reached − z_target‖ vs the control-baseline distance
     (dist_ratio < 1 ⇒ the plan moved us toward the real perturbed state).
  3. GENE-SPACE FIDELITY: decode the chosen set through the (validated mutilation)
     do-operator and score the predicted DE against the OBSERVED DE with the SAME
     scorer the combo eval uses (AUROC / Spearman / Pearson-DEG, targets masked).

Reuses `evaluation.run_norman_combo_eval._score` and the observed DE in WMData, so
numbers are directly comparable to the falsification-gate combo eval.
"""
import numpy as np
import pandas as pd


def _key_str(key):
    return "+".join(key) if isinstance(key, tuple) else str(key)


def evaluate_planner(planner_fn, split_keys, planners, data, label="", effect_clusters=None):
    """Run a planner over split_keys; return (per_target_df, summary_dict).

    Primary metric = `match` (signed Pearson of the reached DE vs observed DE on the
    target's DEGs — the planner's gene-space objective). `recall`/`full_recovery` are
    the (expected-low) gene-identifiability finding; `dist_ratio` is the secondary
    latent-distance view. If `effect_clusters` (an EffectClusters) is given, also report
    perturbation-effect SOFT recall (graded corr + cluster-hit) — credit for picking
    genes with similar effects to the true drivers (no penalty)."""
    from evaluation.run_norman_combo_eval import _score   # WM root already on sys.path
    from core.scoring import de_match
    n_genes = len(data.gene_names)
    rows = []
    for key in split_keys:
        true_gi = planners.true_genes(key)
        chosen = list(planner_fn(key))
        recovered = len(set(chosen) & set(true_gi))
        recall = recovered / max(1, len(true_gi))

        zt = data.target_for(key)
        z_reached = planners.reached_latent(chosen)
        dist = float((z_reached - zt).norm())
        dist_ctrl = float((data.z0 - zt).norm())

        de = data.de_combo[key] if isinstance(key, tuple) else data.de_single[key]
        mask = np.ones(n_genes, bool)
        for gi in true_gi:
            mask[gi] = False
        pred = planners.predict_de(chosen)
        match = de_match(pred, de['lfc'], mask)
        sc = _score(pred, de['lfc'], de['effect'], de['sig'], mask)
        row = {
            'target': _key_str(key), 'k': len(true_gi),
            'chosen': "+".join(data.gene_names[i] for i in chosen),
            'match': match, 'recovered': recovered, 'recall': recall,
            'dist_ratio': dist / (dist_ctrl + 1e-8),
            'auroc': sc.get('auroc', np.nan), 'spearman': sc.get('spearman', np.nan),
            'pearson_deg': sc.get('pearson_deg', np.nan),
        }
        if effect_clusters is not None:
            row.update(effect_clusters.soft_recall(chosen, true_gi))
        rows.append(row)
    df = pd.DataFrame(rows)
    def _m(c):
        return float(df[c].mean(skipna=True)) if len(df) else np.nan
    summary = {
        'label': label, 'n': len(df),
        'match': _m('match'),
        'recall': _m('recall'),
        'full_recovery': float((df['recall'] >= 1.0).mean()) if len(df) else np.nan,
        'dist_ratio': _m('dist_ratio'),
        'auroc': _m('auroc'), 'spearman': _m('spearman'), 'pearson_deg': _m('pearson_deg'),
    }
    if effect_clusters is not None and 'soft_recall_corr' in df:
        summary['soft_recall_corr'] = _m('soft_recall_corr')
        summary['soft_recall_clust'] = _m('soft_recall_clust')
    return df, summary
