"""Merge the cached baseline+MPC ladder with the per-method PPO policy evals into one
final value-add table (held-out combos). Avoids re-running the slow MPC per method.

Reads:
  rl0_mpc/results/ladder_combo.csv          (random, greedy1, additive, MPC)
  rl1_seq_ppo/results/seqppo_test.csv       (per-target → summarized as 'seq_PPO')
  rl2_set_ppo/results/setppo_test.csv       (→ 'set_PPO')
  rl3_mpc_distill/results/distill_test.csv  (→ 'set_PPO_distill')
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = ['match', 'recall', 'full_recovery', 'dist_ratio', 'auroc', 'spearman', 'pearson_deg']
POLICY_CSVS = [
    ('seq_PPO', 'rl1_seq_ppo/results/seqppo_test.csv'),
    ('set_PPO', 'rl2_set_ppo/results/setppo_test.csv'),
    ('set_PPO_distill', 'rl3_mpc_distill/results/distill_test.csv'),
]


def _summarize(label, df):
    row = {'label': label, 'n': len(df)}
    for c in COLS:
        if c == 'full_recovery':
            row[c] = float((df['recall'] >= 1.0).mean()) if 'recall' in df else np.nan
        else:
            row[c] = float(df[c].mean(skipna=True)) if c in df.columns else np.nan
    return row


def main():
    base_path = os.path.join(ROOT, 'rl0_mpc/results/ladder_combo.csv')
    rows = []
    if os.path.exists(base_path):
        rows = pd.read_csv(base_path).to_dict('records')
    for label, rel in POLICY_CSVS:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            rows.append(_summarize(label, pd.read_csv(p)))
        else:
            print(f"[merge] (missing {rel} — skipping {label})")
    df = pd.DataFrame(rows)[['label', 'n'] + COLS]
    pd.set_option('display.width', 170)
    print("\n===== FINAL VALUE-ADD LADDER (held-out Norman combos, gene-space DE-match) =====")
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    out = os.path.join(ROOT, 'rl0_mpc/results/ladder_final.csv')
    df.to_csv(out, index=False)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
