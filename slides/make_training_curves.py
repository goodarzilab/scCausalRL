"""Generate a clean training curves panel for Phase 1 slides."""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'phase1_training.csv')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'training_curves_panel.png')

df = pd.read_csv(LOG_PATH)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('white')

DAG_EPOCH = 20  # DAG activation epoch

panels = [
    ('recon',     'Reconstruction Loss (MSE)',  'tab:blue',   False),
    ('kl',        'KL Divergence',              'tab:orange', False),
    ('pert',      'Perturbation Consistency',   'tab:red',    False),
    ('dag_h',     'DAGMA h(A) (Acyclicity)',    'tab:purple', True),
    ('spectral',  'Spectral Stability',         'tab:green',  False),
    ('sparse',    'Sparsity (L1)',              'tab:brown',  False),
]

for ax, (col, title, color, use_log) in zip(axes.flat, panels):
    if col not in df.columns:
        ax.set_visible(False)
        continue

    vals = df[col].values
    epochs = df['epoch'].values

    ax.plot(epochs, vals, color=color, linewidth=1.8, alpha=0.9)
    ax.axvline(x=DAG_EPOCH, color='grey', ls='--', lw=1.0, alpha=0.6)
    ax.text(DAG_EPOCH + 1, ax.get_ylim()[1] * 0.92, 'DAG\nactivation',
            fontsize=8, color='grey', va='top')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.tick_params(labelsize=9)

    if use_log and vals.max() > 10 * vals[vals > 0].min():
        ax.set_yscale('symlog', linthresh=0.01)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='-')

fig.suptitle('Phase 1: Causal VAE Training (150 epochs, Norman dataset)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUT_PATH}")
