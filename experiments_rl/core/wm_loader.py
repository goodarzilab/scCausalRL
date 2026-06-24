"""Load + freeze the causal-VAE world model for the RL/planning experiments.

Reuses the experiment's own `load_shift_model` (rebuilds the model from the
checkpoint's saved config and restores the `pert_label` buffer), then freezes every
parameter. The shift-native mutilation hyperparameters (u_mode / u_clip_max /
shift_beta) are auto-detected from the checkpoint config so the do-operator is used
EXACTLY as the model was trained — coherence. (exp14 epoch150 was trained with
pert_loss_mode='shift_mutilate', u_mode='shift', u_max=0.5, beta=1.0.)

Default frozen WM = exp14 (Norman combo-trained shift+mutilation). To swap to the
exp16 multi-cell-line model later, pass `wm_root`/`checkpoint_path` for that run.
"""
import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_WM_ROOT = os.path.join(REPO_ROOT, "experiments", "exp14_norman_combotrain")
DEFAULT_CKPT = os.path.join(
    DEFAULT_WM_ROOT, "results", "grid", "exp14_norman_combotrain",
    "checkpoints", "phase1_epoch150.pt")


def load_frozen_wm(checkpoint_path=DEFAULT_CKPT, wm_root=DEFAULT_WM_ROOT, device=None):
    """Returns (model, meta). model is in eval mode with requires_grad=False.

    meta keys: gene_names, gene_to_idx, latent_dim, mut (u_mode/u_clip_max/
    shift_beta/pert_loss_mode), combo_train_keys, combo_test_keys, norman_path,
    checkpoint, device.
    """
    device = (torch.device(device) if device
              else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if wm_root not in sys.path:
        sys.path.insert(0, wm_root)
    from evaluation.run_shift_eval_norman import load_shift_model  # noqa: E402

    model, gene_names, ckpt = load_shift_model(checkpoint_path, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    cfg = ckpt.get('config', {}) or {}
    p1 = cfg.get('phase1', {}) or {}
    mut = dict(
        pert_loss_mode=p1.get('pert_loss_mode', 'shift'),
        u_mode=p1.get('pert_mutilate_u_mode', 'shift'),
        u_clip_max=float(p1.get('pert_mutilate_u_max', 0.5)),
        shift_beta=float(p1.get('pert_mutilate_beta', 1.0)),
    )
    meta = dict(
        gene_names=list(gene_names),
        gene_to_idx={g: i for i, g in enumerate(gene_names)},
        latent_dim=int(model.latent_dim),
        mut=mut,
        combo_train_keys=[tuple(k) for k in p1.get('combo_train_keys', [])],
        combo_test_keys=[tuple(k) for k in p1.get('combo_test_keys', [])],
        norman_path=(cfg.get('data', {}) or {}).get('norman_path'),
        checkpoint=checkpoint_path,
        device=device,
    )
    print(f"[wm_loader] frozen WM: {len(gene_names)} genes, d={meta['latent_dim']}, "
          f"mut={mut['pert_loss_mode']}/u_mode={mut['u_mode']}/u_max={mut['u_clip_max']}, "
          f"combos train/test={len(meta['combo_train_keys'])}/{len(meta['combo_test_keys'])}")
    return model, meta
