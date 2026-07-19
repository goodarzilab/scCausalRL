"""Shared stack builder so every RL entry point selects the dataset/WM the same way.

Defaults reproduce the original Norman behavior EXACTLY (dataset='norman', the exp14
checkpoint), so existing launches are unchanged. Pass --dataset joung + --checkpoint /
--wm_root to freeze a Joung TF-atlas world model instead.
"""
import torch
from core.wm_loader import load_frozen_wm, DEFAULT_CKPT, DEFAULT_WM_ROOT
from core.wm_operator import WorldModelOperator


def add_stack_args(ap):
    ap.add_argument('--dataset', default='norman', choices=['norman', 'joung'])
    ap.add_argument('--checkpoint', default=None, help='frozen WM checkpoint (.pt)')
    ap.add_argument('--wm_root', default=None, help='experiment root for the WM (sys.path + loaders)')
    ap.add_argument('--combo_path', default=None, help='joung: combo h5ad (default exp17 canonical)')
    ap.add_argument('--singles_path', default=None, help='joung: singles h5ad (default load_joung default)')
    ap.add_argument('--curated_topk', type=int, default=256,
                    help="joung 'curated' pool size: top-K TFs by effect ∪ combo genes")


def build_stack(args, dev, n_dag):
    """Returns (model, meta, op, data). `data` exposes the WMData public API for either
    dataset, so env / MPC / policies / validate are dataset-agnostic downstream."""
    ckpt = args.checkpoint or DEFAULT_CKPT
    root = args.wm_root or DEFAULT_WM_ROOT
    model, meta = load_frozen_wm(checkpoint_path=ckpt, wm_root=root, device=dev)
    op = WorldModelOperator(model, meta, n_dag_samples=n_dag)
    n_ctrl = getattr(args, 'n_ctrl', 500)
    if args.dataset == 'joung':
        from core.wm_data_joung import WMDataJoung
        data = WMDataJoung(model, meta, op, singles_path=args.singles_path,
                           combo_path=args.combo_path, n_ctrl_cells=n_ctrl,
                           curated_topk=getattr(args, 'curated_topk', 256))
    else:
        from core.wm_data import WMData
        data = WMData(model, meta, op, n_ctrl_cells=n_ctrl)
    return model, meta, op, data
