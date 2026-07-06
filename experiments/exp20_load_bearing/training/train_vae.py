import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _init_wandb(config, run_name_key='phase1_run_name'):
    """Initialise wandb if enabled in config. Returns the run object or None."""
    wcfg = config.get('wandb', {})
    if not wcfg.get('enabled', False):
        return None
    try:
        import wandb
        run = wandb.init(
            project=wcfg.get('project', 'CausalBioRL'),
            entity=wcfg.get('entity', None),
            name=wcfg.get(run_name_key, None),
            config={
                'data': config['data'],
                'model': config['model'],
                'phase1': config['phase1'],
            },
            reinit=True,
        )
        print(f"wandb run: {run.name}  ({run.url})")
        return run
    except ImportError:
        print("wandb not installed — skipping W&B logging.")
        return None
    except Exception as e:
        print(f"wandb init failed ({e}) — continuing without W&B.")
        return None

from models.causal_vae import CausalVAE
from training.losses import (
    reconstruction_loss, kl_divergence, perturbation_consistency_loss,
    perturbation_consistency_loss_v2, perturbation_set_loss_u,
    perturbation_set_loss_shift, perturbation_set_loss_shift_mutilate,
    compute_total_loss
)
from utils.bio_priors import load_collectri
from data.cell_sentence_dataset import PairedCellSentenceDataset


def make_interventional_dataset(data_dict, gene_split=None):
    """
    Build (x_ctrl, gene_idx, x_pert, dx, pert_modality_id) tuples from the data dict.

    Supports both single-modality datasets (Norman, Replogle) and combined.
    For combined datasets, single_perts keys are namespaced like 'GENE__norman' or
    'GENE__replogle' — we extract the bare gene name for gene_to_idx lookup and
    use the dataset suffix to determine the perturbed-cell modality.

    gene_split: set of gene names (un-namespaced) to include, or None for all.
                Used to restrict to train-split genes during training.
    dx is the precomputed gene-space perturbation effect from de_single.
    Returns: list of (ctrl_X, gene_idx, pert_X, dx, pert_modality_id, ctrl_modality_id) tuples
    """
    adata = data_dict['adata']
    control_idx = data_dict['control_idx']
    single_perts = data_dict['single_perts']
    gene_names = data_dict['gene_names']
    de_single = adata.uns.get('de_single', {})

    import scipy.sparse
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    X_ctrl = X[control_idx]

    # Modality lookup (per-cell), if available
    has_modality = 'modality_id' in adata.obs.columns
    if has_modality:
        modality_col = adata.obs['modality_id'].values.astype(np.int64)
        # Mapping for namespaced keys: 'norman' → CRISPRa (1), 'replogle' → CRISPRi (2)
        suffix_to_modality = {'norman': 1, 'replogle': 2}
        ctrl_modalities = modality_col[control_idx]
    else:
        ctrl_modalities = np.zeros(len(control_idx), dtype=np.int64)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    triples = []
    for key, pert_idx_list in single_perts.items():
        # Parse namespaced key: 'GENE__dataset' → (gene, dataset)
        if '__' in key:
            gene, dataset_suffix = key.split('__', 1)
            pert_modality_id = suffix_to_modality.get(dataset_suffix, 0) if has_modality else 0
        else:
            gene = key
            # Single-modality dataset: read modality from cell labels (or default 0)
            pert_modality_id = int(modality_col[pert_idx_list[0]]) if has_modality and pert_idx_list else 0

        if gene not in gene_to_idx:
            continue
        if gene_split is not None and gene not in gene_split:
            continue

        gene_expr_idx = gene_to_idx[gene]
        X_pert = X[pert_idx_list]
        # Precomputed dx from de_single — use namespaced key for combined datasets
        dx = de_single.get(key, de_single.get(gene, None))
        if dx is None:
            dx = X_pert.mean(axis=0) - X_ctrl.mean(axis=0)
        dx = np.asarray(dx, dtype=np.float32)
        triples.append((X_ctrl, gene_expr_idx, X_pert, dx, pert_modality_id, ctrl_modalities))

    return triples


def sample_interventional_batch(triples, batch_size=64, rng=None):
    """
    Sample a batch for a single randomly-chosen gene perturbation.
    Returns (x_ctrl, gene_idx_scalar, x_pert, dx, pert_modality, ctrl_modality_batch) or None.
    pert_modality is a scalar int; ctrl_modality_batch is a numpy array [batch_size].
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(triples) == 0:
        return None

    # Pick one gene at random
    triple_idx = rng.integers(len(triples))
    triple = triples[triple_idx]
    # Backward compat: old triples had 4 elements; new have 6
    if len(triple) == 4:
        ctrl_X, gene_idx, pert_X, dx = triple
        pert_modality_id = 0
        ctrl_modalities = np.zeros(len(ctrl_X), dtype=np.int64)
    else:
        ctrl_X, gene_idx, pert_X, dx, pert_modality_id, ctrl_modalities = triple

    n_ctrl = len(ctrl_X)
    n_pert = len(pert_X)
    if n_ctrl == 0 or n_pert == 0:
        return None

    n = min(batch_size, n_ctrl, n_pert)
    ctrl_sample = rng.choice(n_ctrl, n, replace=False)
    pert_sample = rng.choice(n_pert, n, replace=False)

    return (ctrl_X[ctrl_sample], gene_idx, pert_X[pert_sample], dx,
            pert_modality_id, ctrl_modalities[ctrl_sample])


def train_vae(config, data_dict=None, device=None):
    """
    Phase 1: Warm-start causal VAE training.

    config: dict from configs/default.yaml
    data_dict: pre-loaded dataset (if None, loads from config)
    device: torch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data if not provided
    if data_dict is None:
        dataset_name = config['data']['dataset']
        if dataset_name == 'norman':
            from data.load_norman import load_norman
            data_dict = load_norman(
                path=config['data']['norman_path'],
                n_hvg=config['data']['n_hvg'],
                train_frac=config['data'].get('single_pert_train_frac', 0.8),
                val_frac=config['data'].get('single_pert_val_frac', 0.1),
                split_seed=config['data'].get('split_seed', 42),
            )
        elif dataset_name == 'joung':
            from data.load_joung import load_joung
            data_dict = load_joung(
                proc_path=config['data'].get('joung_proc_path'),
                train_frac=config['data'].get('single_pert_train_frac', 0.8),
                val_frac=config['data'].get('single_pert_val_frac', 0.1),
                split_seed=config['data'].get('split_seed', 42),
                min_cells_per_pert=config['data'].get('min_cells_per_pert', 5),
                load_combos=bool(config['data'].get('load_combos', False)),
                combo_proc_path=config['data'].get('joung_combo_proc_path'),
                min_combo_cells=int(config['data'].get('min_combo_cells', 20)),
            )
        elif dataset_name == 'combined':
            from data.load_combined import load_combined
            data_dict = load_combined(
                norman_path=config['data']['norman_path'],
                replogle_path=config['data']['replogle_path'],
                n_hvg=config['data']['n_hvg'],
                replogle_max_cells=config['data'].get('replogle_max_cells', 100000),
                train_frac=config['data'].get('single_pert_train_frac', 0.8),
                val_frac=config['data'].get('single_pert_val_frac', 0.1),
                split_seed=config['data'].get('split_seed', 42),
            )
        else:
            from data.load_replogle import load_replogle
            # Optionally union the Replogle HVG list with Norman's perturbed gene set
            # so that Norman external eval has full coverage. Off by default.
            force_include = None
            if config['data'].get('force_include_norman_perts', False):
                from data.norman_perturbed_genes import get_norman_perturbed_genes
                force_include = get_norman_perturbed_genes(
                    path=config['data'].get('norman_path'),
                )
                print(f"  force_include_norman_perts: requesting {len(force_include)} "
                      f"Norman perturbed genes in Replogle HVG set")
            data_dict = load_replogle(
                path=config['data']['replogle_path'],
                n_hvg=config['data']['n_hvg'],
                cell_line=config['data'].get('replogle_cell_line', 'k562'),
                max_cells=config['data'].get('replogle_max_cells', 50000),
                train_frac=config['data'].get('single_pert_train_frac', 0.8),
                val_frac=config['data'].get('single_pert_val_frac', 0.1),
                split_seed=config['data'].get('split_seed', 42),
                force_include=force_include,
            )

    adata = data_dict['adata']
    gene_names = data_dict['gene_names']
    n_genes = len(gene_names)

    # Load expression matrix
    import scipy.sparse
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)

    # Modality IDs (per cell, 0=control, 1=CRISPRa, 2=CRISPRi)
    # For single-modality datasets (Norman, Replogle), all cells get modality 0 — encoder works as before
    if 'modality_id' in adata.obs.columns:
        modality_ids = adata.obs['modality_id'].values.astype(np.int64)
        n_modalities = int(data_dict.get('n_modalities', 3))
        print(f"Multi-modality dataset: {n_modalities} modalities")
    else:
        modality_ids = np.zeros(adata.n_obs, dtype=np.int64)
        n_modalities = 0  # disables modality embedding in encoder

    # Batch IDs (gem_group) for additive batch-effect conditioning in the encoder.
    # Replogle K562 has 48 gem_groups; without this, the encoder mixes plate effects
    # into ε and the perturbation loss is contaminated by them.
    if 'batch_id' in adata.obs.columns:
        batch_ids_all = adata.obs['batch_id'].values.astype(np.int64)
        n_batches = int(data_dict.get('n_batches', int(batch_ids_all.max()) + 1))
        print(f"Batch-effect conditioning: {n_batches} batches (gem_groups)")
    else:
        batch_ids_all = np.zeros(adata.n_obs, dtype=np.int64)
        n_batches = 0

    # Hold out combo perturbations (they are NEVER in training)
    # Only use control + single perturbation cells for training
    control_idx = data_dict['control_idx']
    single_perts = data_dict['single_perts']

    # Training cell indices: control + single pert cells
    train_idx = list(control_idx)
    for idxs in single_perts.values():
        train_idx.extend(idxs)
    train_idx = np.array(sorted(train_idx))

    X_train = X[train_idx]
    modality_train = modality_ids[train_idx]
    batch_train = batch_ids_all[train_idx]
    print(f"Training cells: {len(X_train)} (control: {len(control_idx)}, single perts: {len(train_idx) - len(control_idx)})")

    # Build interventional triples (train-split genes only)
    pert_split = data_dict.get('pert_split', None)
    train_genes = pert_split['train'] if pert_split else None
    triples = make_interventional_dataset(data_dict, gene_split=train_genes)
    if pert_split:
        print(f"Interventional triples: {len(triples)} (train-split only, "
              f"{len(pert_split['train'])} train / {len(pert_split['val'])} val / "
              f"{len(pert_split['test'])} test genes)")
    else:
        print(f"Interventional triples (gene perturbations): {len(triples)}")

    # Drift = shared across-TF mean observed Δ over TRAIN-split genes (the common
    # "differentiation" program). Training on the drift-removed residual (pert_residual
    # _target) focuses the shift encoder on the TF-SPECIFIC effect — feature_probe.py
    # showed the generalizable held-out signal lives entirely in this residual.
    _de = data_dict.get('de_single', {})
    _drift_genes = [g for g in (train_genes or _de.keys()) if g in _de]
    delta_drift_np = (np.mean([np.asarray(_de[g], np.float32) for g in _drift_genes], axis=0)
                      if _drift_genes else None)
    if delta_drift_np is not None:
        print(f"Δ-drift computed over {len(_drift_genes)} train TFs "
              f"(||drift||={np.linalg.norm(delta_drift_np):.3f}); "
              f"residual_target={config['phase1'].get('pert_residual_target', False)}")
    delta_drift_t = (torch.tensor(delta_drift_np, dtype=torch.float32, device=device)
                     if delta_drift_np is not None else None)

    # exp20 Phase-2 mode-collapse fix: per-gene std of the drift-removed target Δ over train TFs.
    # Whitening the MSE by this makes the loss reward getting the residual DIRECTION right rather
    # than its magnitude → starves the single-scalar shift_scale shortcut that collapses all TFs.
    pert_whiten_t = None
    if bool(config['phase1'].get('pert_whiten_target', False)) and _drift_genes and delta_drift_np is not None:
        _res = np.stack([np.asarray(_de[g], np.float32) - delta_drift_np for g in _drift_genes])  # [T, G]
        _w = np.maximum(_res.std(0), 1e-3).astype(np.float32)
        pert_whiten_t = torch.tensor(_w, dtype=torch.float32, device=device)
        print(f"pert whiten target: per-gene residual std (median={np.median(_w):.4f}, "
              f"min={_w.min():.4f}, max={_w.max():.4f})")

    # Model parameters
    p1 = config['phase1']
    m = config['model']

    intv_cfg = config.get('intervention', {}) or {}
    use_shift_encoder = bool(m.get('use_shift_encoder', False))

    # ── Optional SENA-δ encoder: build the GO:BP gene→pathway mask on this gene axis ──
    # GO:BP membership (NOT CollecTRI) anchors latent programs to named processes,
    # breaking the encoder/decoder rotational symmetry so A is load-bearing. The mask
    # ships in the checkpoint (persistent buffer); CollecTRI stays eval-only.
    encoder_type = m.get('encoder_type', 'mlp')
    sena_mask = None
    if encoder_type == 'sena':
        from utils.bio_priors import build_go_bp_mask
        sena_mask, sena_terms = build_go_bp_mask(
            gene_names,
            max_term_size=int(m.get('sena_max_term_size', 30)),
            min_genes_in_axis=int(m.get('sena_min_genes', 5)),
            max_overlap=float(m.get('sena_max_overlap', 0.5)),
            leak=float(m.get('sena_leak', 0.1)),
            max_pathways=m.get('sena_max_pathways', None))
        sena_mask = torch.as_tensor(sena_mask, dtype=torch.float32)

    # ── Optional FROZEN decoder (exp20 Phase-2, "SENA for the decoder") ──────────────
    # D = [GO:BP membership cols | PCA-of-expression cols], unit-normed, requires_grad=False.
    # GO cols give interpretability; PCA cols give full gene coverage (GO alone covers ~26%).
    # Freezing D removes the absorption freedom so A must carry propagation. latent_dim is
    # OVERRIDDEN to the built width and written back to config so the eval loader rebuilds it.
    frozen_decoder_weight = None
    if m.get('decoder_type') == 'frozen_go':
        from utils.bio_priors import build_go_bp_mask
        Mgo, _terms = build_go_bp_mask(
            gene_names, max_term_size=int(m.get('frozen_go_max_term', 30)),
            min_genes_in_axis=int(m.get('frozen_go_min_genes', 5)),
            max_overlap=float(m.get('frozen_go_max_overlap', 0.5)),
            leak=0.0, max_pathways=int(m.get('frozen_go_max_pathways', 192)), verbose=True)
        Mgo = torch.as_tensor(Mgo, dtype=torch.float32)                         # [G, Kgo]
        npca = int(m.get('frozen_go_n_pca', 64))
        rsel = np.random.default_rng(0).choice(X.shape[0], min(20000, X.shape[0]), replace=False)
        Xs = X[rsel]
        Xs = (Xs.toarray() if hasattr(Xs, 'toarray') else np.asarray(Xs)).astype(np.float32)
        Xs = Xs - Xs.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
        Dpca = torch.as_tensor(Vt[:npca].T.copy(), dtype=torch.float32)         # [G, npca]
        Dfz = torch.cat([Mgo, Dpca], dim=1)
        Dfz = Dfz / (Dfz.norm(dim=0, keepdim=True) + 1e-8)                      # unit columns
        frozen_decoder_weight = Dfz
        config['model']['latent_dim'] = int(Dfz.shape[1])
        m = config['model']
        print(f"[frozen_go] decoder D = [GO {Mgo.shape[1]} | PCA {npca}] frozen → "
              f"latent_dim={Dfz.shape[1]} (covers {(Mgo.sum(1) > 0).sum()}/{n_genes} genes via GO)")

    model = CausalVAE(
        n_genes=n_genes,
        latent_dim=m['latent_dim'],
        tau_init=m['tau_init'],
        n_dag_samples=m['n_dag_samples'],
        dagma_s=m['dagma_s'],
        n_modalities=n_modalities,
        n_batches=n_batches,
        u_clip_max=float(intv_cfg.get('u_clip_max', 1.0)),
        decoder_type=m.get('decoder_type', 'linear'),
        decoder_hidden=int(m.get('decoder_hidden', 1024)),
        decoder_layers=int(m.get('decoder_layers', 2)),
        use_shift_encoder=use_shift_encoder,
        shift_label_dim=int(m.get('shift_label_dim', 50)),
        shift_hidden=int(m.get('shift_hidden', 1024)),
        shift_n_layers=int(m.get('shift_n_layers', 1)),
        shift_output_activation=m.get('shift_output_activation', 'none'),
        shift_scale_init=float(m.get('shift_scale_init', 1.0)),
        separate_pert_decoder=bool(m.get('separate_pert_decoder', False)),
        pert_decoder_type=m.get('pert_decoder_type', 'linear'),
        pert_decoder_hidden=int(m.get('pert_decoder_hidden', 1024)),
        pert_decoder_layers=int(m.get('pert_decoder_layers', 2)),
        u_gate_mlp=bool(m.get('u_gate_mlp', False)),
        nonlinear_eps_mix=bool(m.get('nonlinear_eps_mix', False)),
        gate_hidden=int(m.get('gate_hidden', 256)),
        u_gate_pure=bool(m.get('u_gate_pure', False)),
        dag_disabled=bool(m.get('dag_disabled', False)),
        encoder_type=encoder_type,
        sena_mask=sena_mask,
        sena_activation=m.get('sena_activation', 'tanh'),
        pert_intervene_mode=m.get('pert_intervene_mode', 'legacy'),
        use_target_gate=bool(m.get('use_target_gate', False)),
        target_gate_tau=float(m.get('target_gate_tau', 1.0)),
        target_gate_bias_init=float(m.get('target_gate_bias_init', 3.0)),
        frozen_decoder_weight=frozen_decoder_weight,
        scm_nonlinear=bool(m.get('scm_nonlinear', False)),
        n_picard=int(m.get("scm_picard_iters", 40)),
        scm_solve_ridge=float(m.get("scm_solve_ridge", 0.0)),
    ).to(device)

    print(f"Model: CausalVAE(n_genes={n_genes}, latent_dim={m['latent_dim']}, "
          f"decoder={m.get('decoder_type', 'linear')}, shift_encoder={use_shift_encoder})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Perturbation labels for the shift encoder (SCCVAE-style) ──────────────
    # Top-PCA of the *transposed* control expression matrix: a per-gene embedding
    # derived purely from control co-expression, so it exists for held-out /
    # unseen genes (the zero-shot generalisation engine). Computed once and
    # stored in the model's pert_label buffer (ships with the checkpoint).
    if use_shift_encoder:
        # Per-gene shift-encoder descriptor. 'pca' = control co-expression PCA (baseline,
        # info-free for silent TFs → discrimination collapse); 'pca+regulon'/'pca+esm' add
        # TF-identity signal that generalises zero-shot. Width must == shift_label_dim.
        from utils.pert_descriptors import build_pert_descriptor
        descriptor = m.get('shift_descriptor', 'pca')
        desc = build_pert_descriptor(X, control_idx, gene_names, m, config, descriptor=descriptor)
        if desc.shape != tuple(model.pert_label.shape):
            raise ValueError(f"descriptor '{descriptor}' width {desc.shape[1]} != shift_label_dim "
                             f"{model.pert_label.shape[1]} — set model.shift_label_dim accordingly")
        model.set_pert_labels(torch.as_tensor(desc, dtype=torch.float32))
        print(f"Shift encoder: descriptor='{descriptor}' shape {tuple(model.pert_label.shape)}")

    # ── Fate labels for the identifiability regularizer (Joung NNLS annotation) ──
    # Per-gene integer fate id (Cao-mouse developmental fate), aligned to gene_names;
    # -1 = unannotated. Only built when pert_ident_weight>0. The SupCon-by-fate term in
    # the loss pulls same-fate TF shifts together / different-fate apart → structures the
    # latent along the identifiable developmental axis.
    fate_labels_t = None
    if float(p1.get('pert_ident_weight', 0.0)) > 0:
        import pandas as pd
        ann_csv = ('/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/'
                   'analysis/mouse_dev_annotation/joung_tf_developmental_annotation.csv')
        ann = pd.read_csv(ann_csv)
        fate_of = dict(zip(ann['TF'].astype(str), ann['main_fate'].astype(str)))
        cats = sorted({f for f in fate_of.values() if f and f != 'nan'})
        cat_id = {c: i for i, c in enumerate(cats)}
        fl = np.full(len(gene_names), -1, dtype=np.int64)
        n_ann = 0
        for gi, g in enumerate(gene_names):
            f = fate_of.get(str(g))
            if f and f in cat_id:
                fl[gi] = cat_id[f]; n_ann += 1
        fate_labels_t = torch.tensor(fl, dtype=torch.long, device=device)
        print(f"Identifiability reg: ident_weight={p1.get('pert_ident_weight')} "
              f"({n_ann}/{len(gene_names)} genes annotated, {len(cats)} fates)")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=p1['lr'])
    warmup_epochs = p1.get('lr_warmup_epochs', 5)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    # Load CollecTRI for biological prior
    try:
        collectri_net = load_collectri()
        print(f"Loaded CollecTRI: {len(collectri_net)} TF-target pairs")
    except Exception as e:
        print(f"CollecTRI not available ({e}), skipping prior loss")
        collectri_net = None

    # DAGMA augmented Lagrangian state
    lambda_dagma = float(p1['dagma_lambda_init'])
    rho_dagma = float(p1['dagma_rho_init'])
    rho_max = float(p1['dagma_rho_max'])

    # Training schedule
    n_epochs = p1['n_epochs']
    recon_phase_epochs = p1['reconstruction_phase_epochs']  # 20
    dag_start_epoch = recon_phase_epochs + 1  # 21

    # Logging
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    log_path = 'logs/phase1_training.csv'

    rng = np.random.default_rng(42)

    # Wandb
    wandb_run = _init_wandb(config, run_name_key='phase1_run_name')

    # DataLoader for observational data (cell, modality_id, batch_id per row).
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    modality_train_tensor = torch.tensor(modality_train, dtype=torch.long)
    batch_train_tensor = torch.tensor(batch_train, dtype=torch.long)
    obs_dataset = TensorDataset(X_train_tensor, modality_train_tensor, batch_train_tensor)
    # Speed lever: reconstruct over a fresh random subsample of cells each epoch instead of
    # all 1.14M (the dominant per-epoch cost). recon_subsample=0 → full pass (original).
    recon_sub = int(p1.get('recon_subsample', 0))
    if recon_sub and recon_sub < len(obs_dataset):
        from torch.utils.data import RandomSampler
        obs_sampler = RandomSampler(obs_dataset, replacement=False, num_samples=recon_sub)
        obs_loader = DataLoader(obs_dataset, batch_size=p1['batch_size'], sampler=obs_sampler, drop_last=False)
        print(f"[speed] reconstruction subsample: {recon_sub}/{len(obs_dataset)} cells/epoch (fresh each epoch)")
    else:
        obs_loader = DataLoader(obs_dataset, batch_size=p1['batch_size'], shuffle=True, drop_last=False)

    # Cell-sentence DataLoader for the perturbation-consistency loss.
    # Each item is a (ctrl_set, pert_set) of S cells paired within the same gem_group.
    # This replaces the random ctrl-pert pairing that contaminated previous grid runs.
    pert_scheme = (config.get('intervention') or {}).get('scheme', 'threshold')
    # Both the u-score loss and the shift-encoder loss operate on paired cell
    # sentences, so either enables the sentence loader.
    use_cell_sentences = (pert_scheme == 'u_score') or use_shift_encoder
    if use_cell_sentences:
        sentence_S = int(p1.get('cell_sentence_S', 64))
        n_per_block = int(p1.get('n_sentences_per_block', 4))
        min_block_cells = int(p1.get('min_block_cells', 8))
        # ── Combo training (Tier-3): include a TRAIN split of combos in the
        # cell-sentence loader so the mutilation operator can LEARN epistasis.
        # Held-out combos (combo_test) are evaluated separately. Deterministic
        # per-combo hash split (independent of order / gene set).
        combo_perts_all = data_dict.get('combo_perts', {}) or {}
        combo_train_keys, combo_test_keys = set(), set()
        if combo_perts_all and float(p1.get('combo_train_frac', 0.0)) > 0:
            holdout = p1.get('combo_holdout_keys')
            if holdout:
                # FIXED holdout: train on every combo EXCEPT these (used to honestly hold out
                # the RL test pairs, since the RL split is independent of the hash split).
                ho = {tuple(sorted(k)) for k in holdout}
                for key in combo_perts_all:
                    (combo_test_keys if tuple(sorted(key)) in ho else combo_train_keys).add(key)
                print(f"Combo FIXED-HOLDOUT split: {len(combo_train_keys)} train / "
                      f"{len(combo_test_keys)} test (held out {len(ho)} given pairs); {len(combo_perts_all)} total")
            else:
                import hashlib
                frac = float(p1.get('combo_train_frac', 0.7))
                sseed = int(config['data'].get('split_seed', 42))
                for key in combo_perts_all:
                    hh = int(hashlib.md5(f"{sseed}|{key[0]}|{key[1]}".encode()).hexdigest(), 16) % 1000 / 1000.0
                    (combo_train_keys if hh < frac else combo_test_keys).add(key)
                print(f"Combo split: {len(combo_train_keys)} train / {len(combo_test_keys)} test "
                      f"combos (frac={frac}); {len(combo_perts_all)} total")
            # stash split into config so it travels with every checkpoint (eval filters held-out)
            config.setdefault('phase1', {})['combo_test_keys'] = [list(k) for k in sorted(combo_test_keys)]
            config['phase1']['combo_train_keys'] = [list(k) for k in sorted(combo_train_keys)]
        try:
            sentence_dataset = PairedCellSentenceDataset(
                adata=adata,
                single_perts=single_perts,
                control_idx=control_idx,
                gene_names=gene_names,
                S=sentence_S,
                n_sentences_per_block=n_per_block,
                min_block_cells=min_block_cells,
                gene_filter=(pert_split['train'] if pert_split else None),
                seed=int(config['data'].get('split_seed', 42)),
                modality_id_lookup=modality_ids if n_modalities > 0 else None,
                combo_perts=combo_perts_all if combo_train_keys else None,
                combo_filter=combo_train_keys if combo_train_keys else None,
            )
            sentence_loader = DataLoader(
                sentence_dataset,
                batch_size=int(p1.get('sentence_batch_size', 8)),
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
            sentence_iter = None  # initialised lazily inside the epoch loop
            print(f"Cell-sentence loader: {len(sentence_dataset)} sentences/epoch, "
                  f"batch={p1.get('sentence_batch_size', 8)}, S={sentence_S}")
        except RuntimeError as e:
            # Phase F: if u_score was explicitly requested, silently falling back to
            # the threshold path is a footgun (see grid jobs 2357403 / 2358314).
            # Hard-fail instead so the user sees the failure immediately.
            raise RuntimeError(
                f"intervention.scheme == 'u_score' but the cell-sentence loader "
                f"could not be built: {e}. Most likely cause: "
                f"min_block_cells is too high for the configured replogle_max_cells. "
                f"Lower it in configs/*.yaml or increase replogle_max_cells. "
                f"To force the legacy path, set intervention.scheme == 'threshold'."
            ) from e
    else:
        sentence_loader = None

    best_loss_recon = float('inf')  # Phase A (epochs 1-recon_phase_epochs)
    best_loss_dag = float('inf')    # Phase C (epochs 51+), used by Phase 2
    best_epoch = 0

    print(f"\nStarting Phase 1 training for {n_epochs} epochs...")
    print(f"  Phase A (epochs 1-{recon_phase_epochs}): Reconstruction only, DAG frozen")
    print(f"  Phase B (epochs {dag_start_epoch}-50): DAG learning activated, full loss")
    print(f"  Phase C (epochs 51-150): Full loss with temperature annealing")

    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'total_loss', 'recon', 'kl', 'dag_h', 'sparse', 'spectral', 'ortho', 'pert', 'prior', 'd_sparse', 'h_val', 'tau', 'lambda_dagma', 'rho'])

    # theta-warmin: when scheme=u_score we run Phase A through the SCM solve with
    # theta FROZEN, so the encoder/decoder learn to operate on z = (I-A^T)^-1 ε
    # before theta starts moving.  Eliminates the recon explosion seen at the
    # Phase A→B transition in the legacy bypass_scm pipeline (slurm job 2349145).
    theta_frozen_phase_a = use_cell_sentences

    # Per-module grad norms captured at the last optimization step of each epoch
    # (filled in the step block before zero_grad; read by the W&B logger).
    last_grad_norms = {}

    for epoch in range(1, n_epochs + 1):
        model.train()

        # Refresh decoder-loading matrix L (used by u-score intervention scheme,
        # harmless for the threshold path). Cheap: O(n_genes · d), once per epoch.
        try:
            model.compute_loadings()
        except Exception as _e:
            pass

        # Determine phase parameters
        dag_active = epoch >= dag_start_epoch

        # Freeze theta during Phase A (u-score path only): SCM solve happens but
        # theta does not update.  Unfreeze at dag_start_epoch.
        if theta_frozen_phase_a:
            model.dag.theta.requires_grad_(dag_active)

        # Beta annealing: 0.01 -> 1.0 over first 50 epochs (from dag_start onward)
        if epoch < dag_start_epoch:
            beta = p1['beta_start']
        else:
            progress = (epoch - dag_start_epoch) / max(p1['beta_anneal_epochs'], 1)
            beta = p1['beta_start'] + (p1['beta_end'] - p1['beta_start']) * min(progress, 1.0)

        # Temperature annealing: 1.0 -> 0.25 over beta_anneal_epochs from DAG start
        # (proposal: τ annealed 1.0 → 0.25 starting at DAG activation)
        tau_val = float(config['model']['tau_init'])
        if epoch >= dag_start_epoch:
            progress = (epoch - dag_start_epoch) / max(p1['beta_anneal_epochs'], 1)
            tau_val = max(config['model']['tau_min'], tau_val - min(progress, 1.0) * (tau_val - config['model']['tau_min']))
        model.set_tau(tau_val)

        # Fine-tuning phase: lower learning rate
        if epoch == 101:
            for pg in optimizer.param_groups:
                pg['lr'] = p1['lr'] * 0.1

        epoch_losses = {k: 0.0 for k in ['total', 'recon', 'kl', 'dag', 'dag_lagrangian', 'sparse', 'spectral', 'ortho', 'pert', 'prior', 'd_sparse']}
        n_batches = 0
        accumulation_steps = p1.get('accumulation_steps', 1)

        # DAG loss warm-in: ramp from 0→1 over dag_warmin_epochs after activation
        dag_warmin_epochs = p1.get('dag_warmin_epochs', 10)
        if not dag_active:
            dag_ramp = 0.0
        elif epoch < dag_start_epoch + dag_warmin_epochs:
            dag_ramp = (epoch - dag_start_epoch + 1) / dag_warmin_epochs
        else:
            dag_ramp = 1.0

        # ── Three-stage L0 gate schedule (DECOUPLED from dag_ramp) ──────────────────────
        # Phase A (recon) → Phase B1 (A learns structure with the target gate held OPEN,
        # L0=0, so A becomes competent on the DENSE shift) → Phase B2 (ramp L0 to sparsify
        # the gate NOW that A can carry the propagation). Prevents the gate-vs-A race where
        # L0 closes gates before A can absorb the redirected signal (esm collapse).
        # Defaults (start=dag_start, warmin=dag_warmin) reproduce the coupled behaviour.
        l0_start = int(p1.get('pert_target_l0_start_epoch', dag_start_epoch))
        l0_warmin = max(int(p1.get('pert_target_l0_warmin', dag_warmin_epochs)), 1)
        if epoch < l0_start:
            l0_ramp = 0.0
        elif epoch < l0_start + l0_warmin:
            l0_ramp = (epoch - l0_start + 1) / l0_warmin
        else:
            l0_ramp = 1.0

        optimizer.zero_grad()

        # Reset cell-sentence iterator each epoch (it is consumed in sync with obs batches)
        if use_cell_sentences:
            sentence_iter = iter(sentence_loader)
        else:
            sentence_iter = None

        for batch_idx, (x_batch, mod_batch, batch_batch) in enumerate(obs_loader):
            x_batch = x_batch.to(device)
            mod_batch = mod_batch.to(device)
            batch_batch = batch_batch.to(device)
            mod_arg = mod_batch if n_modalities > 0 else None
            batch_arg = batch_batch if n_batches > 0 else None

            # Forward pass.  Two regimes:
            #   - u-score (use_cell_sentences=True): always run the SCM solve.  In
            #     Phase A theta is frozen, so the encoder/decoder adapt to the SCM
            #     coupling at A=σ(-3)≈4.7% density without theta drift.
            #   - legacy threshold: keep the original bypass_scm + scm_alpha behaviour.
            if use_cell_sentences:
                scm_alpha = None  # always full SCM
                bypass_scm_flag = False
            else:
                scm_alpha = dag_ramp if (dag_active and dag_ramp < 1.0) else None
                bypass_scm_flag = not dag_active
            out = model(x_batch, bypass_scm=bypass_scm_flag, scm_alpha=scm_alpha,
                        modality_id=mod_arg, batch_id=batch_arg)
            x_hat, mu, log_var, epsilon, z, A = (
                out['x_hat'], out['mu'], out['log_var'],
                out['epsilon'], out['z'], out['A']
            )

            # ── Perturbation-consistency loss ───────────────────────────────
            pert_loss_val = None
            pert_diag = None
            if dag_active:
                if use_cell_sentences:
                    # Set-level u-score loss on paired cell sentences.
                    try:
                        sent = next(sentence_iter)
                    except StopIteration:
                        sentence_iter = iter(sentence_loader)
                        sent = next(sentence_iter)
                    x_ctrl_set = sent['x_ctrl'].to(device)         # [B, S, n_genes]
                    x_pert_set = sent['x_pert'].to(device)         # [B, S, n_genes]
                    sent_gene = sent['gene_idx'].to(device)         # [B]
                    sent_batch = sent['batch_id'].to(device)        # [B]
                    sent_mod = sent['modality_id'].to(device) if n_modalities > 0 else None
                    # Build per-sentence pert_direction tensor if the dataset
                    # exposes per-cell perturbation direction (e.g. combined Norman+Replogle).
                    # Otherwise default to -1 (CRISPRi-style) — config can override.
                    default_dir = int(p1.get('pert_direction', -1))
                    pert_dir_batch = torch.full((sent_gene.shape[0],), default_dir,
                                                dtype=torch.long, device=device)
                    # If the sentence dataset attaches a 'pert_direction' column, use it
                    if 'pert_direction' in sent:
                        pert_dir_batch = sent['pert_direction'].to(device).long()
                    if use_shift_encoder and p1.get('pert_loss_mode', 'shift') == 'shift_mutilate':
                        # Tier-2: shift defines the latent target, mutilation (do-operator)
                        # propagates it. u_mode='loading' clamps the perturbed gene's
                        # programs; 'shift' clamps the programs the shift drives.
                        pert_loss_val, pert_diag = perturbation_set_loss_shift_mutilate(
                            x_ctrl_set, x_pert_set,
                            gene_idx=sent_gene,
                            batch_id=sent_batch if n_batches > 0 else torch.zeros_like(sent_batch),
                            modality_id=sent_mod,
                            model=model,
                            n_dag_samples=p1.get('pert_n_dag_samples', 1),
                            mse_weight=float(p1.get('pert_mse_weight', 1.0)),
                            mmd_weight=float(p1.get('pert_mmd_weight', 1.0)),
                            energy_weight=float(p1.get('pert_energy_weight', 0.0)),
                            u_mode=p1.get('pert_mutilate_u_mode', 'loading'),
                            u_clip_max=float(p1.get('pert_mutilate_u_max', 0.5)),
                            shift_beta=float(p1.get('pert_mutilate_beta', 1.0)),
                            shift_l1_weight=float(p1.get('pert_shift_l1', 0.0)),
                            contrastive_weight=float(p1.get('pert_contrastive_weight', 0.0)),
                            contrastive_temp=float(p1.get('pert_contrastive_temp', 0.2)),
                            drift=delta_drift_t,
                            residual_target=bool(p1.get('pert_residual_target', False)),
                            fate_labels=fate_labels_t,
                            ident_weight=float(p1.get('pert_ident_weight', 0.0)),
                            ident_temp=float(p1.get('pert_ident_temp', 0.2)),
                            dag_util_weight=float(p1.get('pert_dag_util_weight', 0.0)),
                            dag_util_margin=float(p1.get('pert_dag_util_margin', 0.0)),
                            target_l0_weight=float(p1.get('pert_target_l0', 0.0)) * l0_ramp,
                            target_whiten=pert_whiten_t,
                        )
                    elif use_shift_encoder:
                        # Learned-shift set loss (MSE on mean shift + MMD [+ energy]).
                        pert_loss_val, pert_diag = perturbation_set_loss_shift(
                            x_ctrl_set, x_pert_set,
                            gene_idx=sent_gene,
                            batch_id=sent_batch if n_batches > 0 else torch.zeros_like(sent_batch),
                            modality_id=sent_mod,
                            model=model,
                            n_dag_samples=p1.get('pert_n_dag_samples', 1),
                            mse_weight=float(p1.get('pert_mse_weight', 1.0)),
                            mmd_weight=float(p1.get('pert_mmd_weight', 1.0)),
                            energy_weight=float(p1.get('pert_energy_weight', 0.0)),
                            cos_weight=float(p1.get('pert_cos_weight', 0.0)),
                            shift_l1_weight=float(p1.get('pert_shift_l1', 0.0)),
                        )
                    else:
                        pert_loss_val, pert_diag = perturbation_set_loss_u(
                            x_ctrl_set, x_pert_set,
                            gene_idx=sent_gene,
                            batch_id=sent_batch if n_batches > 0 else torch.zeros_like(sent_batch),
                            modality_id=sent_mod,
                            model=model,
                            n_dag_samples=p1.get('pert_n_dag_samples', 8),
                            loss_kind=p1.get('pert_loss_kind', 'energy'),
                            match_delta=p1.get('pert_match_delta', True),  # Phase H default
                            input_mode=p1.get('input_mode', 'observed_full'),
                            pert_direction_batch=pert_dir_batch,
                            canonical_magnitude=float(p1.get('canonical_magnitude', 2.0)),
                            aux_prop_weight=float(p1.get('aux_prop_weight', 0.0)),
                        )
                elif len(triples) > 0:
                    # Legacy threshold-mutilation path.  Kept for the
                    # intervention.scheme=threshold ablation only.
                    pert_batch = sample_interventional_batch(triples, batch_size=64, rng=rng)
                    if pert_batch is not None:
                        x_ctrl_np, gene_idx, x_pert_np, dx_np, pert_mod_id, ctrl_mod_np = pert_batch
                        x_ctrl_t = torch.tensor(x_ctrl_np, dtype=torch.float32, device=device)
                        x_pert_t = torch.tensor(x_pert_np, dtype=torch.float32, device=device)
                        dx_t = torch.tensor(dx_np, dtype=torch.float32, device=device)
                        if n_modalities > 0:
                            ctrl_mod_t = torch.tensor(ctrl_mod_np, dtype=torch.long, device=device)
                            pert_mod_t = torch.full((x_pert_t.shape[0],), pert_mod_id,
                                                    dtype=torch.long, device=device)
                        else:
                            ctrl_mod_t = None
                            pert_mod_t = None
                        pert_loss_val, pert_diag = perturbation_consistency_loss_v2(
                            x_ctrl_t, x_pert_t, dx_t, model,
                            n_dag_samples=p1.get('pert_n_dag_samples', 8),
                            threshold_frac=p1.get('pert_mutilate_threshold_frac', 0.1),
                            ctrl_modality_id=ctrl_mod_t,
                            pert_modality_id=pert_mod_t,
                        )

            # Prior loss
            prior_loss_val = None
            if dag_active and collectri_net is not None and p1.get('lambda_prior', 0) > 0:
                try:
                    from training.losses import biological_prior_loss
                    prior_loss_val = biological_prior_loss(
                        model, gene_names, collectri_net,
                        margin=p1.get('prior_margin', 0.1),
                    )
                except Exception:
                    pass

            # Compute total loss
            losses = compute_total_loss(
                x=x_batch,
                x_hat=x_hat,
                mu=mu,
                log_var=log_var,
                A=A,
                model=model,
                beta=beta,
                lambda_pert=p1['lambda_pert'] * dag_ramp,
                lambda_dag=dag_ramp,
                lambda_sparse=p1['lambda_sparse'] * dag_ramp,
                lambda_sparse_mode=p1.get('lambda_sparse_mode', 'product'),
                lambda_spectral=p1['lambda_spectral'] * dag_ramp,
                lambda_ortho=p1['lambda_ortho'],
                lambda_prior=p1.get('lambda_prior', 0.1) * dag_ramp,
                lambda_dagma=lambda_dagma,
                rho_dagma=rho_dagma,
                lambda_D_sparse=p1.get('lambda_D_sparse', 0.0) * dag_ramp,
                pert_loss=pert_loss_val,
                prior_loss=prior_loss_val,
                dag_active=dag_active,
                lambda_recon=float(p1.get('lambda_recon', 1.0)),
                residual_recon=bool(p1.get('residual_recon', False)),
                drift=delta_drift_t,
                kl_free_bits=float(p1.get('kl_free_bits', 0.0)),
            )

            # ── ε-decorrelation loss (make A load-bearing) ──────────────────────
            # Penalise the off-diagonal of the batch covariance of the exogenous noise ε.
            # With an orthonormal decoder + decorrelated ε, program-program correlation can
            # ONLY be produced by A (Cov(z)=(I−Aᵀ)⁻¹Cov(ε)(I−A)⁻¹ ⇒ A becomes the Cholesky
            # factor of the latent precision). Closes the "route correlation through the
            # aggregate posterior of ε instead of A" escape hatch that KL does not block.
            lam_eps = float(p1.get('lambda_eps_decorr', 0.0))
            if lam_eps > 0 and dag_active and epsilon.shape[0] > 2:
                e = epsilon - epsilon.mean(0, keepdim=True)          # center → covariance
                cov = (e.T @ e) / (e.shape[0] - 1)                   # [d, d]
                eps_decorr = (cov - torch.diag(torch.diagonal(cov))).pow(2).sum()
                losses['total'] = losses['total'] + lam_eps * dag_ramp * eps_decorr
                losses['eps_decorr'] = eps_decorr.detach()

            # NaN guard: skip a non-finite batch so a transient blowup (e.g. an ill-conditioned
            # SCM solve at DAG-activation for large latent_dim) can't poison ALL weights via
            # backward → NaN grads → NaN params. Drop the batch, zero grads, continue.
            if not torch.isfinite(losses['total']):
                optimizer.zero_grad()
                continue

            # Gradient accumulation: scale loss and accumulate
            scaled_loss = losses['total'] / accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(obs_loader):
                # Capture per-module grad norms BEFORE clip/zero. The old code read
                # p.grad in the once-per-epoch W&B block, which runs AFTER the final
                # optimizer.zero_grad() (set_to_none=True) → every grad is None →
                # grad_norm/* logged as 0. Snapshot here (overwrites each step, so it
                # holds the last optimization step's pre-clip norms).
                if wandb_run is not None:
                    _modules = [('encoder', model.encoder), ('decoder', model.decoder),
                                ('dag', model.dag)]
                    if getattr(model, 'shift_encoder', None) is not None:
                        _modules.append(('shift_encoder', model.shift_encoder))
                    for _nm, _mod in _modules:
                        _tot = 0.0
                        for _p in _mod.parameters():
                            if _p.grad is not None:
                                _tot += _p.grad.data.norm(2).item() ** 2
                        last_grad_norms[f'grad_norm/{_nm}'] = _tot ** 0.5
                    if getattr(model, 'shift_scale', None) is not None and model.shift_scale.grad is not None:
                        last_grad_norms['grad_norm/shift_scale'] = model.shift_scale.grad.norm(2).item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Track unscaled loss values for logging
            for k in epoch_losses:
                v = losses.get(k, 0.0)
                if isinstance(v, torch.Tensor):
                    v = v.item()
                epoch_losses[k] += v
            n_batches += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(epoch_losses['total'])

        # DAGMA Lagrangian update every 20 epochs
        if dag_active and epoch % 20 == 0:
            with torch.no_grad():
                A_eval = model.dag.get_edge_probs()
                h_val = model.dag.dagma_constraint(A_eval).item()
            if h_val > 1e-4:
                lambda_dagma = lambda_dagma + rho_dagma * h_val
                rho_dagma = min(rho_dagma * 2.0, rho_max)
        else:
            h_val = epoch_losses.get('dag', 0.0)

        # Save best model (phase-aware)
        _ckpt_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_losses['total'],
            'config': config,
            'gene_names': gene_names,
            'edge_probs': model.dag.get_edge_probs().detach().cpu().numpy(),
            'pert_split': {k: list(v) for k, v in pert_split.items()} if pert_split else None,
        }
        if epoch <= recon_phase_epochs:
            # Phase A: track best recon-only model
            if epoch_losses['total'] < best_loss_recon:
                best_loss_recon = epoch_losses['total']
                torch.save(_ckpt_data, 'checkpoints/phase1_best_recon.pt')
        elif epoch > recon_phase_epochs:
            # DAG-active phase: the checkpoint Phase 2 uses
            if epoch_losses['total'] < best_loss_dag:
                best_loss_dag = epoch_losses['total']
                best_epoch = epoch
                torch.save(_ckpt_data, 'checkpoints/phase1_best.pt')

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'gene_names': gene_names,
            }, f'checkpoints/phase1_epoch{epoch}.pt')

        # Log
        with open(log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch,
                f"{epoch_losses['total']:.4f}",
                f"{epoch_losses['recon']:.4f}",
                f"{epoch_losses['kl']:.4f}",
                f"{epoch_losses.get('dag', 0):.4f}",
                f"{epoch_losses.get('sparse', 0):.4f}",
                f"{epoch_losses.get('spectral', 0):.4f}",
                f"{epoch_losses.get('ortho', 0):.4f}",
                f"{epoch_losses.get('pert', 0):.4f}",
                f"{epoch_losses.get('prior', 0):.4f}",
                f"{epoch_losses.get('d_sparse', 0):.6f}",
                f"{h_val:.6f}",
                f"{tau_val:.4f}",
                f"{lambda_dagma:.4f}",
                f"{rho_dagma:.2f}",
            ])

        # Wandb logging (every epoch)
        if wandb_run is not None:
            log_dict = {
                'epoch': epoch,
                'loss/total': epoch_losses['total'],
                'loss/recon': epoch_losses['recon'],
                'loss/kl': epoch_losses['kl'],
                'loss/pert': epoch_losses.get('pert', 0.0),
                'loss/dag_h': epoch_losses.get('dag', 0.0),
                'loss/sparse': epoch_losses.get('sparse', 0.0),
                'loss/spectral': epoch_losses.get('spectral', 0.0),
                'loss/ortho': epoch_losses.get('ortho', 0.0),
                'loss/prior': epoch_losses.get('prior', 0.0),
                'loss/d_sparse': epoch_losses.get('d_sparse', 0.0),
                'dag/h_val': h_val,
                'dag/lambda_dagma': lambda_dagma,
                'dag/rho_dagma': rho_dagma,
                'dag/dag_ramp': dag_ramp,
                'train/tau': tau_val,
                'train/beta': beta,
                'train/lr': optimizer.param_groups[0]['lr'],
            }
            # Per-module gradient norms — captured in the step block BEFORE
            # optimizer.zero_grad() (reading p.grad here would always be None/0,
            # since the epoch's final step zeroes grads before this block runs).
            log_dict.update(last_grad_norms)
            # DAG edge statistics and z-scale monitoring
            with torch.no_grad():
                edge_probs = model.dag.get_edge_probs()
                log_dict['dag/edge_density'] = edge_probs.mean().item()
                log_dict['dag/max_edge_prob'] = edge_probs.max().item()
                log_dict['dag/theta_std'] = model.dag.theta.std().item()
                log_dict['dag/theta_mean'] = model.dag.theta.mean().item()
                # z-scale: check if SCM is amplifying epsilon
                log_dict['z/z_std'] = out['z'].std().item()
                log_dict['z/z_max'] = out['z'].abs().max().item()
                log_dict['z/epsilon_std'] = out['epsilon'].std().item()
            # Perturbation diagnostics (from last batch's pert_diag).
            # New factored pert loss returns 'loss', 'mean_dz_norm', 'mean_u_l1'.
            # Legacy threshold pert loss returned 'mse', 'mean_affected'. Support both.
            if pert_diag is not None:
                # Loss value: 'loss' (new) or 'mse' (legacy)
                pert_loss_val = pert_diag.get('loss', pert_diag.get('mse'))
                if pert_loss_val is not None:
                    log_dict['pert_diag/loss'] = pert_loss_val
                # Auxiliary diagnostics — only log keys that exist
                for k_in, k_out in [
                    ('mean_affected',   'pert_diag/mean_affected_programs'),  # legacy
                    ('mean_dz_norm',    'pert_diag/mean_dz_norm'),
                    ('mean_u_l1',       'pert_diag/mean_u_l1'),
                    ('match_delta',     'pert_diag/match_delta'),
                    # shift-encoder loss diagnostics
                    ('mse_term',        'pert_diag/mse_term'),
                    ('mmd_term',        'pert_diag/mmd_term'),
                    ('energy_term',     'pert_diag/energy_term'),
                    ('shift_l1_term',   'pert_diag/shift_l1_term'),
                    ('mean_shift_norm', 'pert_diag/mean_shift_norm'),
                    ('shift_scale',     'pert_diag/shift_scale'),
                ]:
                    if k_in in pert_diag:
                        v = pert_diag[k_in]
                        # match_delta is bool; cast for wandb
                        log_dict[k_out] = float(v) if isinstance(v, bool) else v
            wandb_run.log(log_dict, step=epoch)

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            dag_str = f"dag_h={epoch_losses.get('dag', 0):.4f}, " if dag_active else ""
            pert_str = f"pert={epoch_losses.get('pert', 0):.4f}, " if dag_active else ""
            l0_str = (f"dag_ramp={dag_ramp:.2f}, l0_ramp={l0_ramp:.2f}, " if dag_active else "")
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"total={epoch_losses['total']:.4f}, "
                f"recon={epoch_losses['recon']:.4f}, "
                f"kl={epoch_losses['kl']:.4f}, "
                f"{dag_str}{pert_str}{l0_str}"
                f"tau={tau_val:.3f}"
            )

    print(f"\nPhase 1 complete. Best DAG loss: {best_loss_dag:.4f} at epoch {best_epoch}")
    print(f"Best recon-only loss: {best_loss_recon:.4f}")
    print(f"Checkpoint saved to checkpoints/phase1_best.pt")

    if wandb_run is not None:
        wandb_run.summary['best_loss_dag'] = best_loss_dag
        wandb_run.summary['best_loss_recon'] = best_loss_recon
        wandb_run.summary['best_epoch'] = best_epoch
        wandb_run.finish()

    return model, data_dict
