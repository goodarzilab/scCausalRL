import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine


def mse_on_degs(y_pred, y_true, de_effects, threshold=0.1):
    """
    MSE computed only on differentially expressed genes.
    
    y_pred: array [n_genes] - predicted expression
    y_true: array [n_genes] - true expression
    de_effects: array [n_genes] - differential expression effect (perturbed - control)
    threshold: float - minimum |effect| to count as DEG
    """
    deg_mask = np.abs(de_effects) > threshold
    if deg_mask.sum() == 0:
        return np.mean((y_pred - y_true) ** 2)
    return np.mean((y_pred[deg_mask] - y_true[deg_mask]) ** 2)


def pearson_correlation(y_pred, y_true):
    """Pearson correlation between predicted and actual values."""
    if len(y_pred) < 2:
        return 0.0
    try:
        r, _ = pearsonr(y_pred.ravel(), y_true.ravel())
        return float(r) if not np.isnan(r) else 0.0
    except Exception:
        return 0.0


def cosine_similarity(y_pred, y_true):
    """Cosine similarity between predicted and actual perturbation effect vectors."""
    try:
        return 1.0 - cosine(y_pred.ravel(), y_true.ravel())
    except Exception:
        return 0.0


def direction_accuracy(pred_effect, true_effect, threshold=0.05):
    """
    Fraction of DEGs where we correctly predict the sign of change.
    """
    deg_mask = np.abs(true_effect) > threshold
    if deg_mask.sum() == 0:
        return 0.0
    correct = np.sign(pred_effect[deg_mask]) == np.sign(true_effect[deg_mask])
    return float(correct.mean())


def pearson_on_degs(pred_effect, true_effect, threshold=0.1):
    """Pearson correlation of predicted vs actual fold changes on DEGs."""
    deg_mask = np.abs(true_effect) > threshold
    if deg_mask.sum() < 2:
        return 0.0
    return pearson_correlation(pred_effect[deg_mask], true_effect[deg_mask])


def compute_mmd(X, Y, gamma=1.0):
    """
    Maximum Mean Discrepancy between two sets of samples.
    Uses RBF kernel.
    X, Y: arrays of shape [n, d]
    """
    def rbf_kernel(A, B, gamma):
        diff = A[:, None, :] - B[None, :, :]  # [n, m, d]
        sq_dist = (diff ** 2).sum(-1)  # [n, m]
        return np.exp(-gamma * sq_dist)
    
    Kxx = rbf_kernel(X, X, gamma).mean()
    Kyy = rbf_kernel(Y, Y, gamma).mean()
    Kxy = rbf_kernel(X, Y, gamma).mean()
    return float(Kxx + Kyy - 2 * Kxy)
