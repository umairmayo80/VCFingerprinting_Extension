"""
training.py — 5-fold StratifiedKFold training and evaluation for vcfp_reproduction.

Supports:
  - Sklearn classifiers (GaussianNB, SVC, AdaBoost, RF, XGBoost, KNN, LR)
  - Custom JaccardClassifier
  - PyTorch DL models (1D-CNN, LSTM/GRU) via a separate training loop

All results are returned as structured dicts for downstream evaluation.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data_loader import load_dataset, load_trace, invert_label_map
from src.feature_extraction import (
    compute_features_batch,
    compute_psvm_feature,
)
from src.models import JaccardClassifier


# ── Seeding ────────────────────────────────────────────────────────────────

def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Sequence Preparation (DL) ─────────────────────────────────────────────

def load_trace_sequence(filepath: str, pad_len: int = config.SEQUENCE_PAD_LEN) -> np.ndarray:
    """
    Load a trace file and return a 2D array of shape (pad_len, 2):
        col 0 = size (normalised to [0,1] by /1500)
        col 1 = direction (+1 or -1)
    Truncated or zero-padded to `pad_len`.
    """
    df = load_trace(filepath)
    seq = np.zeros((pad_len, 2), dtype=np.float32)
    n = min(len(df), pad_len)
    if n > 0:
        seq[:n, 0] = df["size"].values[:n] / 1500.0   # normalise
        seq[:n, 1] = df["direction"].values[:n].astype(np.float32)
    return seq


def build_sequence_tensor(
    filepaths: List[str],
    pad_len: int = config.SEQUENCE_PAD_LEN,
) -> torch.Tensor:
    """Build (N, pad_len, 2) tensor from list of file paths."""
    seqs = np.stack([load_trace_sequence(fp, pad_len) for fp in filepaths])
    return torch.tensor(seqs, dtype=torch.float32)


# ── Sklearn Cross-Validation ───────────────────────────────────────────────

def cross_validate_sklearn(
    model_factory,            # callable() → sklearn estimator
    X: np.ndarray,            # feature matrix
    y: np.ndarray,            # integer labels
    n_folds: int = config.N_FOLDS,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    """
    Standard StratifiedKFold CV for sklearn estimators.

    Returns dict with keys:
        fold_accuracies, mean_accuracy, std_accuracy
        all_true, all_pred  (for metric computation across all fold predictions)
    """
    set_global_seed(seed)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_accs = []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = float(np.mean(preds == y_test))
        fold_accs.append(acc)
        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    return {
        "fold_accuracies": fold_accs,
        "mean_accuracy":   float(np.mean(fold_accs)),
        "std_accuracy":    float(np.std(fold_accs)),
        "all_true":        np.array(all_true),
        "all_pred":        np.array(all_pred),
    }


# ── Jaccard Cross-Validation ───────────────────────────────────────────────

def cross_validate_jaccard(
    filepaths: List[str],
    y: np.ndarray,
    n_folds: int = config.N_FOLDS,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    """
    StratifiedKFold CV for the custom JaccardClassifier.
    X is file paths, not a numerical matrix.
    """
    set_global_seed(seed)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fps = np.array(filepaths)

    fold_accs = []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(fps, y)):
        train_fps = fps[train_idx].tolist()
        test_fps  = fps[test_idx].tolist()
        y_train   = y[train_idx]
        y_test    = y[test_idx]

        clf = JaccardClassifier()
        clf.fit(train_fps, y_train.tolist())
        preds = clf.predict(test_fps)

        acc = float(np.mean(preds == y_test))
        fold_accs.append(acc)
        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    return {
        "fold_accuracies": fold_accs,
        "mean_accuracy":   float(np.mean(fold_accs)),
        "std_accuracy":    float(np.std(fold_accs)),
        "all_true":        np.array(all_true),
        "all_pred":        np.array(all_pred),
    }


# ── PyTorch DL Training Loop ───────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)
    acc = float(np.mean(preds_arr == labels_arr))
    return acc, preds_arr, labels_arr


def cross_validate_dl(
    model_factory,        # callable(n_classes) → nn.Module
    filepaths: List[str],
    y: np.ndarray,
    n_folds: int = config.N_FOLDS,
    seed: int = config.RANDOM_SEED,
    max_epochs: int = config.DL_MAX_EPOCHS,
    batch_size: int = config.DL_BATCH_SIZE,
    lr: float = config.DL_LR,
    patience: int = config.DL_PATIENCE,
    pad_len: int = config.SEQUENCE_PAD_LEN,
) -> Dict:
    """
    StratifiedKFold CV for PyTorch DL models.
    Loads traces as (N, pad_len, 2) sequences.
    Includes early stopping.
    """
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(set(y.tolist()))

    # Pre-load all sequences once (expensive if done per fold)
    print("[training] Loading sequence tensors for DL training...")
    X_all = build_sequence_tensor(filepaths, pad_len)
    y_all = torch.tensor(y, dtype=torch.long)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all.numpy(), y)):
        print(f"  [DL Fold {fold+1}/{n_folds}]", end=" ")
        torch.manual_seed(seed + fold)

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test  = X_all[test_idx]
        y_test  = y_all[test_idx]

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )

        model = model_factory(n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        best_acc = 0.0
        no_improve = 0

        for epoch in range(max_epochs):
            _train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            val_acc, _, _ = _eval_epoch(model, test_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        _, preds, labels = _eval_epoch(model, test_loader, device)
        fold_acc = float(np.mean(preds == labels))
        fold_accs.append(fold_acc)
        all_true.extend(labels.tolist())
        all_pred.extend(preds.tolist())
        print(f"acc={fold_acc:.4f}")

    return {
        "fold_accuracies": fold_accs,
        "mean_accuracy":   float(np.mean(fold_accs)),
        "std_accuracy":    float(np.std(fold_accs)),
        "all_true":        np.array(all_true),
        "all_pred":        np.array(all_pred),
    }
