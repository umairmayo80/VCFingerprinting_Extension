"""
models.py — Classifier definitions for vcfp_reproduction.

Covers:
  Original paper models:
    (1) JaccardClassifier  — custom similarity-based classifier
    (2) LLNBClassifier     — GaussianNB on LL-NB histogram features
    (3) VNGppClassifier    — GaussianNB on VNG++ features
    (4) PSVMAdaBoost       — AdaBoostClassifier on P-SVM features (paper's fallback)
    (5) PSVMSvc            — SVC(rbf) on P-SVM features (original intent, tuned)

  New traditional ML models:
    (6) RFClassifier       — Random Forest
    (7) XGBClassifier_     — XGBoost
    (8) KNNClassifier      — K-Nearest Neighbours
    (9) LRClassifier       — Logistic Regression

  Deep learning models (PyTorch):
    (10) Conv1DModel       — 1D-CNN on raw packet sequences
    (11) LSTMModel         — LSTM/GRU on raw packet sequences
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.feature_extraction import (
    jaccard_similarity,
    build_class_sets,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Jaccard Classifier
# ═══════════════════════════════════════════════════════════════════════════

class JaccardClassifier:
    """
    Classifies a test trace by finding the training class whose majority-voted
    Jaccard set has the highest Jaccard similarity with the test set.

    Reimplemented from Liberatore & Levine [5] via paper Sec. III-A.
    Not a sklearn estimator — handled separately in training.py.
    """

    def __init__(self):
        self.class_sets_: Optional[Dict[int, Set[int]]] = None

    def fit(self, X_filepaths: List[str], y: List[int]) -> "JaccardClassifier":
        """
        Build representative class sets from training file paths.
        X_filepaths: list of CSV file paths (strings)
        y: integer labels aligned to X_filepaths
        """
        self.class_sets_ = build_class_sets(X_filepaths, y)
        self.classes_ = sorted(self.class_sets_.keys())
        return self

    def predict(self, X_filepaths: List[str]) -> np.ndarray:
        """Predict class for each test trace file."""
        from src.feature_extraction import compute_jaccard_set
        predictions = []
        for fp in X_filepaths:
            test_set = compute_jaccard_set(fp)
            best_class, best_sim = -1, -1.0
            for cls, cls_set in self.class_sets_.items():
                sim = jaccard_similarity(test_set, cls_set)
                if sim > best_sim:
                    best_sim = sim
                    best_class = cls
            predictions.append(best_class)
        return np.array(predictions)


# ═══════════════════════════════════════════════════════════════════════════
# 2-5. Sklearn-based Original Paper Models
# ═══════════════════════════════════════════════════════════════════════════

def make_llnb_model() -> GaussianNB:
    """LL-NB: GaussianNB on LL-NB histogram features."""
    return GaussianNB()


def make_vngpp_model() -> GaussianNB:
    """VNG++: GaussianNB on VNG++ features."""
    return GaussianNB()


def make_psvm_adaboost_model(
    n_estimators: int = config.ADABOOST_N_ESTIMATORS,
    random_state: int = config.RANDOM_SEED,
) -> Pipeline:
    """
    P-SVM (AdaBoost): AdaBoostClassifier with P-SVM features.
    The paper switched to AdaBoost because SVM achieved only 1.2%.
    Uses standard scaler in a pipeline for stability.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            algorithm="SAMME",
        )),
    ])


def make_psvm_svc_model(
    C: float = 10.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    random_state: int = config.RANDOM_SEED,
) -> Pipeline:
    """
    P-SVM (SVM): SVC with RBF kernel on P-SVM features.
    The original study got 1.2%; we attempt with tuned C.
    Uses StandardScaler — critical for SVM performance.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            decision_function_shape="ovo",
        )),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# 6-9. New Traditional ML Models
# ═══════════════════════════════════════════════════════════════════════════

def make_random_forest_model(
    n_estimators: int = config.RF_N_ESTIMATORS,
    random_state: int = config.RANDOM_SEED,
) -> RandomForestClassifier:
    """Random Forest — handles high-dimensional histograms well."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def make_xgboost_model(
    random_state: int = config.RANDOM_SEED,
):
    """XGBoost — gradient boosting for tabular classification."""
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )


def make_knn_model(k: int = config.KNN_K) -> KNeighborsClassifier:
    """K-Nearest Neighbours — non-parametric baseline."""
    return KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)


def make_logistic_regression_model(
    random_state: int = config.RANDOM_SEED,
) -> Pipeline:
    """Logistic Regression — linear baseline, requires scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# 10. 1D-CNN (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

class Conv1DModel(nn.Module):
    """
    1D Convolutional Neural Network operating on raw packet sequences.

    Input: (batch, seq_len, 2)  — each timestep = (size, direction)
    Output: (batch, n_classes)  — logits

    Inspired by deep fingerprinting approaches (Sirinam et al. [13]).
    """

    def __init__(self, n_classes: int, seq_len: int = config.SEQUENCE_PAD_LEN):
        super().__init__()
        self.seq_len = seq_len

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=8, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(0.1),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=8, padding=4),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(0.1),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(0.1),
        )

        # Compute flattened size
        dummy = torch.zeros(1, 2, seq_len)
        with torch.no_grad():
            flat_size = self.conv_block(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 2) → transpose to (batch, 2, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════
# 11. LSTM/GRU (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM on raw packet sequences.

    Input: (batch, seq_len, 2)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_classes: int,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_gru: bool = False,
    ):
        super().__init__()
        rnn_cls = nn.GRU if use_gru else nn.LSTM

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # ×2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 2)
        out, _ = self.rnn(x)          # (batch, seq_len, hidden*2)
        out = out[:, -1, :]           # last timestep
        return self.classifier(out)


# ═══════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════

SKLEARN_MODELS = {
    "LL-NB":              (make_llnb_model,              "llnb"),
    "VNG++":              (make_vngpp_model,             "vngpp"),
    "P-SVM (AdaBoost)":   (make_psvm_adaboost_model,    "psvm"),
    "P-SVM (SVM)":        (make_psvm_svc_model,         "psvm"),
    "Random Forest":      (make_random_forest_model,    "psvm"),
    "XGBoost":            (make_xgboost_model,          "psvm"),
    "KNN":                (make_knn_model,              "psvm"),
    "Logistic Regression":(make_logistic_regression_model, "psvm"),
}

DL_MODELS = {
    "1D-CNN": Conv1DModel,
    "LSTM":   LSTMModel,
    "GRU":    lambda nc: LSTMModel(nc, use_gru=True),
}
