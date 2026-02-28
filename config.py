"""
config.py — Global configuration for vcfp_reproduction
All paths, seeds, and hyperparameters live here.
"""

import os
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── Dataset ────────────────────────────────────────────────────────────────
# Absolute path to the trace_csv directory (original dataset, read-only)
PROJECT_ROOT = Path(__file__).parent.resolve()
ORIGINAL_DATA_ROOT = PROJECT_ROOT / "data"          # VCFingerprinting/data
TRACE_CSV_DIR      = ORIGINAL_DATA_ROOT / "trace_csv"
QUERY_LIST_FILE    = ORIGINAL_DATA_ROOT / "amazon_echo_query_list_100.xlsx"

# ── Results ────────────────────────────────────────────────────────────────
RESULTS_DIR         = PROJECT_ROOT / "results"
FIGURES_DIR         = RESULTS_DIR / "figures"
TABLES_DIR          = RESULTS_DIR / "tables"
SEMANTIC_VECTORS_DIR = RESULTS_DIR / "semantic_vectors"
MODELS_DIR          = RESULTS_DIR / "models"

# Create result dirs on import if missing
for _d in [FIGURES_DIR, TABLES_DIR, SEMANTIC_VECTORS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Cross-Validation ────────────────────────────────────────────────────────
N_FOLDS: int = 5

# ── Feature Extraction Hyperparameters ─────────────────────────────────────
# LL-NB: rounding to nearest multiple (best from paper = 100)
LLNB_ROUNDING_PARAM: int = 100
LLNB_SIZE_RANGE: tuple = (-1500, 1501)

# VNG++: burst rounding (best from paper = 5000)
VNGPP_ROUNDING_PARAM: int = 5000
VNGPP_BURST_RANGE: tuple = (-400_000, 400_001)

# P-SVM: burst rounding 
PSVM_ROUNDING_PARAM: int = 5000
PSVM_BURST_RANGE: tuple = (-200_000, 200_001)

# ── Semantic Evaluation ─────────────────────────────────────────────────────
# Strategy: 'sentence_transformers' (no training, default) or 'doc2vec'
SEMANTIC_STRATEGY: str = "sentence_transformers"
SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

# Doc2Vec params (only used if SEMANTIC_STRATEGY == 'doc2vec')
DOC2VEC_VECTOR_SIZE: int = 100
DOC2VEC_EPOCHS: int = 50
DOC2VEC_WINDOW: int = 15
DOC2VEC_MIN_COUNT: int = 1

# ── PyTorch / DL ────────────────────────────────────────────────────────────
DL_BATCH_SIZE: int = 32
DL_MAX_EPOCHS: int = 100
DL_LR: float = 1e-3
DL_PATIENCE: int = 10            # early stopping patience
SEQUENCE_PAD_LEN: int = 512      # packets per trace (pad/truncate)

# ── Model-specific Hyperparameters ──────────────────────────────────────────
RF_N_ESTIMATORS: int = 300
KNN_K: int = 3
ADABOOST_N_ESTIMATORS: int = 200
