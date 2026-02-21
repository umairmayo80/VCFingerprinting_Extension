"""
utils.py — Shared utilities for vcfp_reproduction.

Covers: seeding, plotting helpers, progress helpers, I/O convenience.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Reproducibility ────────────────────────────────────────────────────────

def seed_everything(seed: int = config.RANDOM_SEED) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Plotting Helpers ───────────────────────────────────────────────────────

# Global style
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = sns.color_palette("muted", 12)


def plot_class_balance(
    label_counts: Dict[str, int],
    save_path: Optional[Path] = None,
    top_n: int = 50,
) -> plt.Figure:
    """Bar chart of samples per class."""
    labels = sorted(label_counts.keys(), key=lambda k: k)[:top_n]
    counts = [label_counts[l] for l in labels]
    short_labels = [l[:20] for l in labels]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(counts)), counts, color=PALETTE[0])
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_ylabel("Sample Count")
    ax.set_title(f"Class Balance (first {top_n} classes)" if top_n < len(label_counts)
                 else "Class Balance")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_trace_length_distribution(
    trace_lengths: List[int],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Histogram of packet counts per trace."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trace_lengths, bins=40, color=PALETTE[1], edgecolor="white")
    ax.set_xlabel("Number of Packets per Trace")
    ax.set_ylabel("Frequency")
    ax.set_title("Trace Length Distribution")
    ax.axvline(np.mean(trace_lengths), color="red", linestyle="--", label=f"Mean={np.mean(trace_lengths):.0f}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_packet_size_distribution(
    sizes_up: List[float],
    sizes_down: List[float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Stacked histogram of packet sizes by direction."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sizes_up, bins=50, alpha=0.6, color=PALETTE[2], label="Upstream (+1)")
    ax.hist(sizes_down, bins=50, alpha=0.6, color=PALETTE[3], label="Downstream (−1)")
    ax.set_xlabel("Packet Size (bytes)")
    ax.set_ylabel("Frequency")
    ax.set_title("Packet Size Distribution by Direction")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_upstream_downstream_scatter(
    up_bytes: List[float],
    down_bytes: List[float],
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter: upstream vs downstream total bytes per trace."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(up_bytes, down_bytes, alpha=0.4, s=15, color=PALETTE[4])
    ax.set_xlabel("Upstream Bytes")
    ax.set_ylabel("Downstream Bytes")
    ax.set_title("Upstream vs Downstream Bytes per Trace")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_burst_distribution(
    burst_sizes: List[float],
    burst_counts: List[int],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Two-panel: burst size histogram + burst count histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.hist(burst_sizes, bins=50, color=PALETTE[5], edgecolor="white")
    ax1.set_xlabel("Burst Size (bytes, signed)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Burst Size Distribution")

    ax2.hist(burst_counts, bins=30, color=PALETTE[6], edgecolor="white")
    ax2.set_xlabel("Number of Bursts per Trace")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Burst Count Distribution")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_trace_duration_distribution(
    durations: List[float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Histogram of trace durations in seconds."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(durations, bins=40, color=PALETTE[7], edgecolor="white")
    ax.set_xlabel("Trace Duration (seconds)")
    ax.set_ylabel("Frequency")
    ax.set_title("Trace Duration Distribution")
    ax.axvline(np.mean(durations), color="red", linestyle="--",
               label=f"Mean={np.mean(durations):.2f}s")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    max_classes: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    If n_classes > max_classes, shows top-N confused classes only.
    """
    n = cm.shape[0]
    if n > max_classes:
        # Select top-max_classes most-confused classes (off-diagonal)
        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)
        top_idx = np.argsort(off_diag.sum(axis=1))[-max_classes:]
        top_idx = np.sort(top_idx)
        cm = cm[np.ix_(top_idx, top_idx)]
        title = f"Confusion Matrix — {model_name} (top {max_classes} confounded)"
    else:
        title = f"Confusion Matrix — {model_name}"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, ax=ax, cmap="Blues", fmt="d", linewidths=0.3)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_model_comparison(
    df,
    metric: str = "Accuracy",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing models on a single metric."""
    # Filter out paper reference rows for the bar chart
    mask = ~df["Model"].str.endswith("[PAPER]")
    subset = df[mask].copy()

    fig, ax = plt.subplots(figsize=(10, max(4, len(subset) * 0.45)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(subset))]
    bars = ax.barh(subset["Model"], subset[metric], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(f"Model Comparison — {metric}")
    ax.invert_yaxis()
    ax.set_xlim(0, subset[metric].max() * 1.15 if subset[metric].max() > 0 else 1)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_norm_dist_comparison(
    df,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart of Normalized Semantic Distance per model.
    Includes a dashed line at 49.5 (random guess baseline).
    """
    mask = ~df["Model"].str.endswith("[PAPER]") & df["Norm Sem Dist"].notna()
    subset = df[mask].copy()

    fig, ax = plt.subplots(figsize=(10, max(4, len(subset) * 0.45)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(subset))]
    bars = ax.barh(subset["Model"], subset["Norm Sem Dist"], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.axvline(49.5, color="red", linestyle="--", linewidth=1.5, label="Random guess (49.5)")
    ax.set_xlabel("Normalized Semantic Distance (lower = better)")
    ax.set_title("Normalized Semantic Distance by Model")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── I/O Helpers ────────────────────────────────────────────────────────────

def ensure_results_dirs() -> None:
    """Create results subdirectories if they don't exist."""
    for d in [config.FIGURES_DIR, config.TABLES_DIR,
              config.SEMANTIC_VECTORS_DIR, config.MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
