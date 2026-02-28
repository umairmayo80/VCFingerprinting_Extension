"""
feature_extraction.py — All traffic feature pipelines for vcfp_reproduction.

Four feature sets reimplemented from scratch based on the paper:
  1. LL-NB   : Histogram of (size × direction), rounded to LLNB_ROUNDING_PARAM
  2. VNG++   : Burst byte histogram + trace time + upstream/downstream totals
  3. P-SVM   : VNG++ burst histogram + 5 scalar traffic statistics
  4. Jaccard : Set of unique (size × direction) values (used directly for similarity)

"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data_loader import load_trace


# ── Utility: Histogram Binning ─────────────────────────────────────────────

def _round_to_multiple(value: float, multiple: int) -> int:
    """Round `value` to the nearest `multiple`."""
    return int(multiple * round(float(value) / multiple))


def _build_bin_edges(range_min: int, range_max: int, step: int) -> List[int]:
    """Build a list of bin-edge breakpoints from range_min to range_max by step."""
    edges = list(range(range_min, range_max, step))
    return edges


def _histogram(values: List[float], range_min: int, range_max: int, step: int) -> np.ndarray:
    """
    Count how many values fall into each bin [edge_i, edge_{i+1}).
    Values outside the range are clamped to the first/last bin.
    """
    edges = np.arange(range_min, range_max, step)
    counts, _ = np.histogram(values, bins=np.append(edges, range_max))
    return counts.astype(float)


# ── Burst Computation ──────────────────────────────────────────────────────

def _compute_bursts(df: pd.DataFrame) -> List[float]:
    """
    Group consecutive same-direction packets into bursts.
    A burst's value is the sum of (size × direction) for all packets in it.

    Returns a list of burst bytes (signed).
    """
    if df.empty:
        return []

    bursts: List[float] = []
    current_direction = df.iloc[0]["direction"]
    current_burst: float = df.iloc[0]["size"] * df.iloc[0]["direction"]

    for _, row in df.iloc[1:].iterrows():
        if row["direction"] == current_direction:
            current_burst += row["size"] * row["direction"]
        else:
            bursts.append(current_burst)
            current_direction = row["direction"]
            current_burst = row["size"] * row["direction"]

    bursts.append(current_burst)  # last burst
    return bursts


# ── LL-NB Feature Extraction ───────────────────────────────────────────────

def compute_llnb_feature(
    filepath: str,
    rounding_param: int = config.LLNB_ROUNDING_PARAM,
    range_min: int = config.LLNB_SIZE_RANGE[0],
    range_max: int = config.LLNB_SIZE_RANGE[1],
) -> np.ndarray:
    """
    LL-NB features: histogram of (size × direction) values, rounded to
    the nearest multiple of `rounding_param`.

    Reimplemented from paper description (Liberatore & Levine [5] via Kennedy et al.).
    """
    df = load_trace(filepath)
    if df.empty:
        n_bins = math.ceil((range_max - range_min) / rounding_param)
        return np.zeros(n_bins)

    # Signed packet sizes — bin directly WITHOUT rounding 
    signed_sizes = (df["size"] * df["direction"]).tolist()
    return _histogram(signed_sizes, range_min, range_max, rounding_param)


# ── VNG++ Feature Extraction ───────────────────────────────────────────────

def compute_vngpp_feature(
    filepath: str,
    rounding_param: int = config.VNGPP_ROUNDING_PARAM,
    range_min: int = config.VNGPP_BURST_RANGE[0],
    range_max: int = config.VNGPP_BURST_RANGE[1],
) -> np.ndarray:
    """
    VNG++ features (Dyer et al. [8] adapted for VCFP):
        - Histogram of burst bytes (rounded)
        - Total trace time
        - Upstream total bytes
        - Downstream total bytes

    Reimplemented from paper description.
    """
    df = load_trace(filepath)
    if df.empty:
        n_bins = math.ceil((range_max - range_min) / rounding_param)
        return np.zeros(n_bins + 3)

    # Burst bytes 
    bursts = _compute_bursts(df)
    burst_hist = _histogram(bursts, range_min, range_max, rounding_param)

    # Scalar features
    total_trace_time = float(df["time"].max() - df["time"].min())
    upstream   = float((df.loc[df["direction"] == 1, "size"]).sum())
    downstream = float((df.loc[df["direction"] == -1, "size"]).sum())

    scalars = np.array([total_trace_time, upstream, downstream])
    return np.concatenate([burst_hist, scalars])


# ── P-SVM Feature Extraction ───────────────────────────────────────────────

def compute_psvm_feature(
    filepath: str,
    rounding_param: int = config.PSVM_ROUNDING_PARAM,
    range_min: int = config.PSVM_BURST_RANGE[0],
    range_max: int = config.PSVM_BURST_RANGE[1],
) -> np.ndarray:
    """
    P-SVM features (Panchenko et al. [7] adapted for VCFP):
        - Burst byte histogram (same as VNG++)
        - Upstream total bytes
        - Downstream total bytes
        - Incoming packet ratio (downstream / total)
        - Total packet count
        - Burst count
    HTML/packet-size-52 markers omitted as noted in paper Sec. III-A.

    Reimplemented from paper description.
    """
    df = load_trace(filepath)
    if df.empty:
        n_bins = math.ceil((range_max - range_min) / rounding_param)
        return np.zeros(n_bins + 5)

    # Burst bytes
    bursts = _compute_bursts(df)
    burst_hist = _histogram(bursts, range_min, range_max, rounding_param)

    total_packets = len(df)
    up_packets   = int((df["direction"] == 1).sum())
    down_packets = int((df["direction"] == -1).sum())

    upstream   = float((df.loc[df["direction"] == 1, "size"]).sum())
    downstream = float((df.loc[df["direction"] == -1, "size"]).sum())
    incoming_ratio = down_packets / total_packets if total_packets > 0 else 0.0
    burst_count = float(len(bursts))

    scalars = np.array([upstream, downstream, incoming_ratio, float(total_packets), burst_count])
    return np.concatenate([burst_hist, scalars])


# ── Jaccard Set Extraction ─────────────────────────────────────────────────

def compute_jaccard_set(filepath: str) -> Set[int]:
    """
    Jaccard representation: set of unique signed packet sizes.
    Each element = int(size × direction).

    Reimplemented from paper description (Liberatore & Levine [5]).
    """
    df = load_trace(filepath)
    if df.empty:
        return set()
    return set((df["size"] * df["direction"]).astype(int).tolist())


def jaccard_similarity(set_a: Set[int], set_b: Set[int]) -> float:
    """Jaccard similarity J = |A∩B| / |A∪B|. Returns 0 for empty sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def build_class_sets(filepath_list: List[str], label_list: List[int]) -> Dict[int, Set[int]]:
    """
    Build representative Jaccard sets for each class from training data.
    An element is included in the class set if it appears in at least
    ceil(n/2) training traces (majority vote), matching original approach.
    """
    from collections import defaultdict

    # Group filepaths by label
    class_files: Dict[int, List[str]] = defaultdict(list)
    for fp, lbl in zip(filepath_list, label_list):
        class_files[lbl].append(fp)

    class_sets: Dict[int, Set[int]] = {}
    for lbl, fps in class_files.items():
        if len(fps) == 1:
            class_sets[lbl] = compute_jaccard_set(fps[0])
        else:
            threshold = math.ceil(len(fps) / 2)
            # Collect all individual sets
            all_sets = [compute_jaccard_set(fp) for fp in fps]
            # Count how many sets each element appears in
            element_counts: Dict[int, int] = defaultdict(int)
            for s in all_sets:
                for elem in s:
                    element_counts[elem] += 1
            class_sets[lbl] = {
                elem for elem, cnt in element_counts.items() if cnt >= threshold
            }

    return class_sets


# ── Batch Feature Computation ──────────────────────────────────────────────

def compute_features_batch(
    filepaths: List[str],
    method: str,
    **kwargs,
) -> np.ndarray:
    """
    Compute feature matrix for a list of trace files.

    Args:
        filepaths: list of CSV file paths
        method: one of 'llnb', 'vngpp', 'psvm'

    Returns: np.ndarray of shape (n_samples, n_features)
    """
    dispatch = {
        "llnb":  compute_llnb_feature,
        "vngpp": compute_vngpp_feature,
        "psvm":  compute_psvm_feature,
    }
    if method not in dispatch:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(dispatch)}")

    fn = dispatch[method]
    features = [fn(fp, **kwargs) for fp in filepaths]
    return np.array(features, dtype=np.float64)


# ── Sanity Check ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader import load_dataset

    fps, lbls, lmap = load_dataset()
    sample = fps[:3]

    print("=== LL-NB Features ===")
    f = compute_llnb_feature(sample[0])
    print(f"  shape={f.shape}, non-zero bins={np.count_nonzero(f)}")

    print("=== VNG++ Features ===")
    f = compute_vngpp_feature(sample[0])
    print(f"  shape={f.shape}, non-zero bins={np.count_nonzero(f)}")

    print("=== P-SVM Features ===")
    f = compute_psvm_feature(sample[0])
    print(f"  shape={f.shape}, non-zero bins={np.count_nonzero(f)}")

    print("=== Jaccard Set ===")
    s = compute_jaccard_set(sample[0])
    print(f"  |set|={len(s)}, sample elements={list(s)[:5]}")
