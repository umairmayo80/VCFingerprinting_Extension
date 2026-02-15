"""
data_loader.py — Load and parse the VCFP trace_csv dataset.

Each CSV file has columns:
    <index>, time, size, direction
where direction: +1 = speaker→cloud, -1 = cloud→speaker.
Label is extracted from the filename via regex.
"""

import os
import re
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Label Extraction ───────────────────────────────────────────────────────

# Pattern: capture the command name prefix before _<digit>
_LABEL_PATTERN = re.compile(r"^([a-zA-Z'_]+)_\d.*\.csv$")


def extract_label(filepath: str) -> str:
    """
    Extract the voice command label from a trace CSV filename.

    Example:
        'how_many_days_until_christmas_5_30s_L_1.csv'  -> 'how_many_days_until_christmas'
        'do_dogs_dream_5_1.csv'                         -> 'do_dogs_dream'
    """
    fname = os.path.basename(filepath)
    m = _LABEL_PATTERN.match(fname)
    if m:
        return m.group(1)
    # Fallback: strip the numeric suffix
    stem = os.path.splitext(fname)[0]
    parts = stem.rsplit("_", maxsplit=2)
    return parts[0]


# ── Single File Loader ─────────────────────────────────────────────────────

def load_trace(filepath: str) -> pd.DataFrame:
    """
    Load a single trace CSV into a DataFrame with columns:
        time (float), size (float), direction (int: +1 or -1)

    The CSV may have an auto-index column as the first column.
    Lines starting with a comma (header) are skipped.
    """
    # Read the raw file, skip lines starting with ','
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(","):
                continue
            # Try comma-split; col 0 may be an integer index
            parts = line.split(",")
            if len(parts) == 4:
                # index, time, size, direction
                try:
                    t = float(parts[1])
                    s = float(parts[2])
                    d = int(float(parts[3]))
                    rows.append((t, s, d))
                except ValueError:
                    continue  # header row
            elif len(parts) == 3:
                try:
                    t = float(parts[0])
                    s = float(parts[1])
                    d = int(float(parts[2]))
                    rows.append((t, s, d))
                except ValueError:
                    continue

    return pd.DataFrame(rows, columns=["time", "size", "direction"])


# ── Full Dataset Loader ────────────────────────────────────────────────────

def build_label_map(filepaths: List[str]) -> Dict[str, int]:
    """
    Create a sorted, deterministic string→int label map.
    """
    labels = sorted({extract_label(fp) for fp in filepaths})
    return {lbl: idx for idx, lbl in enumerate(labels)}


def load_dataset(
    trace_dir: Optional[Path] = None,
    sort: bool = True,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Walk the trace_csv directory and return:
        filepaths  : list of absolute file path strings (deterministically sorted)
        labels     : list of integer class IDs aligned to filepaths
        label_map  : {command_str -> int_id}

    Returns raw file paths (not loaded DataFrames) so feature extraction
    modules can load on-demand during cross-validation.
    """
    if trace_dir is None:
        trace_dir = config.TRACE_CSV_DIR

    trace_dir = Path(trace_dir)
    if not trace_dir.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")

    filepaths = sorted(
        [str(trace_dir / f) for f in os.listdir(trace_dir) if f.endswith(".csv")]
    )

    if not filepaths:
        raise ValueError(f"No CSV files found in {trace_dir}")

    label_map = build_label_map(filepaths)
    labels = [label_map[extract_label(fp)] for fp in filepaths]

    return filepaths, labels, label_map


def invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    """Return inverted int→str mapping."""
    return {v: k for k, v in label_map.items()}


def get_all_command_strings(label_map: Dict[str, int]) -> List[str]:
    """Return sorted list of all unique voice command strings."""
    return sorted(label_map.keys())


# ── Sanity Check ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    fps, lbls, lmap = load_dataset()
    print(f"Total samples : {len(fps)}")
    print(f"Unique classes: {len(lmap)}")
    print(f"Label map sample: {dict(list(lmap.items())[:3])}")
    # Verify one trace loads cleanly
    df = load_trace(fps[0])
    print(f"\nSample trace: {fps[0]}")
    print(df.head())
    print(f"Rows: {len(df)}, Directions: {df['direction'].unique()}")
