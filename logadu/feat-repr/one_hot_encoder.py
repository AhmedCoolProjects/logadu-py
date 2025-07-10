"""one_hot_encoder.py

Utilities for converting Drain/Spell/FT-Tree template IDs to one-hot
vectors compatible with the DeepLog architecture.

Usage (command-line):
    python one_hot_encoder.py --template_csv templates.csv \
                              --structure_csv structure.csv \
                              --out enc.npy

The script will:
1. Read all EventId values from *templates.csv* (or *structure.csv*).
2. Build an integer index: 0‥|V|-1 where |V| is the number of unique
   templates.
3. Save a NumPy array of shape (|V|, |V|) where row *i* is the one-hot
   vector for template *i* (optional – can also pickle the index dict).

Importable API functions are provided for integration with notebooks.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

###############################################################################
# Core utility functions
###############################################################################

def read_event_ids(csv_path: Path) -> List[str]:
    """Extract the *EventId* column from a Drain/Spell/FT-Tree CSV."""
    ids: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "EventId" not in reader.fieldnames:
            raise ValueError(f"CSV {csv_path} lacks an 'EventId' column")
        ids = [row["EventId"].strip() for row in reader]
    return ids


def build_index(event_ids: List[str]) -> Dict[str, int]:
    """Return a mapping EventId → integer index (stable sort)."""
    vocab = sorted(set(event_ids))  # deterministic ordering
    return {eid: idx for idx, eid in enumerate(vocab)}


def eventid_to_onehot(event_id: str, index: Dict[str, int]) -> np.ndarray:
    """Return a 1-D one-hot numpy vector for *event_id*."""
    vocab_size = len(index)
    onehot = np.zeros(vocab_size, dtype=np.float32)
    onehot[index[event_id]] = 1.0
    return onehot


def dump_numpy_matrix(index: Dict[str, int], out_path: Path) -> None:
    """Materialise the full |V|×|V| identity matrix to *out_path*."""
    vocab_size = len(index)
    mat = np.eye(vocab_size, dtype=np.float32)
    np.save(out_path, mat)
    print(f"Saved one-hot matrix of shape {mat.shape} → {out_path}")


def dump_json_index(index: Dict[str, int], out_path: Path) -> None:
    out_path.write_text(json.dumps(index, indent=2))
    print(f"Saved EventId→index mapping (|V|={len(index)}) → {out_path}")

###############################################################################
# CLI driver
###############################################################################

def _cli():
    p = argparse.ArgumentParser(description="Build one-hot encodings for log templates (DeepLog style)")
    p.add_argument("--template_csv", type=Path, required=True, help="Path to templates.csv or structure.csv containing EventId column")
    p.add_argument("--out", type=Path, default="one_hot.npy", help="Destination .npy file for identity matrix (default: one_hot.npy)")
    p.add_argument("--index_json", type=Path, default="event_index.json", help="Destination JSON mapping file (default: event_index.json)")
    args = p.parse_args()

    ids = read_event_ids(args.template_csv)
    idx_map = build_index(ids)
    dump_numpy_matrix(idx_map, args.out)
    dump_json_index(idx_map, args.index_json)

if __name__ == "__main__":
    _cli()
