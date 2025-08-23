"""
Status:
- Supports the 3 Paradigms: Supervised, Semi-supervised, Unsupervised (paradigm)
- Supports 2 Representations: Sequence Vectors, Mean-aggregated Vector (type)

TODO:
- [x] Add type for 'indices': Sequences of Indices
- [ ] Add type for 'raw': Raw Log Messages (Reduced to the ones with first unique EventIds)
"""


import pandas as pd
import torch
import numpy as np
from typing import Optional, Literal
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl
import click
from sklearn.model_selection import train_test_split


class SingleDatasetValidationDataLoader(pl.LightningDataModule):
    """
    DataModule for log anomaly detection with support for:
      - supervised
      - semi (one-class; train on normal only)
      - unsupervised (no labels used for train/val)

    Pipeline (same as before):
      1) Create sequences of semantic vectors from a log file and a vector map.
      2) (If available) assign a binary anomaly label per sequence (any-event-is-1).
      3) Deduplicate sequences.
      5) Split into train / val / test according to paradigm.
      6) Build TensorDatasets.

    Notes:
      • For semi-supervised, training is strictly normal-only; validation can be normal-only
        or mixed depending on `val_contains`.
      • For unsupervised, train/val labels are filled with -1 placeholders; test keeps real
        labels if available (else -1).
    """

    def __init__(
        self,
        csv_file_path: str,
        vector_map_path: str,
        window_size: int,
        # 'seq_vectors_and_label' (sequence) or 'seq_vector_and_label' (mean-aggregated)
        # CHANGED
        type: Literal['seq_vectors_and_label', 'seq_vector_and_label', 'indices'],

        col_label_name: str,
        col_eventid_name: str,
        col_content_name: str,
        col_template_name: str,
        # ----- new knobs -----
        paradigm: Literal["supervised", "semi", "unsupervised"] = "supervised",
        normal_label: int = 0,
        val_contains: Literal["normal_only", "mixed"] = "normal_only",
        train_frac: float = 0.72,
        val_frac: float = 0.08,
        test_frac: float = 0.20,
        # ---------------------
        batch_size: int = 128,
        num_workers: int = 0,
        random_state: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Will be set in setup()
        self.input_dim: Optional[int] = None
        self.vocab_size: Optional[int] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    # -------------------- Lightning hooks --------------------

    def setup(self, stage: Optional[str] = None):
        if self.hparams.type in ("seq_vectors_and_label", "seq_vector_and_label"):
            self._seq_vectors_and_label()
        elif self.hparams.type == "indices":
            self._seq_indices_and_label()
        else:
            raise ValueError(f"Unknown type={self.hparams.type!r}")

    # -------------------- Internal helpers --------------------

    def _seq_indices_and_label(self):
        """
        Build datasets of (seq_indices, next_index, label) where:
        - seq_indices: [window_size] ints (factorized/compact ids)
        - next_index:  scalar int (the event id immediately after the window)
        - label:       0/1; any anomalous event within the window => 1
        """
        # --- Step 0: Load Data (event ids + labels only) ---
        df = pd.read_csv(
            self.hparams.csv_file_path,
            usecols=[self.hparams.col_eventid_name,
                     self.hparams.col_label_name]
        )

        # Compact, zero-based ids (safe even if input ids are sparse/large)
        # ids_compact[i] in [0..V-1], where V = number of unique event ids
        ids_compact, uniques = pd.factorize(
            df[self.hparams.col_eventid_name].to_numpy(), sort=True)
        event_ids = torch.as_tensor(ids_compact, dtype=torch.long)     # [N]
        labels_t = torch.as_tensor(
            df[self.hparams.col_label_name].to_numpy(dtype=np.int8), dtype=torch.long)

        self.vocab_size = int(len(uniques))  # used by LogBERT

        W = int(self.hparams.window_size)
        N = int(event_ids.numel())

        if N < W + 1:
            raise ValueError(
                f"Not enough events ({N}) for window_size={W} with next-index target (need >= W+1).")

        # --- Step 1 & 2: Sliding windows of indices + next index + window label ---
        # seq windows via unfold: shape [N - W + 1, W]
        seq_windows = event_ids.unfold(
            dimension=0, size=W, step=1)        # [N-W+1, W]
        # the next index right after each window: length [N - W]
        # [N-W]
        next_indices = event_ids[W:]
        # labels per window: any anomaly inside window => 1
        label_windows = labels_t.unfold(
            dimension=0, size=W, step=1)       # [N-W+1, W]
        window_labels = label_windows.any(
            dim=1).long()                    # [N-W+1]

        # Align lengths (drop the last window that has no "next")
        seq_windows = seq_windows[:-1]             # [N-W, W]
        window_labels = window_labels[:-1]         # [N-W]
        assert seq_windows.size(0) == next_indices.size(
            0) == window_labels.size(0)

        # --- Step 3: Deduplicate by (window, next) pair (keep first; preserve chronology) ---
        # We build a combined matrix: [seq_tokens..., next_token] and dedupe rows.
        seq_np = seq_windows.cpu().numpy()                 # shape: [N, W]
        nxt_np = next_indices.view(-1, 1).cpu().numpy()    # shape: [N, 1]
        seq_next_np = np.concatenate(
            [seq_np, nxt_np], axis=1)  # shape: [N, W+1]

        _, first_idx = np.unique(seq_next_np, axis=0, return_index=True)
        # chronological order
        first_idx = np.sort(first_idx)
        first_idx_t = torch.from_numpy(first_idx).long()

        X = seq_windows[first_idx_t]     # [M, W]  unique windows
        y = window_labels[first_idx_t]   # [M]
        nxt = next_indices[first_idx_t]    # [M]
        M = len(X)
        click.secho(f"Found {M} unique (window, next) pairs.", fg="green")

        # --- Step 4: Chronological split ---
        n_train = int(round(self.hparams.train_frac * M))
        n_val = int(round(self.hparams.val_frac * M))
        if n_train + n_val > M:
            n_val = max(0, M - n_train)
        n_test = M - n_train - n_val

        X_train_base = X[:n_train]
        y_train_base = y[:n_train]
        nxt_train_base = nxt[:n_train]
        X_val_base = X[n_train:n_train+n_val]
        y_val_base = y[n_train:n_train+n_val]
        nxt_val_base = nxt[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        nxt_test = nxt[n_train+n_val:]

        # --- Step 5: Prepare sets by paradigm ---
        paradigm = self.hparams.paradigm
        normal_label = getattr(self.hparams, "normal_label", 0)
        val_contains = getattr(self.hparams, "val_contains", "normal_only")

        if paradigm == "supervised":
            X_train, y_train, nxt_train = X_train_base, y_train_base, nxt_train_base
            X_val,   y_val,   nxt_val = X_val_base,   y_val_base,   nxt_val_base

        elif paradigm == "semi":
            # Train strictly on normals within TRAIN block
            train_mask = (y_train_base == normal_label)
            if train_mask.sum().item() == 0:
                raise ValueError(
                    "No normal windows in the chronological training block for semi-supervised training.")
            X_train = X_train_base[train_mask]
            nxt_train = nxt_train_base[train_mask]
            y_train = torch.full(
                (len(X_train),), normal_label, dtype=torch.long)

            # Validation: normal_only or mixed, from VAL block
            if val_contains == "normal_only":
                val_mask = (y_val_base == normal_label)
                if val_mask.sum().item() == 0:
                    click.secho(
                        "Warning: no normal-only windows in validation block; using mixed val.", fg="yellow")
                    X_val, y_val, nxt_val = X_val_base, y_val_base, nxt_val_base
                else:
                    X_val, y_val, nxt_val = X_val_base[val_mask], y_val_base[val_mask], nxt_val_base[val_mask]
            else:
                X_val, y_val, nxt_val = X_val_base, y_val_base, nxt_val_base

        elif paradigm == "unsupervised":
            X_train, X_val = X_train_base, X_val_base
            nxt_train, nxt_val = nxt_train_base, nxt_val_base
            y_train = torch.full((len(X_train),), -1, dtype=torch.long)
            y_val = torch.full((len(X_val),),   -1, dtype=torch.long)

        else:
            raise ValueError(f"Unknown paradigm ={paradigm!r}")

        # --- Post-split summary ---
        click.secho(
            f"Split complete: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}",
            fg="green"
        )
        # quick label summaries (unsupervised shows -1 placeholders for train/val)
        click.secho(
            f"Train labels: {torch.bincount(y_train[y_train>=0]) if (y_train>=0).any() else '[-1 placeholders]'}", fg="yellow")
        click.secho(
            f"Val labels:   {torch.bincount(y_val[y_val>=0]) if (y_val>=0).any() else '[-1 placeholders]'}", fg="yellow")
        click.secho(f"Test labels:  {torch.bincount(y_test)}", fg="yellow")

        # --- Infer vocab size for index tokens (0..vocab_size-1) ---
        # Safer to compute from the FULL pre-split arrays if you still have them in scope:
        #   all_max = max(int(seq_windows.max()), int(next_indices.max()))
        # If not, compute from the split tensors:
        def _safe_max(t: torch.Tensor) -> int:
            return int(t.max().item()) if t.numel() > 0 else 0

        all_max_idx = max(
            _safe_max(X_train), _safe_max(X_val), _safe_max(X_test),
            _safe_max(nxt_train), _safe_max(nxt_val), _safe_max(nxt_test)
        )
        self.vocab_size = all_max_idx + 1
        click.secho(f"Detected vocab_size={self.vocab_size}", fg="cyan")

        # Ensure dtypes (Long for indices and labels)
        X_train = X_train.long()
        X_val = X_val.long()
        X_test = X_test.long()
        nxt_train = nxt_train.long()
        nxt_val = nxt_val.long()
        nxt_test = nxt_test.long()
        y_train = y_train.long()
        y_val = y_val.long()
        y_test = y_test.long()

        # --- Wrap into TensorDatasets: (seq_window, next_index, label) ---
        self.train_dataset = TensorDataset(X_train, nxt_train, y_train)
        self.val_dataset = TensorDataset(X_val,   nxt_val,   y_val)
        self.test_dataset = TensorDataset(X_test,  nxt_test,  y_test)

        click.secho("Datasets ready (seq, next, label).", fg="green")

    def _seq_vectors_and_label(self):
        # --- Step 0: Load Data ---
        df = pd.read_csv(self.hparams.csv_file_path, usecols=[
                         self.hparams.col_eventid_name, self.hparams.col_label_name])
        vector_map = torch.load(self.hparams.vector_map_path)
        self.input_dim = next(iter(vector_map.values())).shape[0]

        event_ids = df[self.hparams.col_eventid_name].values
        labels_np = df[self.hparams.col_label_name].to_numpy(dtype=np.int8)

        # Create the full sequence of event vectors for all logs
        zero_vec = torch.zeros(self.input_dim)
        event_tensor = torch.stack(
            [vector_map.get(eid, zero_vec) for eid in event_ids])
        label_tensor = torch.as_tensor(labels_np, dtype=torch.long)

        # --- Step 1 & 2: Create Sequences of Vectors and Their Labels ---
        click.echo(
            f"Step 1 & 2: Creating sliding windows of size {self.hparams.window_size} and their labels...")

        # Use torch.unfold to create sliding windows efficiently
        # Result shape: (num_windows, vector_dim, window_size)
        sequences_unfolded = event_tensor.unfold(
            dimension=0, size=self.hparams.window_size, step=1)
        # Permute to get (num_windows, window_size, vector_dim) which is standard for LSTMs/GRUs
        all_sequences = sequences_unfolded.permute(0, 2, 1)

        # Create corresponding windows for labels
        label_windows = label_tensor.unfold(
            dimension=0, size=self.hparams.window_size, step=1)
        # A sequence is anomalous (1) if any event within it is anomalous
        all_labels = label_windows.any(dim=1).long()

        if self.hparams.type == 'seq_vector_and_label':
            feature_vectors = torch.mean(all_sequences, dim=1)
            click.secho(
                f"Created {len(feature_vectors)} aggregated feature vectors.", fg="green")
        else:
            feature_vectors = all_sequences

        # --- Step 3: Deduplicate (and KEEP chronological order) ---
        click.echo("Step 3: Deduplicating sequences...")
        vectors_flat_np = feature_vectors.view(
            feature_vectors.shape[0], -1).cpu().numpy()
        _, unique_indices_np = np.unique(
            vectors_flat_np, axis=0, return_index=True)
        # ensure chronological order by first occurrence
        unique_indices_np = np.sort(unique_indices_np)
        unique_indices = torch.from_numpy(unique_indices_np).long()

        X = feature_vectors[unique_indices]  # unique vectors
        y = all_labels[unique_indices]  # unique vectors's label
        N = len(X)  # length of unique vectors
        click.secho(f"Found {N} unique feature sets.", fg="green")

        # --- Step 4: Strictly chronological split ---
        click.echo(
            f"Step 4: Chronological split (train={self.hparams.train_frac:.2f}, "
            f"val={self.hparams.val_frac:.2f}, test={self.hparams.test_frac:.2f})"
        )
        # integer-safe sizes (ensure sum == N)
        n_train = int(round(self.hparams.train_frac * N))
        n_val = int(round(self.hparams.val_frac * N))
        if n_train + n_val > N:
            n_val = max(0, N - n_train)
        n_test = N - n_train - n_val

        X_train_base = X[:n_train]
        y_train_base = y[:n_train]
        X_val_base = X[n_train:n_train + n_val]
        y_val_base = y[n_train:n_train + n_val]
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]

        # --- Step 5: Prepare sets by paradigm (no shuffling, no stratify) ---
        paradigm = self.hparams.paradigm
        normal_label = getattr(self.hparams, "normal_label", 0)
        val_contains = getattr(self.hparams, "val_contains", "normal_only")

        if paradigm == "supervised":
            X_train, y_train = X_train_base, y_train_base
            X_val,   y_val = X_val_base,   y_val_base

        elif paradigm == "semi":
            # Train strictly on normals within the TRAIN BLOCK (chronological)
            train_mask = (y_train_base == normal_label)
            if train_mask.sum().item() == 0:
                raise ValueError(
                    "No normal windows in the chronological training block for semi-supervised training.")
            X_train = X_train_base[train_mask]
            # keep labels, but they are all normal; many losses expect a label tensor
            y_train = torch.full(
                (len(X_train),), normal_label, dtype=torch.long)

            # Validation: normal_only or mixed, but still from the VAL BLOCK chronologically
            if val_contains == "normal_only":
                val_mask = (y_val_base == normal_label)
                if val_mask.sum().item() == 0:
                    click.secho(
                        "Warning: no normal-only windows in validation block; using mixed val.", fg="yellow")
                    X_val, y_val = X_val_base, y_val_base
                else:
                    X_val, y_val = X_val_base[val_mask], y_val_base[val_mask]
            else:
                X_val, y_val = X_val_base, y_val_base

        elif paradigm == "unsupervised":
            # Train/val ignore labels; test keeps true labels for evaluation
            X_train, X_val = X_train_base, X_val_base
            y_train = torch.full((len(X_train),), -1, dtype=torch.long)
            y_val = torch.full((len(X_val),),   -1, dtype=torch.long)

        else:
            raise ValueError(f"Unknown paradigm={paradigm!r}")

        click.secho(
            f"Split complete: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", fg="green")
        # quick label summaries (unsupervised shows -1 placeholders for train/val)
        click.secho(
            f"Train labels: {torch.bincount(y_train[y_train>=0]) if (y_train>=0).any() else '[-1 placeholders]'}", fg="yellow")
        click.secho(
            f"Val labels:   {torch.bincount(y_val[y_val>=0]) if (y_val>=0).any() else '[-1 placeholders]'}", fg="yellow")
        click.secho(f"Test labels:  {torch.bincount(y_test)}", fg="yellow")

        # --- Step 6: Wrap datasets ---
        click.echo("Step 6: Preparing final TensorDatasets...")
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val,   y_val)
        self.test_dataset = TensorDataset(X_test,  y_test)

        click.secho(
            "Data preparation pipeline finished successfully!", fg="cyan")

    # inside the DataModule class
    @staticmethod
    def _collate_indices(batch):
        # batch is a list of tuples: (seq [W], next [], label [])
        seq, nxt, lbl = zip(*batch)
        return {
            "seq":   torch.stack(seq, dim=0),   # [B, W]
            "next":  torch.stack(nxt, dim=0),   # [B]
            "label": torch.stack(lbl, dim=0),   # [B]
        }

    def train_dataloader(self):
        if not self.train_dataset:
            raise RuntimeError(
                "The setup() method must be called before the dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,              # keep chronology
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self._collate_indices,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            raise RuntimeError(
                "The setup() method must be called before the dataloader.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self._collate_indices,
        )

    def test_dataloader(self):
        if not self.test_dataset:
            raise RuntimeError(
                "The setup() method must be called before the dataloader.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self._collate_indices,
        )
