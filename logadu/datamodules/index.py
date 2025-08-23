import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import click
# import optional
from typing import Optional

class IndexDataModule(pl.LightningDataModule):
    """
    DataModule for next-event prediction with optional deduplication.
    - If shuffle=True: random (stratified) split (train/val on normal only, test mixed)
    - If shuffle=False: chronological split (first 80% train/val (normal only), last 20% test)
    - Dedup removes duplicate (sequence + next_event) pairs before splitting.
    """
    def __init__(
        self,
        dataset_file: str,
        label_col: str,
        content_col: str,
        eventid_col: str,
        window_size: int = 10,
        batch_size: int = 128,
        remove_duplicates: bool = True,
        shuffle: bool = True,
        # use optional
        num_workers: Optional[int] = 1,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,

    ):
        super().__init__()
        self.dataset_file = dataset_file
        self.label_col = label_col
        self.content_col = content_col
        self.eventid_col = eventid_col
        self.window_size = window_size
        self.batch_size = batch_size
        self.remove_duplicates = remove_duplicates
        self.shuffle = shuffle
        self.num_workers = max(1, (os.cpu_count() or 4) // 2)
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if persistent_workers is not None else (self.num_workers > 0)
        # Will be filled in setup, use optional
        self.vocab_size: Optional[int] = None
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:  # Already processed
            return

        usecols = [self.eventid_col, self.label_col]
        df = pd.read_csv(self.dataset_file, usecols=usecols, low_memory=False)

        # Factorize EventId to contiguous indices
        codes, uniques = pd.factorize(df[self.eventid_col], sort=True) # codes: [0, 1, 2, ...], uniques: [event1, event2, ...]
        self.vocab_size = len(uniques)
        labels = df[self.label_col].to_numpy(dtype=np.int8) # labels: [0, 1, 1, ...]
        del df  # free memory

        if len(codes) <= self.window_size:
            raise ValueError("Not enough events to form a single window.")

        # Vectorized sliding windows
        # windows shape: (N, window_size)
        windows = np.lib.stride_tricks.sliding_window_view(codes, self.window_size) # windows: (N, window_size) N: is the number of windows
        # the last window in windows won't be used since it won't have a coresponding next event, let's delete it
        windows = windows[:-1]
        next_events = codes[self.window_size:]  # shape (N,)
        # Compute window labels: any anomaly inside the window
        label_windows = np.lib.stride_tricks.sliding_window_view(labels, self.window_size)
        label_windows = label_windows[:-1]
        seq_labels = (label_windows.sum(axis=1) > 0).astype(np.int8)

        # Align shape
        click.secho(f"Windows shape: {windows.shape}, Next events shape: {next_events.shape}, Sequence labels shape: {seq_labels.shape}")
        assert windows.shape[0] == next_events.shape[0] == seq_labels.shape[0]

        # Optional dedup (sequence + next)
        if self.remove_duplicates:
            click.secho("Applying deduplication to (sequence, next) pairs (vectorized)...", fg="yellow")
            combo = np.concatenate([windows, next_events[:, None]], axis=1)  # shape (N, window+1)
            # np.unique axis=0 returns sorted unique rows and indices
            unique_rows, unique_indices = np.unique(combo, axis=0, return_index=True)
            # Preserve first occurrence order (np.unique returns sorted)
            unique_indices_sorted = np.sort(unique_indices)
            windows = windows[unique_indices_sorted]
            next_events = next_events[unique_indices_sorted]
            seq_labels = seq_labels[unique_indices_sorted]
            click.echo(f"Number of sequences after deduplication: {len(windows)} / {len(codes)}")

        # Split
        if self.shuffle:
            # Random stratified split (test 20%)
            from sklearn.model_selection import train_test_split
            idx_all = np.arange(len(windows))
            train_val_idx, test_idx = train_test_split(
                idx_all, test_size=0.2, random_state=42, stratify=seq_labels
            )
            # Train/val only normal
            normal_mask = seq_labels[train_val_idx] == 0
            normal_indices = train_val_idx[normal_mask]
            train_idx, val_idx = train_test_split(
                normal_indices, test_size=0.1, random_state=42
            )
        else:
            # Chronological: first 80% -> train/val candidate, last 20% test
            n_total = len(windows)
            train_val_end = int(n_total * 0.8)
            idx_all = np.arange(len(windows))
            train_val_idx = idx_all[:train_val_end]
            test_idx = idx_all[train_val_end:]
            normal_mask = seq_labels[train_val_idx] == 0
            normal_indices = train_val_idx[normal_mask]
            # 90/10 split inside normal_indices preserving order
            split_point = int(len(normal_indices) * 0.9)
            train_idx = normal_indices[:split_point]
            val_idx = normal_indices[split_point:]

        # Build tensors
        def to_tensor(idx):
            return (
                torch.as_tensor(windows[idx], dtype=torch.long),
                torch.as_tensor(next_events[idx], dtype=torch.long),
                torch.as_tensor(seq_labels[idx], dtype=torch.long),
            )

        X_train, y_train_next, _ = to_tensor(train_idx)
        X_val, y_val_next, _ = to_tensor(val_idx)
        X_test, y_test_next, y_test_label = to_tensor(test_idx)

        self.train_dataset = TensorDataset(X_train, y_train_next)
        self.val_dataset = TensorDataset(X_val, y_val_next)
        # Test includes both next-event label and sequence anomaly label
        self.test_dataset = TensorDataset(X_test, y_test_next, y_test_label)

    # Dataloaders (only next-event prediction for training/validation)
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )