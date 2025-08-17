import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import click
from typing import Optional
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from sklearn.model_selection import train_test_split
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except ImportError:
    psutil = None
    
class SlidingWindowDataset(Dataset):
    def __init__(self, event_tensor: torch.Tensor, label_tensor: torch.Tensor, window_size: int):
        self.event_tensor = event_tensor            # [N, D]
        self.label_tensor = label_tensor.to(torch.int8)  # [N]
        self.window_size = window_size
        self.num_windows = self.event_tensor.size(0) - window_size

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        j = idx + self.window_size
        window_vecs = self.event_tensor[idx:j]              # [W, D]
        seq_label = int(torch.any(self.label_tensor[idx:j]))
        return window_vecs, seq_label


class TemplatesDataLoader(pl.LightningDataModule):
    def __init__(self,
                 csv_file_path: str,
                 col_label_name:str,
                 col_content_name: str,
                 col_eventid_name: str,
                 col_template_name: str,
                 # type should be either 'agg_vector', 'vectors', 'indexes', or 'templates'
                 type: str,
                 window_size: int,
                 batch_size: int = 128,
                 remove_duplicates: bool = True,
                 shuffle: bool = True,
                 num_workers: int = 0,
                #  for vectors
                vector_map_path: str = None
                 ):
        super().__init__()
        self.csv_file_path = csv_file_path
        self.col_label_name = col_label_name
        self.col_content_name = col_content_name
        self.col_eventid_name = col_eventid_name
        self.col_template_name = col_template_name
        self.window_size = window_size
        self.batch_size = batch_size
        self.type = type  # 'agg_vector', 'vectors', 'indexes', or 'templates'
        self.remove_duplicates = remove_duplicates
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.vector_map_path = vector_map_path

        # internal variables
        self.vocab_size: Optional[int] = None
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None
        
    def setup(self):
        if self.train_dataset is not None:
            return

        # Load data
        if self.type == 'agg_vector' or self.type == 'vectors':
            self._load_vectors()
        elif self.type == 'indexes':
            self._load_indexes()
        elif self.type == 'templates':
            self._load_templates()
        else:
            raise ValueError(f"Unknown data type: {self.type}")

    
    
    def _load_indexes(self):
        self.df = pd.read_csv(self.csv_file_path, usecols=[self.col_eventid_name, self.col_label_name], low_memory=False)
        
        codes, uniques = pd.factorize(self.df[self.col_eventid_name], sort=True)
        self.vocab_size = len(uniques)
        labels = self.df[self.col_label_name].to_numpy(dtype=np.int8)
        del self.df
        
        if len(codes) <= self.window_size:
            raise ValueError("Not enough data to create sequences with the specified window size.")
        
        windows = np.lib.stride_tricks.sliding_window_view(codes, self.window_size)
        windows = windows[:-1]
        next_events = codes[self.window_size:]
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
    
    def _load_templates(self):
        pass
    
    def _load_vectors(self):
        if self.vector_map_path is None:
            raise ValueError("vector_map_path must be provided for vector type data.")
        
        self.df = pd.read_csv(self.csv_file_path, usecols=[self.col_eventid_name, self.col_label_name], low_memory=False)
        vector_map = torch.load(self.vector_map_path)
        self.input_dim = next(iter(vector_map.values())).shape[0]
        
        zero_vec = torch.zeros(self.input_dim)
        event_vecs = []
        labels = []
        for eid, lbl in zip(self.df[self.col_eventid_name].tolist(), self.df[self.col_label_name].tolist()):
            event_vecs.append(vector_map.get(eid, zero_vec))
            labels.append(lbl)
        event_tensor = torch.stack(event_vecs)           # [N, D] (float32)
        label_tensor = torch.tensor(labels, dtype=torch.int8) # [N]
        
        if psutil:
            rss_mb = _PROC.memory_info().rss / (1024 ** 2)
            click.echo(f"RAM after base tensor build: {rss_mb:,.0f} MB")

                # Desired final ratios: train 72%, val 8%, test 20% (like original)
        test_ratio = 0.20
        val_ratio_of_original = 0.08
        # Convert desired val ratio into fraction of (train+val) block after test removal
        val_ratio_within_temp = val_ratio_of_original / (1 - test_ratio)  # 0.08 / 0.80 = 0.10
        
        if self.type == 'agg_vector':
            click.echo(f"--- Aggregating windows (window_size={self.window_size}) ---")
            num_windows = event_tensor.size(0) - self.window_size
            means = []
            labs = []
            for i in tqdm(range(num_windows), desc="Aggregating"):
                w_end = i + self.window_size
                slice_ = event_tensor[i:w_end]
                means.append(slice_.mean(dim=0))
                labs.append(int(label_tensor[i:w_end].any()))
            X = torch.stack(means)                  # [num_windows, D]
            y = torch.tensor(labs, dtype=torch.long)

            y_np = y.numpy()
            stratify_main = y_np if (np.unique(y_np).size > 1) else None
            if stratify_main is None:
                click.secho("WARNING: Only one class in data; cannot stratify.", fg="yellow")

            # First split (train+val vs test)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=42,
                stratify=stratify_main
            )

            # Second split (train vs val)
            y_temp_np = y_temp.numpy()
            stratify_second = y_temp_np if (np.unique(y_temp_np).size > 1) else None
            if stratify_second is None:
                click.secho("WARNING: Val split not stratified (single class in temp set).", fg="yellow")

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio_within_temp,
                random_state=42,
                stratify=stratify_second
            )

            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)
            self.test_dataset = TensorDataset(X_test, y_test)

        else:
            # Lazy window dataset: build window labels for stratification
            full_dataset = SlidingWindowDataset(event_tensor, label_tensor, self.window_size)
            num_windows = len(full_dataset)
            if num_windows <= 0:
                raise ValueError("No windows produced; check window_size vs data length.")

            # Build window labels efficiently
            # (simple loop; can be optimized if needed)
            win_labels = []
            for i in range(num_windows):
                j = i + self.window_size
                win_labels.append(int(label_tensor[i:j].any()))
            win_labels = np.array(win_labels, dtype=int)
            indices = np.arange(num_windows)

            stratify_main = win_labels if (np.unique(win_labels).size > 1) else None
            if stratify_main is None:
                click.secho("WARNING: Only one class in window labels; cannot stratify.", fg="yellow")

            idx_temp, idx_test, y_temp, y_test = train_test_split(
                indices, win_labels, test_size=test_ratio,
                random_state=42, stratify=stratify_main
            )

            stratify_second = y_temp if (np.unique(y_temp).size > 1) else None
            if stratify_second is None:
                click.secho("WARNING: Val split not stratified (single class in temp indices).", fg="yellow")

            idx_train, idx_val, _, _ = train_test_split(
                idx_temp, y_temp, test_size=val_ratio_within_temp,
                random_state=42, stratify=stratify_second
            )

            self.train_dataset = Subset(full_dataset, idx_train.tolist())
            self.val_dataset = Subset(full_dataset, idx_val.tolist())
            self.test_dataset = Subset(full_dataset, idx_test.tolist())
            
        # Final sanity checks
        def _check_split(ds, name):
            ys = []
            for _, y_ in DataLoader(ds, batch_size=min(512, len(ds))):
                ys.append(y_)
            ys = torch.cat(ys).cpu().numpy()
            uniq = np.unique(ys)
            if uniq.size < 2:
                click.secho(f"WARNING: {name} set has single class {uniq}.", fg="yellow")
        _check_split(self.test_dataset, "Test")
                
        click.secho("Data setup complete (lazy mode)." if not self.type == 'agg_vector' else "Data setup complete (aggregated mode).", fg="green")
        self._data_prepared = True
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )