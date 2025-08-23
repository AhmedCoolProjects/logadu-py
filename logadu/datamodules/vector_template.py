import click
import pytorch_lightning as pl
import torch
import pandas as pd
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
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

class NoAggDataModule(pl.LightningDataModule):
    """
    If aggregate=True -> produce (num_windows, D) mean vectors.
    Else -> lazy sliding windows dataset yielding (W, D) per item.
    """
    def __init__(self, merged_file: str, vector_map_file: str,
                 window_size: int, label_col: str, eventid_col: str, content_col: str,
                 batch_size: int = 256, aggregate: bool = False,
                 num_workers: int = 1, enforce_stratify: bool = True,
                 random_state: int = 42):
        super().__init__()
        self.merged_file = merged_file
        self.vector_map_file = vector_map_file
        self.window_size = window_size
        self.label_col = label_col
        self.eventid_col = eventid_col
        self.content_col = content_col
        self.batch_size = batch_size
        self.aggregate = aggregate
        self.num_workers = num_workers
        self.enforce_stratify = enforce_stratify  # added
        self.random_state = random_state          # added
        self.input_dim = None
        self._data_prepared = False

    def setup(self, stage: str = None):
        if self._data_prepared:
            return

        df = pd.read_csv(self.merged_file, low_memory=False)
        # Enforce dtypes
        if self.eventid_col in df.columns:
            df[self.eventid_col] = df[self.eventid_col].astype(str)
        if self.label_col in df.columns:
            df[self.label_col] = df[self.label_col].astype(int)

        vector_map = torch.load(self.vector_map_file)
        self.input_dim = next(iter(vector_map.values())).shape[0]

        click.echo(f"--- Building base tensors (N={len(df)}, dim={self.input_dim}) ---")
        zero_vec = torch.zeros(self.input_dim)
        event_vecs = []
        labels = []
        for eid, lbl in zip(df[self.eventid_col].tolist(), df[self.label_col].tolist()):
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
        
        if self.aggregate:
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
                X, y, test_size=test_ratio, random_state=self.random_state,
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
                random_state=self.random_state,
                stratify=stratify_second
            )

            from torch.utils.data import TensorDataset
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
                random_state=self.random_state, stratify=stratify_main
            )

            stratify_second = y_temp if (np.unique(y_temp).size > 1) else None
            if stratify_second is None:
                click.secho("WARNING: Val split not stratified (single class in temp indices).", fg="yellow")

            idx_train, idx_val, _, _ = train_test_split(
                idx_temp, y_temp, test_size=val_ratio_within_temp,
                random_state=self.random_state, stratify=stratify_second
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
                
        click.secho("Data setup complete (lazy mode)." if not self.aggregate else "Data setup complete (aggregated mode).", fg="green")
        self._data_prepared = True
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)