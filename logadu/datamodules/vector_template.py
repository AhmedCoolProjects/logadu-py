import click
import pytorch_lightning as pl
import torch
import pandas as pd
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset, Subset
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
                 window_size: int, batch_size: int = 256,
                 aggregate: bool = False,
                 num_workers: int = 4):
        super().__init__()
        self.merged_file = merged_file
        self.vector_map_file = vector_map_file
        self.window_size = window_size
        self.batch_size = batch_size
        self.aggregate = aggregate
        self.num_workers = num_workers
        self.input_dim = None
        self._data_prepared = False

    def setup(self, stage: str = None):
        if self._data_prepared:
            return

        df = pd.read_csv(self.merged_file, low_memory=False)
        # Enforce dtypes
        if 'EventId' in df.columns:
            df['EventId'] = df['EventId'].astype(str)
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)

        vector_map = torch.load(self.vector_map_file)
        self.input_dim = next(iter(vector_map.values())).shape[0]

        click.echo(f"--- Building base tensors (N={len(df)}, dim={self.input_dim}) ---")
        zero_vec = torch.zeros(self.input_dim)
        event_vecs = []
        labels = []
        for eid, lbl in zip(df['EventId'].tolist(), df['label'].tolist()):
            event_vecs.append(vector_map.get(eid, zero_vec))
            labels.append(lbl)
        event_tensor = torch.stack(event_vecs)                # [N, D] (float32)
        label_tensor = torch.tensor(labels, dtype=torch.int8) # [N]

        if psutil:
            rss_mb = _PROC.memory_info().rss / (1024 ** 2)
            click.echo(f"RAM after base tensor build: {rss_mb:,.0f} MB")

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
            total = len(X)
            test_split = int(total * 0.8)
            val_split = int(test_split * 0.9)
            from torch.utils.data import TensorDataset
            self.train_dataset = TensorDataset(X[:val_split], y[:val_split])
            self.val_dataset = TensorDataset(X[val_split:test_split], y[val_split:test_split])
            self.test_dataset = TensorDataset(X[test_split:], y[test_split:])
        else:
            full_dataset = SlidingWindowDataset(event_tensor, label_tensor, self.window_size)
            total = len(full_dataset)
            test_split = int(total * 0.8)
            val_split = int(test_split * 0.9)
            self.train_dataset = Subset(full_dataset, range(0, val_split))
            self.val_dataset = Subset(full_dataset, range(val_split, test_split))
            self.test_dataset = Subset(full_dataset, range(test_split, total))

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