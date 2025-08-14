import click
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd

try:
    import psutil, os
    _PROC = psutil.Process(os.getpid())
except ImportError:
    psutil = None
import torch.nn.functional as F

class MLDataModuleFromMerged(pl.LightningDataModule):
    def __init__(self, merged_file: str, vector_map_file: str, window_size: int,
                 batch_size: int = 256, num_workers: int = 4,
                 use_fast: bool = True):
        super().__init__()
        self.merged_file = merged_file
        self.vector_map_file = vector_map_file
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_fast = use_fast
        self._data_preprocessed = False

    def setup(self, stage: str = None):
        if self._data_preprocessed:
            return

        click.echo("--- Step 1: Loading input files ---")
        df = pd.read_csv(self.merged_file, low_memory=False)
        if 'EventId' in df.columns:
            df['EventId'] = df['EventId'].astype(str)
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)

        vector_map = torch.load(self.vector_map_file)
        embedding_dim = next(iter(vector_map.values())).shape[0]
        zero_vec = torch.zeros(embedding_dim)

        N = len(df)
        if N <= self.window_size:
            raise ValueError(f"Not enough rows ({N}) for window_size={self.window_size}")

        click.echo(f"Building base event tensor (N={N}, dim={embedding_dim}) ...")
        event_tensor = torch.stack([
            vector_map.get(eid, zero_vec)
            for eid in df['EventId'].tolist()
        ])  # [N, D]
        label_tensor = torch.tensor(df['label'].tolist(), dtype=torch.int8)  # [N]

        if psutil:
            click.echo(f"RAM after base tensors: { _PROC.memory_info().rss / (1024**2):,.0f} MB")

        W = self.window_size
        num_windows = N - W

        if self.use_fast:
            click.echo(f"--- Fast aggregation (vectorized) for window_size={W} ---")

            # 1. Window means via cumulative sum
            # cumsum shape: [N, D]; pad a zero row for simpler slicing
            cumsum = torch.vstack([torch.zeros(1, event_tensor.size(1)), event_tensor.cumsum(dim=0)])  # [N+1, D]
            # window sums: cumsum[i+W] - cumsum[i] for i in 0..N-W-1 -> vectorized
            window_sums = cumsum[W:] - cumsum[:-W]               # [num_windows, D]
            means = window_sums / W                              # [num_windows, D]

            # 2. Window labels = any() over window -> max_pool1d over int tensor
            # reshape to [1,1,N] -> max_pool1d kernel=W stride=1 -> [1,1,num_windows]
            lbl_pool = F.max_pool1d(label_tensor.view(1, 1, N).float(), kernel_size=W, stride=1)
            seq_labels = lbl_pool.view(-1).to(torch.long)        # [num_windows]

            X = means
            y = seq_labels
        else:
            # Fallback (original slower loop) if needed
            click.echo(f"--- Slow loop aggregation (window_size={W}) ---")
            feature_vectors = []
            sequence_labels = []
            for i in range(num_windows):
                j = i + W
                slice_ = event_tensor[i:j]
                feature_vectors.append(slice_.mean(dim=0))
                sequence_labels.append(int(label_tensor[i:j].any()))
            X = torch.stack(feature_vectors)
            y = torch.tensor(sequence_labels, dtype=torch.long)

        # Chronological split
        dataset_size = len(X)
        test_split_index = int(dataset_size * 0.8)
        val_split_index = int(test_split_index * 0.9)

        X_train_val, y_train_val = X[:test_split_index], y[:test_split_index]
        self.X_test, self.y_test = X[test_split_index:], y[test_split_index:]
        self.X_train, self.y_train = X_train_val[:val_split_index], y_train_val[:val_split_index]
        self.X_val, self.y_val = X_train_val[val_split_index:], y_train_val[val_split_index:]

        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.val_dataset = TensorDataset(self.X_val, self.y_val)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

        click.secho(f"Prepared {dataset_size} windows (fast={self.use_fast}).", fg="green")
        self._data_preprocessed = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)