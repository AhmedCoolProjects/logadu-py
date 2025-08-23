import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
import pytorch_lightning as pl
import click
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
import hdbscan
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except ImportError:
    psutil = None


class SlidingWindowDataset(Dataset):
    """
    Dataset that returns (window_tensor, window_label)
    window_label = 1 if any event in the window is anomalous else 0
    Assumes event_tensor shape [N, D] (or [N] if embeddings not used)
    """
    def __init__(self, event_tensor: torch.Tensor, label_tensor: torch.Tensor, window_size: int):
        if event_tensor.size(0) != label_tensor.size(0):
            raise ValueError("event_tensor and label_tensor length mismatch")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.event_tensor = event_tensor
        self.label_tensor = label_tensor.to(torch.int8)
        self.window_size = window_size
        # Inclusive count: N - W + 1 windows (standard definition)
        self.num_windows = event_tensor.size(0) - window_size + 1
        if self.num_windows <= 0:
            raise ValueError("Not enough events to form a single window")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx: int):
        j = idx + self.window_size
        window_vecs = self.event_tensor[idx:j]
        # Any anomalous label inside the window
        seq_label = int(self.label_tensor[idx:j].any())
        return window_vecs, seq_label



class TemplatesDataLoader(pl.LightningDataModule):
    """
    DataModule supporting several 'type' modes:
      - 'indexes': integer event ids with next-event prediction (train/val on normal only)
      - 'templates': like 'indexes' but factorizes template column
      - 'agg_vector': aggregated (mean) embedding per sliding window for anomaly classification
      - 'vectors': sliding window dataset returning raw embedding sequences + window label
    """
    def __init__(
        self,
        csv_file_path: str,
        col_label_name: str,
        col_content_name: str,
        col_eventid_name: str,
        col_template_name: str,
        type: str,
        window_size: int,
        batch_size: int = 128,
        remove_duplicates: bool = True,
        shuffle: bool = True,
        num_workers: int = 0,
        vector_map_path: Optional[str] = None,
        random_state: int = 42,
        known_normal_ratio: float = 0.5,  # Ratio of normal logs to use as "known"
        n_ica_components: int = 100,
        min_cluster_size: int = 100,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.csv_file_path = csv_file_path
        self.col_label_name = col_label_name
        self.col_content_name = col_content_name
        self.col_eventid_name = col_eventid_name
        self.col_template_name = col_template_name
        self.window_size = window_size
        self.batch_size = batch_size
        self.type = type
        self.remove_duplicates = remove_duplicates
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.vector_map_path = vector_map_path
        self.random_state = random_state
        self.persistent_workers = persistent_workers

        self.vocab_size: Optional[int] = None
        self.input_dim: Optional[int] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        self.save_hyperparameters()
        
        

    # -------------------- Public API --------------------
    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return
        if self.type in ('agg_vector', 'vectors'):
            self._load_vectors()
        elif self.type == 'indexes':
            self._load_indexes()
        elif self.type == 'cnn_indexes':
            self._load_indexes_for_cnn()
        elif self.type == 'templates':
            pass
        elif self.type == 'semisup':
            self._load_semisup()
        else:
            raise ValueError(f"Unknown data type: {self.type}")

    # -------------------- Internal helpers --------------------
    def _factorize_column(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        codes, uniques = pd.factorize(series, sort=True)
        self.vocab_size = len(uniques)
        return codes.astype(np.int32, copy=False), uniques  # downcast for 
    

    # def _load_indexes(self, is_template: bool = False):
    #     df = pd.read_csv(self.csv_file_path, usecols=[self.col_eventid_name, self.col_label_name], low_memory=False)
    #     codes, uniques = self._factorize_column(df[self.col_eventid_name])
    #     labels = df[self.col_label_name].to_numpy(dtype=np.int8)
    #     del df

    #     if len(codes) <= self.window_size:
    #         raise ValueError("Not enough data to create sequences with the specified window size")

    #     # --- REFACTORED SECTION ---
    #     # For a sequence of length N and a window size W, we can create (N - W) pairs
    #     # of (input_window, next_event_label).
    #     # The last possible window (starting at index N-W) has no "next event" to predict,
    #     # so we exclude it.
    #     num_sequences = len(codes) - self.window_size

    #     # Create the windows for the input sequences (X)
    #     # Shape: (num_sequences, window_size)
    #     all_windows = np.lib.stride_tricks.sliding_window_view(codes, self.window_size)
    #     windows = all_windows[:num_sequences]

    #     # The target (y) for each window is the event that immediately follows it.
    #     # Shape: (num_sequences,)
    #     next_events = codes[self.window_size:]

    #     # Similarly, create windows from the labels to determine if a sequence is anomalous.
    #     # A sequence's label is 1 if any event within it is anomalous.
    #     all_label_windows = np.lib.stride_tricks.sliding_window_view(labels, self.window_size)
    #     label_windows = all_label_windows[:num_sequences]
    #     seq_labels = (label_windows.sum(axis=1) > 0).astype(np.int8)

    #     # --- END REFACTORED SECTION ---

    #     assert windows.shape[0] == next_events.shape[0] == seq_labels.shape[0]

    #     if self.remove_duplicates:
    #         click.secho("Deduplicating (sequence,next) pairs...", fg="yellow")
    #         combo = np.concatenate([windows, next_events[:, None]], axis=1)
    #         # If extremely large, warn
    #         if combo.shape[0] > 5_000_000:
    #             click.secho("Large combo array; dedup may be slow/memory heavy", fg="red")
    #         _, unique_idx = np.unique(combo, axis=0, return_index=True)
    #         unique_idx.sort()
    #         windows = windows[unique_idx]
    #         next_events = next_events[unique_idx]
    #         seq_labels = seq_labels[unique_idx]
    #         click.echo(f"After dedup: {len(windows)} sequences")

    #     # Splitting
    #     idx_all = np.arange(len(windows))
    #     if self.shuffle:
    #         strat = seq_labels if (np.unique(seq_labels).size > 1) else None
    #         train_val_idx, test_idx, y_train_val, y_test = train_test_split(
    #             idx_all, seq_labels, test_size=0.2, random_state=self.random_state, stratify=strat
    #         )
    #     else:
    #         # Deterministic split: last 20% test
    #         split_point = int(len(windows) * 0.8)
    #         train_val_idx = idx_all[:split_point]
    #         test_idx = idx_all[split_point:]
    #         y_train_val = seq_labels[train_val_idx]
    #         y_test = seq_labels[test_idx]

    #     # Train/val only normal (label==0)
    #     normal_mask = y_train_val == 0
    #     normal_indices = train_val_idx[normal_mask]
    #     if len(normal_indices) < 10:
    #         click.secho("WARNING: Very few normal windows for train/val", fg="red")
    #     if self.shuffle:
    #         train_idx, val_idx = train_test_split(
    #             normal_indices, test_size=0.1,
    #             random_state=self.random_state
    #         )
    #     else:
    #         val_split = max(1, int(len(normal_indices) * 0.1))
    #         train_idx = normal_indices[:-val_split]
    #         val_idx = normal_indices[-val_split:]

    #     # Tensor conversion
    #     def to_tensor(idxs):
    #         return (
    #             torch.as_tensor(windows[idxs], dtype=torch.long),
    #             torch.as_tensor(next_events[idxs], dtype=torch.long),
    #             torch.as_tensor(seq_labels[idxs], dtype=torch.long),
    #         )

    #     X_train, y_train_next, _ = to_tensor(train_idx)
    #     X_val, y_val_next, _ = to_tensor(val_idx)
    #     X_test, y_test_next, y_test_seq = to_tensor(test_idx)

    #     self.train_dataset = TensorDataset(X_train, y_train_next)
    #     self.val_dataset = TensorDataset(X_val, y_val_next)
    #     # test includes sequence anomaly label
    #     self.test_dataset = TensorDataset(X_test, y_test_next, y_test_seq)

    #     click.secho(
    #         f"{'Template' if is_template else 'Index'} mode: "
    #         f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
    #         f"vocab_size={self.vocab_size}",
    #         fg="green"
    #     )    


# In your TemplatesDataLoader class

    def _load_vectors(self):
        if self.vector_map_path is None:
            raise ValueError("vector_map_path must be provided for vector type data.")

        click.echo(f"Loading data for vector-based model (type: {self.type})...")
        
        # --- Step 1: Create the base tensors (Unchanged) ---
        df = pd.read_csv(self.csv_file_path, usecols=[self.col_eventid_name, self.col_label_name], low_memory=False)
        vector_map = torch.load(self.vector_map_path)
        self.input_dim = next(iter(vector_map.values())).shape[0]
        
        event_ids = df[self.col_eventid_name].values
        labels_np = df[self.col_label_name].to_numpy(dtype=np.int8)
        del df

        zero_vec = torch.zeros(self.input_dim)
        event_vecs = [vector_map.get(eid, zero_vec) for eid in event_ids]
        event_tensor = torch.stack(event_vecs)
        label_tensor = torch.as_tensor(labels_np, dtype=torch.long)

        # --- Step 2: Create the full sliding window dataset (Unified) ---
        # This dataset can lazily provide either full windows or aggregated windows.
        full_dataset = SlidingWindowDataset(event_tensor, label_tensor, self.window_size)
        num_windows = len(full_dataset)

        # --- Step 3: Pre-calculate all window labels for splitting (Unified) ---
        # This is more efficient than calling __getitem__ repeatedly.
        window_label_arr = (np.lib.stride_tricks.sliding_window_view(
            label_tensor.numpy(), self.window_size).max(axis=1)
        ).astype(np.int8)

        # --- Step 4: Split the window indices (Unified) ---
        indices = np.arange(num_windows)
        stratify_labels = window_label_arr if np.unique(window_label_arr).size > 1 else None
        
        # Split into (train+val) and test
        idx_train_val, idx_test = train_test_split(
            indices, test_size=0.2, random_state=self.random_state, stratify=stratify_labels
        )
        
        # Split (train+val) into train and val
        stratify_train_val = window_label_arr[idx_train_val] if stratify_labels is not None else None
        idx_train, idx_val = train_test_split(
            idx_train_val, test_size=0.125, random_state=self.random_state, stratify=stratify_train_val
        )

        # --- Step 5: Create the final datasets based on the type ---
        if self.type == 'agg_vector':
            click.echo("Aggregating vectors for train/val/test sets...")
            
            # Helper function to aggregate data from a subset of the full dataset
            def aggregate_subset(indices):
                # Create a temporary dataloader to efficiently fetch windows
                subset_loader = DataLoader(Subset(full_dataset, indices), batch_size=512, shuffle=False)
                
                all_means = []
                all_labels = []
                
                for windows, labels in subset_loader:
                    # Calculate mean on the batch and append
                    all_means.append(windows.mean(dim=1))
                    all_labels.append(labels)
                    
                return torch.cat(all_means), torch.cat(all_labels)

            X_train, y_train = aggregate_subset(idx_train)
            X_val, y_val = aggregate_subset(idx_val)
            X_test, y_test = aggregate_subset(idx_test)

            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)
            self.test_dataset = TensorDataset(X_test, y_test)
            
        else: # For 'vectors', 'logrobust', etc.
            # This is your original, memory-efficient logic using Subsets
            self.train_dataset = Subset(full_dataset, idx_train)
            self.val_dataset = Subset(full_dataset, idx_val)
            self.test_dataset = Subset(full_dataset, idx_test)

        click.secho(f"Vector mode data setup complete. Datasets created for type='{self.type}'.", fg="green")

    # -------------------- DataLoaders --------------------
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.persistent_workers and self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.persistent_workers and self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.persistent_workers and self.num_workers > 0)
        
    def _create_windows_and_labels(self):
        """ Helper to create windows and labels, shared by both index methods. """
        df = pd.read_csv(self.csv_file_path, usecols=[self.col_eventid_name, self.col_label_name], low_memory=False)
        codes, _ = self._factorize_column(df[self.col_eventid_name])
        labels = df[self.col_label_name].to_numpy(dtype=np.int8)
        del df

        if len(codes) <= self.window_size:
            raise ValueError("Not enough data to create sequences")

        num_sequences = len(codes) - self.window_size
        
        # Input sequences (X)
        windows = np.lib.stride_tricks.sliding_window_view(codes, self.window_size)[:num_sequences]
        
        # Sequence anomaly labels (y for CNN)
        label_windows = np.lib.stride_tricks.sliding_window_view(labels, self.window_size)[:num_sequences]
        seq_labels = (label_windows.sum(axis=1) > 0).astype(np.int8)

        # Next event labels (y for DeepLog)
        next_events = codes[self.window_size:]
        
        return windows, seq_labels, next_events
        
    def _load_indexes_for_cnn(self):
        """
        NEW METHOD: Loads and splits data for a supervised model like LogCNN.
        Trains, validates, and tests on all data (normal and anomalous).
        """
        windows, seq_labels, _ = self._create_windows_and_labels()

        # Splitting (e.g., 80% train, 10% validation, 10% test)
        idx_all = np.arange(len(windows))
        
        # First split: Separate out the test set (20%)
        stratify_labels = seq_labels if np.unique(seq_labels).size > 1 else None
        train_val_idx, test_idx, y_train_val, _ = train_test_split(
            idx_all, seq_labels, test_size=0.2, random_state=self.random_state, stratify=stratify_labels
        )
        
        # Second split: Split the remaining data into train and validation (e.g., 80/20 of the remainder)
        stratify_labels_tv = y_train_val if np.unique(y_train_val).size > 1 else None
        train_idx, val_idx, _, _ = train_test_split(
            train_val_idx, y_train_val, test_size=0.125, random_state=self.random_state, stratify=stratify_labels_tv
        ) # 0.125 * 0.8 = 0.1 (10% of original)

        # Create TensorDatasets for LogCNN: (sequences, anomaly_labels)
        X_train = torch.as_tensor(windows[train_idx], dtype=torch.long)
        y_train = torch.as_tensor(seq_labels[train_idx], dtype=torch.long)
        self.train_dataset = TensorDataset(X_train, y_train)
        
        X_val = torch.as_tensor(windows[val_idx], dtype=torch.long)
        y_val = torch.as_tensor(seq_labels[val_idx], dtype=torch.long)
        self.val_dataset = TensorDataset(X_val, y_val)
        
        X_test = torch.as_tensor(windows[test_idx], dtype=torch.long)
        y_test = torch.as_tensor(seq_labels[test_idx], dtype=torch.long)
        self.test_dataset = TensorDataset(X_test, y_test)
        
        click.secho(
            f"LogCNN mode: train={len(self.train_dataset)} val={len(self.val_dataset)} test={len(self.test_dataset)} "
            f"vocab_size={self.vocab_size}",
            fg="green"
        )
        
    def _load_indexes(self):
        """
        Original method for DeepLog. Trains/validates on normal data only.
        """
        windows, seq_labels, next_events = self._create_windows_and_labels()

        idx_all = np.arange(len(windows))
        
        # Main split into train/val pool and test pool
        train_val_idx, test_idx, y_train_val, _ = train_test_split(
            idx_all, seq_labels, test_size=0.2, random_state=self.random_state, stratify=seq_labels
        )

        # --- DeepLog Specific Logic ---
        # Filter train/val pool to only include normal sequences (label == 0)
        normal_mask = y_train_val == 0
        normal_indices = train_val_idx[normal_mask]
        
        # Split the normal-only indices into a final train and validation set
        train_idx, val_idx = train_test_split(
            normal_indices, test_size=0.1, random_state=self.random_state
        )

        # Create datasets for DeepLog
        X_train = torch.as_tensor(windows[train_idx], dtype=torch.long)
        y_train_next = torch.as_tensor(next_events[train_idx], dtype=torch.long)
        self.train_dataset = TensorDataset(X_train, y_train_next)

        X_val = torch.as_tensor(windows[val_idx], dtype=torch.long)
        y_val_next = torch.as_tensor(next_events[val_idx], dtype=torch.long)
        self.val_dataset = TensorDataset(X_val, y_val_next)

        # Test set uses all data (normal and anomalous)
        X_test = torch.as_tensor(windows[test_idx], dtype=torch.long)
        y_test_next = torch.as_tensor(next_events[test_idx], dtype=torch.long)
        y_test_seq = torch.as_tensor(seq_labels[test_idx], dtype=torch.long)
        self.test_dataset = TensorDataset(X_test, y_test_next, y_test_seq)
        
        click.secho(
            f"DeepLog mode: train={len(self.train_dataset)} val={len(self.val_dataset)} test={len(self.test_dataset)} "
            f"vocab_size={self.vocab_size}",
            fg="green"
        )
        
    def _load_semisup(self):
       
        # 1. Load data and semantic vectors
        df = pd.read_csv(self.hparams.csv_file_path, usecols=[self.hparams.col_template_name, self.hparams.col_label_name])
        vector_map = torch.load(self.hparams.vector_map_path)
        self.input_dim = next(iter(vector_map.values())).shape[0]
        
        # Create the full sequence of event vectors and their true labels
        event_vecs = [vector_map.get(t, torch.zeros(self.input_dim)) for t in df[self.hparams.col_template_name]]
        event_tensor = torch.stack(event_vecs)
        label_tensor = torch.tensor(df[self.hparams.col_label_name].values, dtype=torch.long)

        # 2. Chronological Split (as described in the paper)
        train_size = int(0.7 * len(df))
        val_size = int(0.1 * len(df))
        
        train_events = event_tensor[:train_size]
        train_labels = label_tensor[:train_size]
        
        val_events = event_tensor[train_size : train_size + val_size]
        val_labels = label_tensor[train_size : train_size + val_size]
        
        test_events = event_tensor[train_size + val_size :]
        test_labels = label_tensor[train_size + val_size :]

        # 3. Probabilistic Label Estimation (on the training set)
        click.echo("Starting Probabilistic Label Estimation...")
        train_dataset_full_labels = SlidingWindowDataset(train_events, train_labels, self.hparams.window_size)
        
        # Get true labels for all windows in the training set
        train_window_true_labels = np.array([train_dataset_full_labels[i][1] for i in range(len(train_dataset_full_labels))])
        
        normal_window_indices = np.where(train_window_true_labels == 0)[0]
        anomalous_window_indices = np.where(train_window_true_labels == 1)[0]
        
        # Split normal indices into "known" and "unknown"
        known_normal_indices, unlabeled_normal_indices = train_test_split(
            normal_window_indices, train_size=(self.hparams.known_normal_ratio)
        )
        
        # Create the unlabeled pool
        unlabeled_pool_indices = np.concatenate([unlabeled_normal_indices, anomalous_window_indices])
        
        # Aggregate sequence vectors for clustering (summation as per paper)
        click.echo(f"Aggregating {len(known_normal_indices) + len(unlabeled_pool_indices)} sequence vectors...")
        known_normal_seq_vecs = torch.stack([train_dataset_full_labels[i][0].sum(dim=0) for i in known_normal_indices])
        unlabeled_pool_seq_vecs = torch.stack([train_dataset_full_labels[i][0].sum(dim=0) for i in unlabeled_pool_indices])

        # Dimension Reduction with FastICA
        click.echo(f"Performing ICA dimension reduction to {self.hparams.n_ica_components} components...")
        ica = FastICA(n_components=self.hparams.n_ica_components, random_state=42, whiten='unit-variance')
        all_seq_vecs = torch.cat([known_normal_seq_vecs, unlabeled_pool_seq_vecs]).numpy()
        reduced_vecs = ica.fit_transform(all_seq_vecs)

        # Clustering with HDBSCAN
        click.echo(f"Clustering with HDBSCAN (min_cluster_size={self.hparams.min_cluster_size})...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.hparams.min_cluster_size)
        cluster_labels = clusterer.fit_predict(reduced_vecs)
        
        # Identify which clusters contain "known normal" samples
        num_known = len(known_normal_indices)
        known_normal_cluster_ids = set(cluster_labels[:num_known])
        
        # Generate new labels for the entire training set
        new_train_labels = np.zeros_like(train_window_true_labels)
        
        # Map cluster labels back to the unlabeled pool
        unlabeled_cluster_labels = cluster_labels[num_known:]
        for i, original_idx in enumerate(unlabeled_pool_indices):
            cluster_id = unlabeled_cluster_labels[i]
            if cluster_id in known_normal_cluster_ids and cluster_id != -1: # Not noise
                new_train_labels[original_idx] = 0 # Normal
            else:
                new_train_labels[original_idx] = 1 # Anomalous
                
        # Known normals are always normal
        new_train_labels[known_normal_indices] = 0
        
        click.secho(f"PLE complete. Generated {sum(new_train_labels)} anomalous labels for training.", fg="green")

        # 4. Create Final Datasets
        # Training set uses the NEWLY generated labels
        self.train_dataset = SlidingWindowDataset(train_events, torch.from_numpy(new_train_labels), self.hparams.window_size)
        
        # Validation and Test sets use the ORIGINAL ground-truth labels
        self.val_dataset = SlidingWindowDataset(val_events, val_labels, self.hparams.window_size)
        self.test_dataset = SlidingWindowDataset(test_events, test_labels, self.hparams.window_size)
        
    # --- DataLoader methods are standard ---
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)