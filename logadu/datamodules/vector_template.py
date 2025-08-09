import click
import pytorch_lightning as pl
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import pickle

class NoAggDataModule(pl.LightningDataModule):
    """
    A unified DataModule for both traditional and deep learning models that
    builds features directly from a merged log file and a vector map.

    This module can either produce:
    1. A fixed-size feature matrix (aggregation=True) for PCA, RF, KNN, etc.
    2. Sequences of semantic vectors (aggregation=False) for LogRobust, NeuralLog.
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
        self.input_dim = None  # Will store the embedding dimension
        self._data_prepared = False

    def setup(self, stage: str = None):
        if self._data_prepared:
            return
        
        # --- 1. Load Input Files ---
        df = pd.read_csv(self.merged_file)
        vector_map = torch.load(self.vector_map_file)
        self.input_dim = next(iter(vector_map.values())).shape[0]

        # --- 2. Generate Sequences and Vectorize ---
        click.echo(f"--- Generating sequences (window_size={self.window_size}) and vectorizing ---")
        
        all_sequences = []
        all_labels = []
        
        iterator = range(len(df) - self.window_size)
        for i in tqdm(iterator, desc="Processing sequences"):
            window_df = df.iloc[i : i + self.window_size]
            
            seq_label = 1 if window_df['label'].any() else 0
            event_ids_in_window = window_df['EventId'].tolist()
            
            vectors_in_sequence = torch.stack([
                vector_map.get(eid, torch.zeros(self.input_dim)) for eid in event_ids_in_window
            ])
            
            all_sequences.append(vectors_in_sequence)
            all_labels.append(seq_label)

        # --- 3. Optional Aggregation Step ---
        if self.aggregate:
            click.echo("--- Aggregating sequences into a fixed-size feature matrix ---")
            # This is the path for traditional ML models (PCA, RF, KNN)
            X = torch.stack([torch.mean(seq, dim=0) for seq in all_sequences])
            y = torch.tensor(all_labels, dtype=torch.long)
        else:
            # This is the path for sequential models (LogRobust, NeuralLog)
            # No padding is needed because all sequences have the same length (window_size)
            X = torch.stack(all_sequences)
            y = torch.tensor(all_labels, dtype=torch.long)

        # --- 4. Perform Chronological Split ---
        click.echo("--- Performing chronological train-val-test split ---")
        dataset_size = len(X)
        test_split_index = int(dataset_size * 0.8)
        val_split_index = int(test_split_index * 0.9)

        self.train_dataset = TensorDataset(X[:val_split_index], y[:val_split_index])
        self.val_dataset = TensorDataset(X[val_split_index:test_split_index], y[val_split_index:test_split_index])
        self.test_dataset = TensorDataset(X[test_split_index:], y[test_split_index:])
        
        click.secho("Data setup complete.", fg="green")
        self._data_prepared = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)