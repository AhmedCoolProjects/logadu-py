import click
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
from tqdm import tqdm

class MLDataModuleFromMerged(pl.LightningDataModule):
    def __init__(self, merged_file: str, vector_map_file: str, window_size: int, batch_size: int = 256, num_workers: int = 4):
        super().__init__()
        self.merged_file = merged_file
        self.vector_map_file = vector_map_file
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self._data_preprocessed = False

    def setup(self, stage: str = None):
        if self._data_preprocessed:
            return
        # --- 1. Load Input Files ---
        click.echo("--- Step 1: Loading input files ---")
        df = pd.read_csv(self.merged_file)
        vector_map = torch.load(self.vector_map_file)
        click.echo(f"Loaded {len(df)} log events and a map of {len(vector_map)} template vectors.")

        # Get the dimension of the vectors (e.g., 300 for FastText)
        embedding_dim = next(iter(vector_map.values())).shape[0]

        # --- 2. Generate Sequences, Vectorize, and Aggregate in a Single Pass ---
        click.echo(f"--- Step 2: Generating and processing sequences with window_size={self.window_size} ---")
        
        feature_vectors = []
        sequence_labels = []

        # Use a sliding window over the raw event DataFrame
        iterator = range(len(df) - self.window_size)
        for i in tqdm(iterator, desc="Processing sequences"):
            window_df = df.iloc[i : i + self.window_size]
            
            # a. Determine the label for the sequence: 1 if any event in the window is anomalous
            seq_label = 1 if window_df['label'].any() else 0
            
            # b. Get the EventIDs for the window
            event_ids_in_window = window_df['EventId'].tolist()
            
            # c. Convert EventIDs to vectors using the map
            # Use .get() to handle EventIDs that might not be in the map (returns a zero vector)
            vectors_in_sequence = [
                vector_map.get(eid, torch.zeros(embedding_dim)) for eid in event_ids_in_window
            ]
            
            # d. Aggregate the sequence of vectors into a single feature vector
            aggregated_vector = torch.mean(torch.stack(vectors_in_sequence), dim=0)
            
            feature_vectors.append(aggregated_vector)
            sequence_labels.append(seq_label)
            
        # count how many zero vectors we have
        zero_vector_count = sum(1 for vec in feature_vectors if torch.all(vec == 0))
        click.secho(f"Found {zero_vector_count} zero vectors in the aggregated feature vectors. out of {len(feature_vectors)}", fg="yellow")

        X = torch.stack(feature_vectors)
        y = torch.tensor(sequence_labels, dtype=torch.long)
        
        # --- 3. Perform Chronological Split ---
        click.echo("--- Step 3: Performing chronological train-val-test split ---")
        dataset_size = len(X)
        test_split_index = int(dataset_size * 0.8) # 80% for train+val, 20% for test
        
        X_train_val, y_train_val = X[:test_split_index], y[:test_split_index]
        self.X_test, self.y_test = X[test_split_index:], y[test_split_index:]
        
        val_split_index = int(len(X_train_val) * 0.9) # 90% of the remainder for train, 10% for val
        
        self.X_train, self.y_train = X_train_val[:val_split_index], y_train_val[:val_split_index]
        self.X_val, self.y_val = X_train_val[val_split_index:], y_train_val[val_split_index:]
        
        # 4. Create TensorDatasets
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.val_dataset = TensorDataset(self.X_val, self.y_val)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

        self._data_preprocessed = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)