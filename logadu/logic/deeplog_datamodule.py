# /logadu/logic/deeplog_datamodule.py

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import ast

class DeepLogDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for DeepLog.

    This class handles loading, splitting, and creating DataLoaders.
    """
    def __init__(self, dataset_file: str, batch_size: int = 32):
        super().__init__()
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.num_classes = None
        self.window_size = None

    def setup(self, stage=None):
        """
        Called on every GPU, this method handles data loading and splitting.
        'stage' can be 'fit', 'validate', 'test', or 'predict'.
        """
        # Load the entire dataset once
        df = pd.read_csv(self.dataset_file)
        df['sequences'] = df['sequences'].apply(ast.literal_eval)

        # Determine vocabulary size (num_classes) from all data
        all_events = set()
        for seq in df['sequences']:
            all_events.update(seq)
        all_events.update(df['next'])
        self.num_classes = max(all_events) + 1

        # Determine window_size from the data
        self.window_size = len(df['sequences'].iloc[0])

        # IMPORTANT: DeepLog only trains on NORMAL data.
        # However, we validate and test on ALL data to see how it performs on anomalies.
        normal_df = df[df['label'] == 0]
        
        # Prepare training data (sequences and next_event from normal logs)
        X_normal = torch.tensor(normal_df['sequences'].tolist(), dtype=torch.float32)
        y_normal = torch.tensor(normal_df['next'].tolist(), dtype=torch.long)
        
        # Create a training set and a validation set from the normal data
        # A validation set is crucial to monitor for overfitting
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_normal, y_normal, test_size=0.2, random_state=42
        )
        
        # The test set should represent the real world, containing both normal and abnormal data
        self.X_test = torch.tensor(df['sequences'].tolist(), dtype=torch.float32)
        self.y_test = torch.tensor(df['next'].tolist(), dtype=torch.long)

    def train_dataloader(self):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        val_dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4)