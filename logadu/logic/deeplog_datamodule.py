import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import ast

class DeepLogDataModule(pl.LightningDataModule):
    def __init__(self, dataset_file: str, batch_size: int = 128):
        super().__init__()
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.vocab_size = None

    def setup(self, stage=None):
        df = pd.read_csv(self.dataset_file)
        df['sequence'] = df['sequence'].apply(ast.literal_eval)

        all_keys = set(df['next'].unique())
        for seq in df['sequence']: all_keys.update(seq)
        self.vocab_size = int(max(all_keys) + 1)

        # Train only on normal data
        train_df = df[df['label'] == 0]
        
        # We need a validation set for Lightning
        train_normal_df, val_normal_df = train_test_split(train_df, test_size=0.1, random_state=42)

        self.X_train = torch.tensor(train_normal_df['sequence'].tolist(), dtype=torch.long)
        self.y_train = torch.tensor(train_normal_df['next'].tolist(), dtype=torch.long)

        self.X_val = torch.tensor(val_normal_df['sequence'].tolist(), dtype=torch.long)
        self.y_val = torch.tensor(val_normal_df['next'].tolist(), dtype=torch.long)

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.X_val, self.y_val), batch_size=self.batch_size)