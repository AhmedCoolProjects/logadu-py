# /logadu/logic/logrobust_datamodule.py

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

class LogRobustDataModule(pl.LightningDataModule):
    """ DataModule for pre-vectorized LogRobust data. """
    def __init__(self, vectorized_file: str, batch_size: int = 32):
        super().__init__()
        self.vectorized_file = vectorized_file
        self.batch_size = batch_size
        self.input_dim = None # Will be determined from loaded data

    def setup(self, stage=None):
        # Load the pre-computed dictionary of tensors
        data = torch.load(self.vectorized_file)
        sequences = data['sequences']
        labels = data['labels']
        
        # Determine the input dimension from the first tensor
        if sequences:
            self.input_dim = sequences[0].shape[1]

        # Split data for training, validation, and testing
        train_seq, test_seq, train_lbl, test_lbl = train_test_split(
            sequences, labels, test_size=0.3, random_state=42, stratify=labels)
        
        train_seq, val_seq, train_lbl, val_lbl = train_test_split(
            train_seq, train_lbl, test_size=0.2, random_state=42, stratify=train_lbl)
        
        self.train_dataset = TensorDataset(pad_sequence(train_seq, batch_first=True), torch.stack(train_lbl))
        self.val_dataset = TensorDataset(pad_sequence(val_seq, batch_first=True), torch.stack(val_lbl))
        self.test_dataset = TensorDataset(pad_sequence(test_seq, batch_first=True), torch.stack(test_lbl))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)