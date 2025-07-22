# /logadu/logic/logbert_datamodule.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import ast
import random

class LogBERTDataset(Dataset):
    """ Custom Dataset for LogBERT's self-supervised pre-training. """
    def __init__(self, sequences, vocab, max_seq_len, mask_prob=0.15):
        self.sequences = sequences
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.dist_token_id = self.vocab['[DIST]']
        self.mask_token_id = self.vocab['[MASK]']

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 1. Add [DIST] token and truncate
        input_ids = [self.dist_token_id] + seq
        input_ids = input_ids[:self.max_seq_len]
        
        # 2. Prepare labels for MLM task
        mlm_labels = [-100] * len(input_ids) # -100 is ignored by CrossEntropyLoss

        # 3. Perform random masking
        for i in range(1, len(input_ids)): # Start from 1 to avoid masking [DIST]
            if random.random() < self.mask_prob:
                mlm_labels[i] = input_ids[i]  # The label is the original token
                input_ids[i] = self.mask_token_id # The input is the [MASK] token
        
        return torch.tensor(input_ids), torch.tensor(mlm_labels)

class LogBERTDataModule(pl.LightningDataModule):
    def __init__(self, dataset_file: str, batch_size: int = 32, max_seq_len: int = 512):
        super().__init__()
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vocab = None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def _build_vocab(self, df):
        all_keys = set(str(key) for seq in df['sequences'] for key in seq)
        # Note: [DIST] is used instead of [CLS] as per the paper
        vocab = {"<pad>": 0, "[DIST]": 1, "[MASK]": 2}
        for key in sorted(list(all_keys)):
            if key not in vocab:
                vocab[key] = len(vocab)
        return vocab

    def setup(self, stage=None):
        df = pd.read_csv(self.dataset_file)
        df['sequences'] = df['sequences'].apply(ast.literal_eval)

        self.vocab = self._build_vocab(df)
        
        # --- CRITICAL: Train and validate ONLY on normal data ---
        normal_df = df[df['label'] == 0]
        
        # Use the full dataset for testing
        test_sequences = df['sequences'].tolist()
        test_labels = df['label'].tolist()
        
        train_seq, val_seq = train_test_split(normal_df['sequences'].tolist(), test_size=0.1, random_state=42)
        
        self.train_dataset = LogBERTDataset(train_seq, self.vocab, self.max_seq_len)
        self.val_dataset = LogBERTDataset(val_seq, self.vocab, self.max_seq_len)
        # Test dataset will be used by the predictor, not the trainer
        self.test_sequences = test_sequences
        self.test_labels = test_labels

    def _collate_fn(self, batch):
        inputs, labels = zip(*batch)
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.vocab["<pad>"])
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return padded_inputs, padded_labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self._collate_fn)