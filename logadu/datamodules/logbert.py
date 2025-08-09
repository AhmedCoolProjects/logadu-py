import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

class LogDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class LogDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, window_size=10, batch_size=128, train_ratio=0.8, val_ratio=0.1):
        super().__init__()
        self.csv_path = csv_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)

        # Map EventId to integer tokens
        event_ids = df["EventId"].unique()
        self.event2idx = {eid: idx+1 for idx, eid in enumerate(event_ids)}  # +1 so 0 is reserved if needed
        vocab_size = len(self.event2idx) + 1  # include reserved token id=0 if needed

        df["EventIdx"] = df["EventId"].map(self.event2idx)

        # Create sequences with sliding window
        sequences, labels = [], []
        events = df["EventIdx"].tolist()
        label_col = df["label"].tolist()

        for i in range(0, len(events) - self.window_size + 1):
            seq = events[i:i + self.window_size]
            seq_label = 1 if any(label_col[i:i + self.window_size]) else 0
            sequences.append(seq)
            labels.append(seq_label)

        sequences = torch.tensor(sequences, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # 1️⃣ Remove duplicates (sequence + label)
        combined = [(tuple(seq.tolist()), int(lbl)) for seq, lbl in zip(sequences, labels)]
        unique_combined = list(set(combined))
        sequences = torch.tensor([list(seq) for seq, _ in unique_combined], dtype=torch.long)
        labels = torch.tensor([lbl for _, lbl in unique_combined], dtype=torch.long)
        
        # 2️⃣ Split into train/test first
        seq_train, seq_test, lbl_train, lbl_test = train_test_split(
            sequences, labels, test_size=1 - self.train_ratio, stratify=labels, random_state=42
        )
        
        # 3️⃣ From train set, keep only normal sequences
        normal_idx = lbl_train == 0
        normal_sequences = seq_train[normal_idx]
        normal_labels = lbl_train[normal_idx]

        # 4️⃣ Split normal sequences into train/val
        n_val = int(len(normal_sequences) * self.val_ratio)
        train_sequences = normal_sequences[:-n_val]
        val_sequences = normal_sequences[-n_val:]

        train_labels = normal_labels[:-n_val]
        val_labels = normal_labels[-n_val:]

        # 5️⃣ Final datasets
        self.train_data = LogDataset(train_sequences, train_labels)
        self.val_data = LogDataset(val_sequences, val_labels)
        self.test_data = LogDataset(seq_test, lbl_test)

        self.vocab_size = vocab_size

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
