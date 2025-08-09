# READY TO USE ====================
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import os
import click

class IndexDataModule(pl.LightningDataModule):
    """
    If Shuffle, then no chronological order is preserved between list of sequences.
    If Remove Duplicates, then after sequencing, no two sequences will be the same with the same next event.
    If Label Type is 'next', then the next event is used as the label.
    """
    def __init__(self, dataset_file: str, split_method: int = 1, window_size: int = 10, batch_size: int = 128, remove_duplicates: bool = True, label_type: str = 'next', shuffle: bool = True):
        super().__init__()
        self.num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
        self.dataset_file = dataset_file
        self.split_method = split_method
        self.window_size = window_size
        self.batch_size = batch_size
        self.vocab_size = None
        self.remove_duplicates = remove_duplicates
        self.shuffle = shuffle
        self.label_type = label_type  # 'next' for predicting next event, 'label' for binary classification
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        
        
    def setup(self, stage=None):
        df = pd.read_csv(self.dataset_file, low_memory=False)

        if self.split_method == 1:
            if self.shuffle:
                train_df, val_df, test_df, self.vocab_size = self._generate_sequences_dedup(df)
            else:
                train_df, val_df, test_df, self.vocab_size = self._generate_sequences_chronological(df)
        else:
            raise ValueError(f"Invalid split method: {self.split_method}. Supported methods are 1 Only for now.")

        # Train set (normal only)
        self.X_train = torch.tensor(train_df['sequence'].tolist(), dtype=torch.long)
        self.y_train = torch.tensor(train_df['next'].tolist(), dtype=torch.long)
        # Validation set (normal only)
        self.X_val = torch.tensor(val_df['sequence'].tolist(), dtype=torch.long)
        self.y_val = torch.tensor(val_df['next'].tolist(), dtype=torch.long)
        # Test set (contains both normal and anomalous data)
        self.X_test = torch.tensor(test_df['sequence'].tolist(), dtype=torch.long)
        self.y_test_next = torch.tensor(test_df['next'].tolist(), dtype=torch.long)
        self.y_test_label = torch.tensor(test_df['label'].tolist(), dtype=torch.long)
        
    def train_dataloader(self):
        if self.label_type == 'next':
            return DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.label_type == 'next':
            return DataLoader(TensorDataset(self.X_val, self.y_val), batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.label_type == 'next':
            return DataLoader(TensorDataset(self.X_test, self.y_test_next, self.y_test_label), batch_size=self.batch_size, num_workers=self.num_workers)



    def _generate_sequences_chronological(self, df):
        df['index'] = df['EventId'].astype('category').cat.codes
        vocab_size = len(df['index'].unique())
        
        # --- 1. Generate ALL sequences chronologically FIRST ---
        sequences, next_events, labels = [], [], []
        for i in range(len(df) - self.window_size):
            sequences.append(df['index'].iloc[i:i + self.window_size].tolist())
            next_events.append(df['index'].iloc[i + self.window_size])
            labels.append(1 if df['label'].iloc[i:i + self.window_size].any() else 0)
            
        data = pd.DataFrame({'sequence': sequences, 'next': next_events, 'label': labels})

        # --- 2. (Optional) Deduplicate ---
        if self.remove_duplicates:
            click.secho("Applying deduplication to sequences...", fg="yellow")
            data['seq_next_key'] = data.apply(lambda row: f"{row['sequence']}_{row['next']}", axis=1)
            data = data.drop_duplicates(subset=['seq_next_key'], keep='first')
            data = data.drop(columns=['seq_next_key']).reset_index(drop=True)
            click.echo(f"Number of sequences after deduplication: {len(data)}")

        # --- 3. Perform a CHRONOLOGICAL split on the (possibly deduplicated) sequences ---
        n_total = len(data)
        train_val_end = int(n_total * 0.8)

        # train val df should contain only normal events
        train_val_df = data.iloc[:train_val_end]
        normal_train_val_df = train_val_df[train_val_df['label'] == 0]
        val_end = int(len(normal_train_val_df) * 0.9)
        # split
        train_df = normal_train_val_df.iloc[:val_end]
        val_df = normal_train_val_df.iloc[val_end:]
        test_df = data.iloc[train_val_end:]
        
        return train_df, val_df, test_df, vocab_size


    def _generate_sequences_dedup(self, df):
        df['index'] = df['EventId'].astype('category').cat.codes
        vocab_size = len(df['index'].unique())
        # 1. create sequences with step=1, next event, and label of the sequence
        sequences, next_events, labels = [], [], []
        for i in range(len(df) - self.window_size):
            sequence = df['index'].iloc[i:i + self.window_size].tolist()
            next_event = df['index'].iloc[i + self.window_size]
            label = 1 if df['label'].iloc[i:i + self.window_size].any() else 0
            
            sequences.append(sequence)
            next_events.append(next_event)
            labels.append(label)
        # 2. create a DataFrame with the sequences, next events, and labels
        data = pd.DataFrame({'sequence': sequences, 'next': next_events, 'label': labels})

        
        # --- 2. (Optional) Deduplicate ---
        if self.remove_duplicates:
            click.secho("Applying deduplication to sequences...", fg="yellow")
            data['seq_next_key'] = data.apply(lambda row: f"{row['sequence']}_{row['next']}", axis=1)
            data = data.drop_duplicates(subset=['seq_next_key'], keep='first')
            data = data.drop(columns=['seq_next_key']).reset_index(drop=True)
            click.echo(f"Number of sequences after deduplication: {len(data)}")
        
        
        # 4. split into train, val, and test sets
        train_val_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
        normal_train_val_df = train_val_df[train_val_df['label'] == 0]
        train_df, val_df = train_test_split(normal_train_val_df, test_size=0.1, random_state=42)
        # 5. reset index
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        # 6. return the DataFrames and vocab_size
        return train_df, val_df, test_df, vocab_size