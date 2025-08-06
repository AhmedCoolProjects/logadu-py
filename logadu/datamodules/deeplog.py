
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import ast


class DeepLogDataModule(pl.LightningDataModule):
    def __init__(self, dataset_file: str, split_method: int, window_size: int = 10, batch_size: int = 128, remove_duplicates: bool = True):
        super().__init__()
        self.dataset_file = dataset_file
        self.split_method = split_method
        self.window_size = window_size
        self.batch_size = batch_size
        self.vocab_size = None
        self.remove_duplicates = remove_duplicates
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        
        
    def setup(self, stage=None):
        df = pd.read_csv(self.dataset_file)
        # if split_method == 1: then we have 'sequence', 'next', and 'label' columns
        # if split_method == 2: then we have 'timestamp', 'content', 'labels', 'rules', 'source_file', 'label', 'EventId', 'EventTemplate', 'LineId'
        if self.split_method == 1:
            if self.remove_duplicates:
                df = df.drop_duplicates(subset=['sequence'], keep='first')
            df['sequence'] = df['sequence'].apply(ast.literal_eval)
            all_keys = set(df['next'].unique())
            for seq in df['sequence']:
                all_keys.update(seq)
            self.vocab_size = int(max(all_keys) + 1) # suggesting the indexing starts from 0 and increments by 1  
        
            train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
            normal_train_val_df = train_val_df[train_val_df['label'] == 0]
            train_df, val_df = train_test_split(normal_train_val_df, test_size=0.1, random_state=42)
        if self.split_method == 2:
            train_df, val_df, test_df, self.vocab_size = self._generate_sequences(df, self.window_size)
        
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
        return DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.X_val, self.y_val), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.X_test, self.y_test_next, self.y_test_label), batch_size=self.batch_size)

    def _generate_sequences(self, df, window_size):
        # 1. for each event, create index integer incrementing from 0
        df['index'] = df['EventId'].astype('category').cat.codes
        all_keys = set(df['index'].unique())
        vocab_size = int(max(all_keys) + 1)  # assuming indexing starts from 0 and increments by 1
        # 2. split into train, and test sets 0.2, we know each index has a label
        # FIXME: DONE, Data Snooping via Random Splitting
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)
        # 3. let's define the sequences and next events where sequences are lists of indices with length window_size and next is the next index after the sequence, the label of a sequence is 1 if any of the events in the sequence is anomalous, otherwise 0
        sequences = []
        next_events = []
        labels = []
        for i in range(len(train_df) - window_size):
            sequence = train_df['index'].iloc[i:i + window_size].tolist()
            next_event = train_df['index'].iloc[i + window_size]
            sequences.append(sequence)
            next_events.append(next_event)
            labels.append(1 if train_df['label'].iloc[i:i + window_size].any() else 0)
        normal_train_df = pd.DataFrame({
            'sequence': sequences,
            'next': next_events,
            'label': labels
        })
        normal_train_df = normal_train_df[normal_train_df['label'] == 0]
        
        # 4. for validation and test sets, we will use the same logic but with step_size=window_size for validation and test sets
        val_sequences = []
        val_next_events = []
        val_labels = []
        for i in range(0, len(val_df) - window_size, window_size):
            sequence = val_df['index'].iloc[i:i + window_size].tolist()
            next_event = val_df['index'].iloc[i + window_size]
            val_sequences.append(sequence)
            val_next_events.append(next_event)
            val_labels.append(1 if val_df['label'].iloc[i:i + window_size].any() else 0)
        normal_val_df = pd.DataFrame({
            'sequence': val_sequences,
            'next': val_next_events,
            'label': val_labels
        })
        normal_val_df = normal_val_df[normal_val_df['label'] == 0]
        
        # 5. for test set, we will use the same logic but with step_size=window_size for test set and we will keep both normal and anomalous events
        test_sequences = []
        test_next_events = []
        test_labels = []
        for i in range(0, len(test_df) - window_size, window_size):
            sequence = test_df['index'].iloc[i:i + window_size].tolist()
            next_event = test_df['index'].iloc[i + window_size]
            test_sequences.append(sequence)
            test_next_events.append(next_event)
            test_labels.append(test_df['label'].iloc[i + window_size])
        test_df = pd.DataFrame({
            'sequence': test_sequences,
            'next': test_next_events,
            'label': test_labels
        })
        return normal_train_df, normal_val_df, test_df, vocab_size