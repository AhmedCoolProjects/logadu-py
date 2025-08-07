
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import ast
import os
from sklearn.model_selection import TimeSeriesSplit


class DeepLogDataModule(pl.LightningDataModule):
    def __init__(self, dataset_file: str, split_method: int, window_size: int = 10, batch_size: int = 128, remove_duplicates: bool = True, n_splits: int = 5, fold_index: int = 0):
        super().__init__()
        self.num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
        self.dataset_file = dataset_file
        self.split_method = split_method
        self.window_size = window_size
        self.batch_size = batch_size
        self.vocab_size = None
        self.remove_duplicates = remove_duplicates
        self.n_splits = n_splits
        self.fold_index = fold_index
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
            train_df, val_df, test_df, self.vocab_size = self._generate_sequences(df)
        if self.split_method == 3:
            train_df, val_df, test_df, self.vocab_size = self._generate_sequences(df, normal_seq=False)
        if self.split_method == 4:
            train_df, val_df, test_df, self.vocab_size = self._generate_sequences_timeseries_cv(df)

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
        return DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.X_val, self.y_val), batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.X_test, self.y_test_next, self.y_test_label), batch_size=self.batch_size, num_workers=self.num_workers)

    def _generate_sequences(self, df, normal_seq: bool = True):
        # 1. for each event, create index integer incrementing from 0
        df['index'] = df['EventId'].astype('category').cat.codes
        all_keys = set(df['index'].unique())
        vocab_size = int(max(all_keys) + 1)  # assuming indexing starts from 0 and increments by 1
        # 2. split into train, and test sets 0.2, we know each index has a label
        ## FIXME 2: DONE, Use a chronological split, NOT a random split
        n_total = len(df)
        dev_end_index = int(n_total * 0.8)  # 80% for train/val

        development_df = df.iloc[:dev_end_index]
        test_df_raw = df.iloc[dev_end_index:]
        
        # 3. Generate sequences for the test set (from the test split)
        test_sequences, test_next_events, test_labels = [], [], []
        for i in range(0, len(test_df_raw) - self.window_size, self.window_size):
            sequence = test_df_raw['index'].iloc[i:i + self.window_size].tolist()
            next_event = test_df_raw['index'].iloc[i + self.window_size]
            ## FIXME 3: DONE, Correctly label test sequences using .any()
            label = 1 if test_df_raw['label'].iloc[i:i + self.window_size].any() else 0
            
            test_sequences.append(sequence)
            test_next_events.append(next_event)
            test_labels.append(label)
            
        test_df = pd.DataFrame({'sequence': test_sequences, 'next': test_next_events, 'label': test_labels})
        
        train_sequences, train_next_events, train_labels = [], [], []
        val_sequences, val_next_events, val_labels = [], [], []
        
        # 4. Generate sequences for train and val sets, using ONLY NORMAL EVENTS
        if normal_seq: # Method 2
            ## FIX 4: More efficient - filter for normal data BEFORE generating sequences
            normal_dev_events = development_df[development_df['label'] == 0]['index'].tolist()
            
            train_end_index = int(len(normal_dev_events) * 0.9) # e.g., 90% of normal data for train
            normal_train_events = normal_dev_events[:train_end_index]
            normal_val_events = normal_dev_events[train_end_index:]

            for i in range(len(normal_train_events) - self.window_size):
                train_sequences.append(normal_train_events[i:i + self.window_size])
                train_next_events.append(normal_train_events[i + self.window_size])
            for i in range(0, len(normal_val_events) - self.window_size, self.window_size):
                val_sequences.append(normal_val_events[i:i + self.window_size])
                val_next_events.append(normal_val_events[i + self.window_size])
            
            train_labels = [0] * len(train_sequences)  # All train sequences are normal
            val_labels = [0] * len(val_sequences)      # All val sequences are normal
        else:
            train_val_end = int(n_total * 0.8)
            train_val_df = df.iloc[:train_val_end]
            
            train_end = int(len(train_val_df) * 0.9) # 90% of the train/val chunk is for training
            train_df_raw = train_val_df.iloc[:train_end]
            val_df_raw = train_val_df.iloc[train_end:]
            for i in range(len(train_df_raw) - self.window_size):
                train_sequences.append(train_df_raw['index'].iloc[i:i + self.window_size].tolist())
                train_next_events.append(train_df_raw['index'].iloc[i + self.window_size])
            for i in range(0, len(val_df_raw) - self.window_size, self.window_size):
                val_sequences.append(val_df_raw['index'].iloc[i:i + self.window_size].tolist())
                val_next_events.append(val_df_raw['index'].iloc[i + self.window_size])
            
            # for labels, we can use the same logic as above
            train_labels = [1 if train_df_raw['label'].iloc[i:i + self.window_size].any() else 0 for i in range(len(train_df_raw) - self.window_size)]
            val_labels = [1 if val_df_raw['label'].iloc[i:i + self.window_size].any() else 0 for i in range(0, len(val_df_raw) - self.window_size, self.window_size)]
            
            
            
        train_df = pd.DataFrame({'sequence': train_sequences, 'next': train_next_events, 'label': train_labels})
        val_df = pd.DataFrame({'sequence': val_sequences, 'next': val_next_events, 'label': val_labels})
        
        train_df = train_df[train_df['label'] == 0].reset_index(drop=True)
        val_df = val_df[val_df['label'] == 0].reset_index(drop=True)
        
        if val_df.empty:
            raise ValueError("Validation set is empty. This can happen if the initial 80% of your data contains very few normal logs. "
                             "Consider adjusting the train/test split percentage.")

        return train_df, val_df, test_df, vocab_size
    
    
    def _generate_sequences_timeseries_cv(self, df):
        df['index'] = df['EventId'].astype('category').cat.codes
        vocab_size = len(df['index'].unique())
        
        # 1. BEST PRACTICE: Create a final, hold-out test set that is NEVER touched during CV.
        n_total = len(df)
        dev_end_index = int(n_total * 0.8) # 80% for development (CV)
        
        development_df = df.iloc[:dev_end_index]
        hold_out_test_df = df.iloc[dev_end_index:]
    
        # 2. Use TimeSeriesSplit on the development set
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        all_splits = list(tscv.split(development_df))
        
        # Select the specific fold for this DataModule instance, all_splits is a list of (train_indices, val_indices) tuples, its length is self.n_splits
        train_indices, val_indices = all_splits[self.fold_index]
        
        train_df_raw = development_df.iloc[train_indices]
        val_df_raw = development_df.iloc[val_indices]

        # 3. Generate sequences for the train set (normal data only, step=1)
        normal_train_events = train_df_raw[train_df_raw['label'] == 0]['index'].tolist()
        train_sequences, train_next_events, train_labels = [], [], []
        for i in range(len(normal_train_events) - self.window_size):
            train_sequences.append(normal_train_events[i:i + self.window_size])
            train_next_events.append(normal_train_events[i + self.window_size])
        train_labels = [0] * len(train_sequences)
        train_df = pd.DataFrame({'sequence': train_sequences, 'next': train_next_events, 'label': train_labels})

        # 4. Generate sequences for the validation set (this fold's "test" set, normal only)
        normal_val_events = val_df_raw[val_df_raw['label'] == 0]['index'].tolist()
        val_sequences, val_next_events, val_labels = [], [], []
        for i in range(0, len(normal_val_events) - self.window_size, self.window_size):
            val_sequences.append(normal_val_events[i:i + self.window_size])
            val_next_events.append(normal_val_events[i + self.window_size])
        val_labels = [0] * len(val_sequences)
        val_df = pd.DataFrame({'sequence': val_sequences, 'next': val_next_events, 'label': val_labels})
        
        # 5. Generate sequences for the final hold-out test set
        test_sequences, test_next_events, test_labels = [], [], []
        for i in range(0, len(hold_out_test_df) - self.window_size, self.window_size):
            sequence = hold_out_test_df['index'].iloc[i:i + self.window_size].tolist()
            next_event = hold_out_test_df['index'].iloc[i + self.window_size]
            label = 1 if hold_out_test_df['label'].iloc[i:i + self.window_size].any() else 0
            test_sequences.append(sequence)
            test_next_events.append(next_event)
            test_labels.append(label)
        test_df = pd.DataFrame({'sequence': test_sequences, 'next': test_next_events, 'label': test_labels})
        
        return train_df, val_df, test_df, vocab_size