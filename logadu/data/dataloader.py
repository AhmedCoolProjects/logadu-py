# In logadu/deep_learning/dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import Counter
from pathlib import Path

class DeepLogDataset(Dataset):
    """
    Custom PyTorch Dataset for DeepLog.
    It loads sequences, builds a vocabulary, and generates training windows.
    """
    # CORRECTED: window_size is now an explicit argument again.
    def __init__(self, sequences_file_path, window_size, vocab=None):
        """
        Args:
            sequences_file_path (str): Path to the .pkl file containing session data.
            window_size (int): The size of the sliding window to be used for model training.
            vocab (dict, optional): A pre-built vocabulary. If None, a new one is created.
        """
        print(f"Loading data from {Path(sequences_file_path).name}...")
        with open(sequences_file_path, 'rb') as f:
            session_data = pickle.load(f)
        
        # We only train DeepLog on normal sequences
        self.sequences = [data['sequence'] for data in session_data.values() if data['label'] == 0]
        print(f"Loaded {len(self.sequences)} normal sequences for training.")

        # This is the model's training window size, passed as an argument.
        self.window_size = window_size
        
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        self.vocab['<UNK>'] = 0
        
        self.id_sequences = self._convert_sequences_to_ids()
        
        print(f"Generating training windows of size {self.window_size}...")
        self.windows, self.labels = self._generate_windows()
        print(f"Created {len(self.windows)} training instances.")

    def _build_vocab(self):
        all_events = [event for seq in self.sequences for event in seq]
        event_counts = Counter(all_events)
        vocab = {event: i + 1 for i, (event, _) in enumerate(event_counts.most_common())}
        return vocab

    def _convert_sequences_to_ids(self):
        id_sequences = []
        for seq in self.sequences:
            id_seq = [self.vocab.get(event, self.vocab['<UNK>']) for event in seq]
            id_sequences.append(id_seq)
        return id_sequences
    
    def _generate_windows(self):
        # This function is correct and does not need changes.
        # It creates the (input, target) pairs for the forecasting task.
        windows = []
        labels = []
        for seq in self.id_sequences: 
            if len(seq) < self.window_size + 1:
                continue
            for i in range(len(seq) - self.window_size):
                input_window = seq[i : i + self.window_size]
                target_label = seq[i + self.window_size]
                windows.append(torch.tensor(input_window, dtype=torch.long))
                labels.append(target_label)
        return windows, labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

# CORRECTED: Re-added window_size as an argument.
def get_deeplog_dataloader(sequences_file, window_size, batch_size, shuffle=True, num_workers=0):
    """ High-level function to create a DataLoader for the DeepLog model. """
    dataset = DeepLogDataset(sequences_file, window_size)
    num_labels = len(dataset.vocab)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, num_labels, dataset.vocab