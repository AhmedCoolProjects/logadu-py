# In logadu/deep_learning/dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import Counter
from pathlib import Path

class DeepLogDataset(Dataset):
    """
    Custom PyTorch Dataset for DeepLog.
    It loads sequences from a .pkl file, where each sequence is a list of string EventIDs.
    It builds an integer vocabulary and generates sliding windows for forecasting.
    """
    def __init__(self, sequences_file_path, vocab=None):
        """
        Args:
            sequences_file_path (str): Path to the .pkl file containing session data.
            vocab (dict, optional): A pre-built vocabulary. If None, a new one is created.
        """
        file_name = Path(sequences_file_path).stem
        window_size = int(file_name.split('_')[1])
        print(f"Loading data from {file_name}...")
        with open(sequences_file_path, 'rb') as f:
            session_data = pickle.load(f)
        
        # We only train DeepLog on normal sequences
        self.sequences = [data['sequence'] for data in session_data.values() if data['label'] == 0]
        print(f"Loaded {len(self.sequences)} normal sequences for training.")

        self.window_size = window_size
        
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        # Add a token for unknown events that might appear in test data
        self.vocab['<UNK>'] = 0
        
        self.id_sequences = self._convert_sequences_to_ids()
        
        print("Generating training windows...")
        self.windows, self.labels = self._generate_windows()
        print(f"Created {len(self.windows)} training instances (windows).")

    def _build_vocab(self):
        """Builds a vocabulary from all event IDs in the normal sequences."""
        all_events = [event for seq in self.sequences for event in seq]
        event_counts = Counter(all_events)
        # Create vocab, reserving index 0 for the <UNK> token
        vocab = {event: i + 1 for i, (event, _) in enumerate(event_counts.most_common())}
        return vocab

    def _convert_sequences_to_ids(self):
        """Converts sequences of string EventIDs to sequences of integer indices."""
        id_sequences = []
        for seq in self.sequences:
            id_seq = [self.vocab.get(event, self.vocab['<UNK>']) for event in seq]
            id_sequences.append(id_seq)
        return id_sequences
    
    def _generate_windows(self):
        """
        Generates input windows and corresponding next-event labels for the LSTM.
        This is the core data preparation step for DeepLog's forecasting task.
        """
        windows = []
        labels = []
        for seq in self.id_sequences:
            # A sequence must be at least window_size + 1 long to have one training sample
            if len(seq) < self.window_size + 1:
                continue
            # Slide a window across each sequence
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

def get_deeplog_dataloader(sequences_file, batch_size, shuffle=True, num_workers=0):
    """
    High-level function to create a DataLoader for the DeepLog model.
    
    Returns:
        DataLoader: The PyTorch DataLoader instance.
        int: The total number of unique labels (vocabulary size) for the model's output layer.
    """
    dataset = DeepLogDataset(sequences_file)
    
    # The model needs to predict any of the tokens in the vocabulary.
    # The size is len(vocab) because we added the <UNK> token.
    num_labels = len(dataset.vocab)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True # Optimizes data transfer to GPU
    )
    return dataloader, num_labels