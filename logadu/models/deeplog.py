# In logadu/deep_learning/model.py

import torch
import torch.nn as nn

class DeepLog(nn.Module):
    def __init__(self, num_labels, hidden_size=128, num_layers=2, dropout=0.5):
        """
        DeepLog LSTM model for sequence prediction.
        
        Args:
            num_labels (int): The total number of unique log templates (vocabulary size).
            hidden_size (int): The size of the LSTM hidden state.
            num_layers (int): The number of stacked LSTM layers.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(DeepLog, self).__init__()
        embedding_dim = hidden_size
        
        self.embedding = nn.Embedding(num_labels, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        # We only care about the output of the last sequence element
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out