# deeplog.py
# This file is adapted from the repository: https://github.com/cqu-isse/Impact_evalution/blob/master/DL_loglizer/deeplog.py
# The local import `_base_model` has been included directly in this file to make it standalone.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# =============================================================================
# LOCAL IMPORT REPLACEMENT: The _base_model class from loglizer.models
# This class was originally in a separate file and is included here for portability.
# =============================================================================
class _base_model(object):
    """
    Abstract base model for all log-based anomaly detection models.
    """
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y_true):
        pass


# =============================================================================
# DEEPLOG MODEL DEFINITION
# This is the main DeepLog class, inheriting from the base model above.
# =============================================================================

class DeepLog(nn.Module):
    """
    The DeepLog model implemented as a PyTorch nn.Module.
    This is the core neural network.
    
    Args:
        input_size (int): The number of features for each log key (typically 1).
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of recurrent layers (e.g., stacked LSTMs).
        num_keys (int): The total number of unique log keys (the vocabulary size).
    """
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The LSTM layer takes sequences of log keys and outputs hidden states.
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        
        # The fully connected layer maps the LSTM output to the vocabulary space
        # to predict the next log key.
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tensor: The output of the model, which are the logits for the next key prediction.
                    Shape: (batch_size, num_keys)
        """
        # Initialize hidden state and cell state with zeros
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We pass the input and hidden/cell states to the LSTM.
        # `out` will contain the output features from the last time step for each sequence.
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output of the last time step for prediction.
        # Shape: (batch_size, hidden_size)
        last_time_step_out = out[:, -1, :]
        
        # Pass the last time step's output to the fully connected layer.
        out = self.fc(last_time_step_out)
        
        return out