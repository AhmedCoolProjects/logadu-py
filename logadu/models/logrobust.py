# /logadu/models/logrobust.py

import torch
import torch.nn as nn

class LogRobust(nn.Module):
    """
    LogRobust model architecture for pre-vectorized input.
    """
    # --- MODIFIED SIGNATURE: No vocab_size needed ---
    def __init__(self, input_dim, hidden_size, num_layers, dropout=0.5):
        super(LogRobust, self).__init__()
        
        # The nn.Embedding layer is REMOVED.
        
        # The LSTM now takes the pre-computed embedding dimension as its input_size.
        self.lstm = nn.LSTM(
            input_dim, # CHANGED from embedding_dim
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Linear(hidden_size * 2, 1, bias=False)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # The embedding step is gone.
        lstm_out, _ = self.lstm(x)
        
        attn_weights = torch.tanh(self.attention(lstm_out))
        attn_weights = torch.softmax(attn_weights, dim=1)
        attended_out = lstm_out * attn_weights
        context_vector = torch.sum(attended_out, dim=1)
        logits = self.classifier(context_vector)
        

        return logits