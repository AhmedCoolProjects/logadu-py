import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Simple Additive Attention mechanism, inspired by the LogRobust paper.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn_w = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        
        # Calculate attention scores
        score = torch.tanh(self.attn_w(lstm_output)) # (batch_size, seq_len, hidden_dim)
        score = self.attn_v(score) # (batch_size, seq_len, 1)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(score, dim=1) # (batch_size, seq_len, 1)
        
        # Calculate the context vector as a weighted sum
        context_vector = lstm_output * attention_weights
        context_vector = torch.sum(context_vector, dim=1) # (batch_size, hidden_dim)
        
        return context_vector, attention_weights


class LogRobust(nn.Module):
    """
    LogRobust model architecture: an Attention-based Bidirectional LSTM.
    This model expects pre-computed semantic vectors as input.
    """
    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.5):
        super(LogRobust, self).__init__()
        
        # The Bi-LSTM layer processes the sequences of semantic vectors
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # The Attention layer
        # The input dimension is hidden_size * 2 because the LSTM is bidirectional
        self.attention = Attention(hidden_size * 2)

        # The final classifier layer for binary classification
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Pass through the Bi-LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size * 2)

        # Apply the attention mechanism
        context_vector, _ = self.attention(lstm_out)
        # context_vector shape: (batch_size, hidden_size * 2)
        
        # Pass the final context vector to the classifier to get a single logit
        logits = self.classifier(context_vector)
        # logits shape: (batch_size, 1)

        return logits