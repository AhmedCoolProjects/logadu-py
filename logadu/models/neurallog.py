import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # The input to this module is often (seq_len, batch, dim), but our model uses
        # (batch, seq_len, dim), so we transpose before and after.
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1)
    
class NeuralLogModel(nn.Module):
    """
    The core neural network architecture for NeuralLog.
    It uses a Transformer Encoder to process sequences of vectors.
    """
    def __init__(self, input_dim: int, n_head: int, hidden_dim: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=input_dim, dropout=dropout)

        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True # Crucial for our input shape
        )

        # Stacking the layers to create the full encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Final classifier layer
        self.classifier = nn.Linear(input_dim, 1)
        self.input_dim = input_dim

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            Tensor: Logits of shape (batch_size, 1)
        """
        # 1. Add positional encoding
        x = self.pos_encoder(x)

        # 2. Pass through the transformer encoder
        transformer_output = self.transformer_encoder(x)

        # 3. Pool the output of the transformer
        # We'll use mean pooling over the sequence dimension, which is a common strategy.
        pooled_output = torch.mean(transformer_output, dim=1)

        # 4. Classify the pooled output
        logits = self.classifier(pooled_output)

        return logits