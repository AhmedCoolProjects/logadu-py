# /logadu/models/neurallog.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """ Sinusoidal positional encoding for Transformer. """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class NeuralLog(nn.Module):
    """
    NeuralLog model: A Transformer Encoder for classifying sequences of semantic vectors.
    """
    def __init__(self, input_dim, hidden_dim=2048, num_layers=1, num_attention_heads=12):
        super(NeuralLog, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=input_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_attention_heads,
            dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
        transformer_output = self.transformer_encoder(x)
        
        # Use mean pooling over the sequence dimension
        pooled_output = transformer_output.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits