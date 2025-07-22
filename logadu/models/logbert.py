# /logadu/models/logbert.py

import torch
import torch.nn as nn

class LogBERT(nn.Module):
    """
    LogBERT model, adapted for the self-supervised framework from Guo et al.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size=256, num_layers=4, num_attention_heads=4, max_seq_len=512):
        super(LogBERT, self).__init__()

        # Embedding layer for log keys
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Positional encoding to understand the order of logs
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))

        # Core Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_attention_heads,
            dim_feedforward=hidden_size, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head for the Masked Log Key Prediction (MLKP) task
        self.mlm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        embeddings = self.embedding(x)
        embeddings += self.positional_encoding[:, :x.size(1), :]

        transformer_output = self.transformer_encoder(embeddings)
        # output shape: (batch_size, sequence_length, embedding_dim)
        
        # For MLKP, we output the logits for every position in the sequence
        mlm_logits = self.mlm_head(transformer_output)
        
        # For VHM, we use the output of the first token ([DIST])
        # as a representation of the entire sequence.
        dist_output = transformer_output[:, 0, :]
        
        # Return both outputs in a dictionary for clarity
        return {
            "mlm_logits": mlm_logits,   # For Masked Log Key Prediction
            "dist_output": dist_output # For Hypersphere Minimization
        }