import torch
import torch.nn as nn
import math

class LogBERT(nn.Module):
    """
    LogBERT model, adapted for the self-supervised framework from Guo et al.
    Uses a standard Transformer Encoder architecture.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=4, num_attention_heads=4, max_seq_len=512):
        super(LogBERT, self).__init__()

        # 1. Embedding Layer: Converts log key indexes into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Positional Encoding: Injects information about the order of logs in the sequence
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))

        # 3. Transformer Encoder: The core of the model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. MLM Head: A linear layer to predict masked log keys from the Transformer's output
        self.mlm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        The forward pass of the model.
        
        Args:
            x (Tensor): Input tensor of log key indexes, shape (batch_size, sequence_length)
        
        Returns:
            dict: A dictionary containing the logits for the MLM task and the sequence
                  representation for the VHM task.
        """
        # Get embeddings and add positional encoding
        embeddings = self.embedding(x)
        embeddings += self.positional_encoding[:, :x.size(1), :]

        # Pass through the main Transformer Encoder
        transformer_output = self.transformer_encoder(embeddings)
        # output shape: (batch_size, sequence_length, embedding_dim)

        # --- Prepare outputs for the two self-supervised tasks ---
        
        # Output for Masked Log Key Prediction (MLKP)
        mlm_logits = self.mlm_head(transformer_output)
        
        # Output for Volume of Hypersphere Minimization (VHM)
        # We use the output of the first token ([DIST]) as a representation of the entire sequence.
        dist_output = transformer_output[:, 0, :]
        
        return {
            "mlm_logits": mlm_logits,
            "dist_output": dist_output
        }