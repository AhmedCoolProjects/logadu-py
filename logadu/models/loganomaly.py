import torch
import torch.nn as nn

class LogAnomalyModel(nn.Module):
    def __init__(self, semantic_input_dim, num_templates, hidden_size=128, num_layers=2):
        super(LogAnomalyModel, self).__init__()
        
        # --- Stream 1: Sequential Model ---
        # Takes a sequence of semantic vectors
        self.sequential_lstm = nn.LSTM(
            input_size=semantic_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # --- Stream 2: Quantitative Model ---
        # Takes a sequence of count vectors
        self.quantitative_lstm = nn.LSTM(
            input_size=num_templates,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # --- Prediction Heads ---
        # We need two output layers, one to predict the next semantic vector
        # and one to predict the next log template index (from the counts).
        
        # Predicts the next template's index (classification over all known templates)
        self.quantitative_head = nn.Linear(hidden_size, num_templates)
        
        # Predicts the next template's semantic vector (regression)
        self.sequential_head = nn.Linear(hidden_size, semantic_input_dim)

    def forward(self, seq_vectors, count_vectors):
        """
        The forward pass takes two inputs.
        - seq_vectors: (batch, seq_len, semantic_input_dim)
        - count_vectors: (batch, seq_len, num_templates)
        """
        
        # Process sequential data
        # We only need the output of the last LSTM cell
        _, (seq_hidden, _) = self.sequential_lstm(seq_vectors)
        
        # Process quantitative data
        _, (quant_hidden, _) = self.quantitative_lstm(count_vectors)
        
        # Get the hidden state from the last layer
        # Shape: (batch, hidden_size)
        last_seq_hidden = seq_hidden[-1]
        last_quant_hidden = quant_hidden[-1]
        
        # --- Make Predictions ---
        # The paper suggests combining them, but a simpler start is to predict separately
        # and combine the losses. A more advanced approach would concatenate the hidden
        # states before a final prediction layer.
        
        # Predict the probability distribution for the next template index
        next_template_logits = self.quantitative_head(last_quant_hidden)
        
        # Predict the semantic vector of the next template
        next_semantic_vector = self.sequential_head(last_seq_hidden)
        
        return next_template_logits, next_semantic_vector