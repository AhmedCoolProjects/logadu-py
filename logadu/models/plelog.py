import torch
import torch.nn as nn
import torch.nn.functional as F


class PLELogModel(nn.Module):
    """
    The core neural network architecture for PLELog.
    It consists of a bidirectional GRU followed by an attention mechanism.
    """
    def __init__(self, input_dim, hidden_dim=100, n_layers=1, attention_dim=200):
        super(PLELogModel, self).__init__()
        
        # Bidirectional GRU Layer
        # Note: The repository uses bidirectional=True. This is important.
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )

        # Attention Mechanism
        # The GRU output for each timestep is hidden_dim * 2 (forward + backward)
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
        # Final Classifier
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            Tensor: Logits of shape (batch_size, 1)
        """
        # gru_outputs shape: (batch_size, sequence_length, hidden_dim * 2)
        gru_outputs, _ = self.gru(x)

        # --- Attention Calculation ---
        # e shape: (batch_size, sequence_length, 1)
        e = self.attention_layer(gru_outputs)
        
        # alpha shape: (batch_size, sequence_length, 1)
        # These are the attention weights
        alpha = F.softmax(e, dim=1)

        # Apply attention weights to the GRU outputs
        # context shape: (batch_size, 1, hidden_dim * 2)
        context = torch.bmm(alpha.transpose(1, 2), gru_outputs)
        
        # context shape after squeeze: (batch_size, hidden_dim * 2)
        context = context.squeeze(1)

        # --- Classification ---
        # logits shape: (batch_size, 1)
        logits = self.classifier(context)
        
        return logits
