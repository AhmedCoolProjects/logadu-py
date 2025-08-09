import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LogCNN(nn.Module):
    """
    A CNN-based model for log anomaly detection, inspired by Lu et al. (2018).

    This model is designed for a forecasting task: given a sequence of log keys,
    it predicts the next log key in the sequence.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 128, kernel_sizes: List[int] = [3, 4, 5]):
        """
        Args:
            vocab_size (int): The total number of unique log keys.
            embedding_dim (int): The size of the learned vector for each log key.
            hidden_size (int): The number of output channels for each convolutional filter.
            kernel_sizes (List[int]): A list of kernel sizes (n-gram sizes) for the parallel conv layers.
        """
        super(LogCNN, self).__init__()

        # 1. Embedding Layer: Converts log key indexes into dense vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. Parallel Convolutional Layers
        # This ModuleList holds a separate 1D convolutional layer for each kernel size.
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_size, kernel_size=k)
            for k in kernel_sizes
        ])

        # 3. Final Fully-Connected Layer for Classification
        # The input size is the number of filters * hidden_size, as we concatenate their outputs.
        self.fc = nn.Linear(len(kernel_sizes) * hidden_size, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        The forward pass of the model.
        
        Args:
            x (Tensor): Input tensor of log key indexes, shape: (batch_size, sequence_length)
        
        Returns:
            Tensor: Logits (raw scores) for the next key prediction, shape: (batch_size, vocab_size)
        """
        # 1. Apply embedding
        embedded = self.embedding(x)
        # -> Shape: (batch_size, sequence_length, embedding_dim)
        
        # 2. Prepare for Conv1d
        # PyTorch's Conv1d expects input of shape (batch_size, channels, length).
        # We permute the dimensions so that the embedding dimension acts as the "channel".
        embedded = embedded.permute(0, 2, 1)
        # -> Shape: (batch_size, embedding_dim, sequence_length)

        # 3. Apply parallel convolutions and pooling
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # -> conved is a list of tensors, each with shape: (batch_size, hidden_size, new_length)

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # -> pooled is a list of tensors, each with shape: (batch_size, hidden_size)

        # 4. Concatenate and classify
        concatenated = torch.cat(pooled, dim=1)
        # -> concatenated shape: (batch_size, len(kernel_sizes) * hidden_size)
        
        logits = self.fc(concatenated)
        # -> logits shape: (batch_size, vocab_size)

        return logits