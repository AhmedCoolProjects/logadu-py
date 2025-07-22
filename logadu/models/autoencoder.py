# /logadu/models/autoencoder.py

import torch.nn as nn

class AutoEncoder(nn.Module):
    """
    A simple feed-forward Autoencoder model.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()

        # The Encoder part compresses the input into a smaller latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim) # The "bottleneck" layer
        )

        # The Decoder part tries to reconstruct the original input from the latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        """
        The forward pass of the model.
        Args:
            x (Tensor): The input tensor.
        Returns:
            Tensor: The reconstructed tensor.
        """
        # Pass input through the encoder, then the decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded