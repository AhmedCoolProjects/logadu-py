import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    """
    A Sequence-to-Sequence Autoencoder with an Attention mechanism.
    This model learns to reconstruct a summary representation of a log sequence.
    """
    def __init__(self, input_dim, hidden_size, num_layers, latent_dim, dropout=0.1):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # RNN layer to process the sequence
        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True, # Bi-LSTM
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism layers
        self.attention_w = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)
        self.attention_u = nn.Linear(self.hidden_size * 2, 1, bias=False)

        # Encoder and Decoder for the bottleneck
        self.encoder = nn.Linear(self.hidden_size * 2, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.hidden_size * 2)

    def attention(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size * 2)
        attn_tanh = torch.tanh(self.attention_w(lstm_output))
        attn_scores = self.attention_u(attn_tanh).squeeze(-1)
        
        # Softmax to get attention weights
        alphas = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(lstm_output * alphas, dim=1)
        return context_vector

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Pass through RNN
        lstm_out, _ = self.rnn(x)
        
        # Get the summary vector via attention
        representation = self.attention(lstm_out)
        
        # Pass through the encoder-decoder bottleneck
        encoded = self.encoder(representation)
        reconstructed = self.decoder(encoded)
        
        return representation, reconstructed