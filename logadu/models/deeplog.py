import torch
import torch.nn as nn
import click
from typing import Optional



class ModelOutput:
    # ... (same as before)
    def __init__(self, logits, probabilities, loss=None, embeddings=None):
        self.logits = logits
        self.probabilities = probabilities
        self.loss = loss
        self.embeddings = embeddings

class DeepLog(nn.Module):
    # ... (same as before, with one small correction in the embedding layer)
    def __init__(self,
                 hidden_size: int = 100,
                 num_layers: int = 2,
                 vocab_size: int = 100,
                 embedding_dim: int = 100,
                 dropout: float = 0.5,
                 criterion: Optional[nn.Module] = None):
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        # The vocab_size passed should be the number of unique keys.
        # Embedding layer needs to handle indexes from 0 to vocab_size-1.
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.criterion = criterion

    def forward(self, batch, device='cpu'):
        x = batch['sequential'].to(device)
        try:
            # The 'next' event is the label for the model's training loss
            y = batch['label'].to(device) 
        except KeyError:
            y = None
        
        x_embedded = self.embedding(x)
        out, _ = self.lstm(x_embedded)
        
        logits = self.fc(out[:, -1, :])
        probabilities = torch.softmax(logits, dim=-1)
        
        loss = None
        if y is not None and self.criterion is not None:
            loss = self.criterion(logits, y.view(-1))

        return ModelOutput(logits=logits, probabilities=probabilities, loss=loss, embeddings=out[:, -1, :])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



