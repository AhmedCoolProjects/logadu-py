import torch.nn as nn
import torch

class ModelOutput:
    def __init__(self, logits, probabilities, loss=None, embeddings=None):
        self.logits = logits
        self.probabilities = probabilities
        self.loss = loss
        self.embeddings = embeddings

class DeepLog(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_layers=2, dropout=0.5):
        super(DeepLog, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, batch):
        # No 'device' argument needed here.
        # The tensors in 'batch' will already be on the correct device.
        x = batch['sequential']
        
        # The calculation of logits and probabilities remains the same
        x_embedded = self.embedding(x)
        out, _ = self.lstm(x_embedded)
        
        logits = self.fc(out[:, -1, :])
        probabilities = torch.softmax(logits, dim=-1)
        
        # We remove the loss calculation from the base model's forward pass.
        # This is the sole responsibility of the LightningModule.
        return ModelOutput(logits=logits, probabilities=probabilities, embeddings=out[:, -1, :])