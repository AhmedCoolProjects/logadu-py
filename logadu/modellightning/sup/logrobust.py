import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from torch.optim import SGD

# Helper module for the attention mechanism


class Attention(nn.Module):
    """
    Attention mechanism layer.

    Takes the output of a Bi-LSTM and computes a weighted sum (context vector)
    based on learned attention scores.
    """

    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        # The attention layer is a small feed-forward network
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        # lstm_output shape: (batch_size, sequence_length, hidden_size)

        # Pass each time step's hidden state through the attention network
        # to get an "energy" score.
        attn_energies = self.attn_net(lstm_output).squeeze(2)
        # attn_energies shape: (batch_size, sequence_length)

        # Apply softmax to get normalized attention weights
        alpha = F.softmax(attn_energies, dim=1)
        # alpha shape: (batch_size, sequence_length)

        # Unsqueeze alpha to allow for element-wise multiplication with LSTM outputs
        alpha = alpha.unsqueeze(2)
        # alpha shape: (batch_size, sequence_length, 1)

        # Compute the context vector as the weighted sum of LSTM outputs
        context_vector = torch.sum(lstm_output * alpha, dim=1)
        # context_vector shape: (batch_size, hidden_size)

        return context_vector


class LogRobustModel(pl.LightningModule):
    """
    PyTorch Lightning implementation of the LogRobust model.
    """

    def __init__(
        self,
        input_dim: int,  # Dimension of the semantic vectors
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_prob: float = 0.5,
        learning_rate: float = 1e-3
    ):
        super(LogRobustModel, self).__init__()
        self.save_hyperparameters()

        # 1. Bidirectional LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.hparams.input_dim,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            bidirectional=True,  # This is the key for Bi-LSTM
            batch_first=True,
            dropout=self.hparams.dropout_prob if self.hparams.num_layers > 1 else 0
        )

        # 2. Attention Layer
        # The input to attention is twice the hidden size because the LSTM is bidirectional
        self.attention = Attention(self.hparams.hidden_size * 2)

        # 3. Dropout and Final Classifier Layer
        self.dropout = nn.Dropout(self.hparams.dropout_prob)
        # Output size 2 (Normal/Anomalous)
        self.fc = nn.Linear(self.hparams.hidden_size * 2, 2)

        self.loss_fn = nn.CrossEntropyLoss()
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)

        # Pass through Bi-LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size * 2)

        # Pass through Attention layer
        context_vector = self.attention(lstm_out)
        # context_vector shape: (batch_size, hidden_size * 2)

        dropped_out = self.dropout(context_vector)

        # Final classification
        logits = self.fc(dropped_out)
        # logits shape: (batch_size, 2)

        return logits

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        sequences, anomaly_labels = batch
        logits = self(sequences)
        loss = self.loss_fn(logits, anomaly_labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        sequences, anomaly_labels = batch
        logits = self(sequences)
        loss = self.loss_fn(logits, anomaly_labels)
        self.log('val_loss', loss)

        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == anomaly_labels).item() / \
            (len(anomaly_labels) * 1.0)
        self.log('val_accuracy', acc)

    def test_step(self, batch: tuple, batch_idx: int):
        sequences, anomaly_labels = batch
        logits = self(sequences)
        predicted_labels = torch.argmax(logits, dim=1)
        self.test_step_outputs.append(
            {'preds': predicted_labels, 'targets': anomaly_labels})

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        all_preds = torch.cat([x['preds']
                              for x in self.test_step_outputs]).cpu().numpy()
        all_targets = torch.cat([x['targets']
                                for x in self.test_step_outputs]).cpu().numpy()
        self.test_step_outputs.clear()

        report_dict = classification_report(all_targets, all_preds, target_names=[
                                            'Normal (0)', 'Anomalous (1)'], zero_division=0, output_dict=True, digits=4)
        self.log_dict({
            'test_accuracy': report_dict['accuracy'],
            'test_macro_f1': report_dict['macro avg']['f1-score']
        })

        print("\n" + "="*60 + "\nTest Set Classification Report:\n" + "="*60)
        print(classification_report(all_targets, all_preds, digits=4,
              target_names=['Normal (0)', 'Anomalous (1)'], zero_division=0))
        print("="*60)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
