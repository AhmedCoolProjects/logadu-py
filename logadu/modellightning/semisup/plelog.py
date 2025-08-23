import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from sklearn.metrics import classification_report

# The Attention module can be reused directly from the LogRobust implementation
class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )

    def forward(self, gru_output: torch.Tensor) -> torch.Tensor:
        attn_energies = self.attn_net(gru_output).squeeze(2)
        alpha = F.softmax(attn_energies, dim=1).unsqueeze(2)
        context_vector = torch.sum(gru_output * alpha, dim=1)
        return context_vector

class PLELogModel(pl.LightningModule):
    """
    PyTorch Lightning implementation of the PLELog model.
    
    This model uses an attention-based Bidirectional GRU for classification.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_prob: float = 0.5,
        learning_rate: float = 1e-3
    ):
        super(PLELogModel, self).__init__()
        self.save_hyperparameters()

        # --- KEY CHANGE: Use nn.GRU instead of nn.LSTM ---
        self.gru = nn.GRU(
            input_size=self.hparams.input_dim,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.hparams.dropout_prob if self.hparams.num_layers > 1 else 0
        )

        self.attention = Attention(self.hparams.hidden_size * 2)
        self.dropout = nn.Dropout(self.hparams.dropout_prob)
        self.fc = nn.Linear(self.hparams.hidden_size * 2, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        context_vector = self.attention(gru_out)
        dropped_out = self.dropout(context_vector)
        logits = self.fc(dropped_out)
        return logits

    # The training, validation, test, and optimizer methods are IDENTICAL to
    # LogRobust and LogCNN as it's a supervised classification task at its core.
    def training_step(self, batch: tuple, batch_idx: int):
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
        acc = torch.sum(preds == anomaly_labels).item() / len(anomaly_labels)
        self.log('val_accuracy', acc)

    def test_step(self, batch: tuple, batch_idx: int):
        sequences, anomaly_labels = batch
        logits = self(sequences)
        predicted_labels = torch.argmax(logits, dim=1)
        self.test_step_outputs.append({'preds': predicted_labels, 'targets': anomaly_labels})

    def on_test_epoch_end(self):
        # This logic can be copied from the previous answers
        if not self.test_step_outputs: return
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs]).cpu().numpy()
        self.test_step_outputs.clear()
        
        report_dict = classification_report(all_targets, all_preds, target_names=['Normal (0)', 'Anomalous (1)'], zero_division=0, output_dict=True)
        self.log_dict({
            'test_accuracy': report_dict['accuracy'],
            'test_f1_anomaly': report_dict['Anomalous (1)']['f1-score'],
            'test_macro_f1': report_dict['macro avg']['f1-score']
        })
        print("\n" + "="*60 + "\nTest Set Classification Report:\n" + "="*60)
        print(classification_report(all_targets, all_preds, target_names=['Normal (0)', 'Anomalous (1)'], zero_division=0))
        print("="*60)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)