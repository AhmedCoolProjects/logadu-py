import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from sklearn.metrics import classification_report
import click
import numpy as np

from logadu.models.logrobust import LogRobust

class LogRobustLightning(pl.LightningModule):
    def __init__(self, input_dim, hidden_size, num_layers, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.model = LogRobust(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self.model(sequences).squeeze(1)
        loss = self.criterion(logits, labels.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self.model(sequences).squeeze(1)
        loss = self.criterion(logits, labels.float())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1', self.f1(torch.sigmoid(logits), labels))

    def test_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self.model(sequences).squeeze(1)
        preds = (torch.sigmoid(logits) > 0.5).long()
        self.test_step_outputs.append({'preds': preds, 'labels': labels})
    
    def on_test_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu().numpy()
        
        # Assertions to ensure label integrity
        assert all_labels.ndim == 1, f"Expected 1D labels, got shape {all_labels.shape}"
        unique_labels = np.unique(all_labels)
        assert set(unique_labels).issubset({0, 1}), f"Labels must be within {{0,1}}, found {unique_labels}"

        if unique_labels.size < 2:
            # Avoid sklearn ValueError by not calling classification_report with mismatched target_names
            counts = {int(l): int((all_labels == l).sum()) for l in unique_labels}
            click.secho("WARNING: Only one class present in test results.", fg="yellow")
            click.echo(f"Class distribution: {counts}")
            # Still show simple accuracy (all correct if only one class and model predicted same)
            simple_acc = (all_preds == all_labels).mean()
            click.echo(f"Simple accuracy: {simple_acc:.4f}")

            raise ValueError("Only one class present in test results.")
        else:
            # Safe call specifying labels to prevent mismatch errors
            report = classification_report(
                all_labels,
                all_preds,
                labels=[0, 1],
                target_names=['Normal', 'Anomalous'],
                digits=4,
                zero_division=0
            )
            click.echo(report)
        click.echo("="*55)
        self.test_step_outputs.clear()