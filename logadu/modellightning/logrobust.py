import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from sklearn.metrics import classification_report
import click

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

        click.echo("\n" + "="*55)
        click.secho(f"  LogRobust Final Test Report", bold=True)
        click.echo("="*55)
        report = classification_report(
            all_labels, all_preds, target_names=['Normal', 'Anomalous'],
            digits=4, zero_division=0
        )
        click.echo(report)
        click.echo("="*55)
        self.test_step_outputs.clear()