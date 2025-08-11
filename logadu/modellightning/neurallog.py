import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from sklearn.metrics import classification_report
from logadu.models.neurallog import NeuralLogModel
import click

class NeuralLogLightning(pl.LightningModule):
    def __init__(self, input_dim, n_head, hidden_dim, n_layers, learning_rate=1e-4, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.model = NeuralLogModel(
            input_dim=self.hparams.input_dim,
            n_head=self.hparams.n_head,
            hidden_dim=self.hparams.hidden_dim,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout
        )
        
        # Metrics and test output storage
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        preds = torch.sigmoid(logits)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1', self.val_f1(preds, y), prog_bar=True)
        self.log('val_acc', self.val_acc(preds, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        preds = (torch.sigmoid(logits) > 0.5).int()
        self.test_step_outputs.append({'preds': preds, 'labels': y})
        self.log('test_loss', loss)

    def on_test_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu().numpy()

        click.echo("\n" + "="*55)
        click.secho(f"  NeauralLog Final Test Report", bold=True)
        click.echo("="*55)
        report = classification_report(
            all_labels, all_preds, target_names=['Normal', 'Anomalous'],
            digits=4, zero_division=0
        )
        click.echo(report)
        click.echo("="*55)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)