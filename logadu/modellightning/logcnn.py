import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from sklearn.metrics import classification_report
import click
from logadu.models.logcnn import LogCNN

class LogCNNLightning(pl.LightningModule):
    """ PyTorch Lightning module for the LogCNN model. """
    def __init__(self, vocab_size, embedding_dim, hidden_size, learning_rate=0.001, top_k=9):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = LogCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size
        )
        self.top_k = top_k
        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.vocab_size, top_k=self.top_k)
        
        # Lists to store outputs for the final test report
        self.test_step_predictions = []
        self.test_step_labels = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def training_step(self, batch, batch_idx):
        sequences, next_events = batch
        logits = self.model(sequences)
        loss = self.criterion(logits, next_events)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, next_events = batch
        logits = self.model(sequences)
        loss = self.criterion(logits, next_events)
        self.log('val_loss', loss, prog_bar=True)
        self.log(f'val_acc_top{self.top_k}', self.val_accuracy(logits, next_events))
    
    def test_step(self, batch, batch_idx):
        sequences, next_events, anomaly_labels = batch
        logits = self.model(sequences)
        probabilities = torch.softmax(logits, dim=-1)
        
        topk_preds = torch.topk(probabilities, k=self.top_k).indices
        is_in_topk = (next_events.unsqueeze(1) == topk_preds).any(dim=1)
        predicted_labels = (~is_in_topk).long()

        self.test_step_predictions.append(predicted_labels)
        self.test_step_labels.append(anomaly_labels)

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_step_predictions).cpu().numpy()
        all_labels = torch.cat(self.test_step_labels).cpu().numpy()

        click.echo("\n" + "="*55)
        click.secho(f"  LogCNN Final Test Report (top_k={self.top_k})", bold=True)
        click.echo("="*55)
        report = classification_report(
            all_labels, all_preds, target_names=['Normal', 'Anomalous'],
            digits=4, zero_division=0
        )
        click.echo(report)
        click.echo("="*55)

        self.test_step_predictions.clear()
        self.test_step_labels.clear()