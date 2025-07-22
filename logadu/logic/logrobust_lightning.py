# /logadu/logic/logrobust_lightning.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

from logadu.models.logrobust import LogRobust


class LogRobustLightning(pl.LightningModule):
    # --- MODIFIED SIGNATURE: No vocab_size, uses input_dim ---
    def __init__(self, input_dim, hidden_size, num_layers, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.model = LogRobust(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # This loss function is suitable for binary classification (logits output)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics for binary classification
        self.accuracy = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.precision_metric = BinaryPrecision()
        self.recall_metric = BinaryRecall()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences).squeeze(1) # Remove last dim to match label shape
        loss = self.criterion(logits, labels.float())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.accuracy(logits, labels), on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences).squeeze(1)
        loss = self.criterion(logits, labels.float())
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(logits, labels))
        self.log('val_f1', self.f1(logits, labels))

    def test_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences).squeeze(1)
        loss = self.criterion(logits, labels.float())
        
        self.log('test_loss', loss)
        self.log('test_acc', self.accuracy(logits, labels))
        self.log('test_f1', self.f1(logits, labels))
        self.log('test_precision', self.precision_metric(logits, labels))
        self.log('test_recall', self.recall_metric(logits, labels))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)