# /logadu/logic/neurallog_lightning.py

import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from logadu.models.neurallog import NeuralLog

class NeuralLogLightning(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, num_attention_heads, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = NeuralLog(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            num_attention_heads=num_attention_heads
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        
    # The training_step, validation_step, test_step, and configure_optimizers
    # methods will be identical to the ones in LogRobustLightning.
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
        self.log('val_f1', self.f1(logits, labels))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)