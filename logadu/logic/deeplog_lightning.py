# /logadu/logic/deeplog_lightning.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy

# Import your existing model definition
from logadu.models.deeplog import DeepLog

class DeepLogLightning(pl.LightningModule):
    """
    PyTorch Lightning module for the DeepLog model.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_keys, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.model = DeepLog(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_keys=num_keys
        )

        self.criterion = nn.CrossEntropyLoss()

        # --- FIX IS HERE ---
        # You must specify the task type and the number of classes for the metric.
        # self.hparams.num_keys is available because of self.save_hyperparameters()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_keys, top_k=9)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_keys, top_k=9)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_keys, top_k=9)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        sequences = sequences.view(sequences.size(0), -1, 1)
        
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_top9', self.train_accuracy(logits, labels), on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        sequences = sequences.view(sequences.size(0), -1, 1)

        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc_top9', self.val_accuracy(logits, labels))

    def test_step(self, batch, batch_idx):
        sequences, labels = batch
        sequences = sequences.view(sequences.size(0), -1, 1)
        
        logits = self(sequences)
        loss = self.criterion(logits, labels)
        
        self.log('test_loss', loss)
        self.log('test_acc_top9', self.test_accuracy(logits, labels))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer