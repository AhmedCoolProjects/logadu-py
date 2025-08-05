import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from logadu.models.deeplog import DeepLog 

class DeepLogLightning(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size, num_layers, embedding_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.model = DeepLog(
            vocab_size=vocab_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            embedding_dim=embedding_dim
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.vocab_size, top_k=9)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def forward(self, sequences):
        """ Defines the forward pass for inference. """
        # PyTorch Lightning automatically moves the input `sequences` tensor to the correct device.
        batch = {'sequential': sequences}
        # We no longer pass the device argument.
        return self.model(batch).probabilities

    def training_step(self, batch, batch_idx):
        sequences, next_events = batch
        batch_dict = {'sequential': sequences}
        
        # The model is already on the correct device, as are the input tensors.
        logits = self.model(batch_dict).logits
        
        # The target tensor `next_events` is also automatically moved to the device.
        loss = self.criterion(logits, next_events)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, next_events = batch
        batch_dict = {'sequential': sequences}
        
        output = self.model(batch_dict)
        
        loss = self.criterion(output.logits, next_events)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc_top9', self.val_accuracy(output.logits, next_events))
