# /logadu/logic/logbert_lightning.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
from logadu.models.logbert import LogBERT

class LogBERTLightning(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_attention_heads, 
                 alpha=1.0, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = LogBERT(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim, # Pass it to the model
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads
        )
        self.mlkp_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.vhm_criterion = nn.MSELoss()
        
        # --- THE FIX ---
        # The center must have the same dimension as the model's output embeddings.
        self.register_buffer("center", torch.zeros(self.hparams.embedding_dim))
        
        self.nu = 1e-3

    def training_step(self, batch, batch_idx):
        input_ids, mlm_labels = batch
        outputs = self.model(input_ids)
        
        # 1. Calculate Masked Log Key Prediction (MLKP) loss
        mlkp_logits = outputs["mlm_logits"]
        mlkp_loss = self.mlkp_criterion(mlkp_logits.view(-1, self.hparams.vocab_size), mlm_labels.view(-1))
        
        # 2. Calculate Volume of Hypersphere Minimization (VHM) loss
        dist_outputs = outputs["dist_output"]
        vhm_loss = self.vhm_criterion(dist_outputs, self.center.repeat(dist_outputs.size(0), 1))

        # 3. Combine the losses
        loss = mlkp_loss + self.hparams.alpha * vhm_loss
        
        self.log_dict({
            'train_loss': loss,
            'train_mlkp_loss': mlkp_loss,
            'train_vhm_loss': vhm_loss
        }, prog_bar=True)
        
        # 4. Update the center `c` using an exponential moving average
        with torch.no_grad():
            batch_center = torch.mean(dist_outputs, dim=0)
            self.center = (1 - self.nu) * self.center + self.nu * batch_center

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, mlm_labels = batch
        outputs = self.model(input_ids)
        mlkp_logits = outputs["mlm_logits"]
        dist_outputs = outputs["dist_output"]
        
        mlkp_loss = self.mlkp_criterion(mlkp_logits.view(-1, self.hparams.vocab_size), mlm_labels.view(-1))
        vhm_loss = self.vhm_criterion(dist_outputs, self.center.repeat(dist_outputs.size(0), 1))
        
        loss = mlkp_loss + self.hparams.alpha * vhm_loss
        self.log_dict({'val_loss': loss, 'val_mlkp_loss': mlkp_loss})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)