import pytorch_lightning as pl
import torch
import torch.nn as nn
from logadu.models.logbert import LogBERT

class LogBERTLightning(pl.LightningModule):
    def __init__(self, vocab_size, alpha=1.0, learning_rate=1e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = LogBERT(vocab_size=vocab_size)
        self.mlkp_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.vhm_criterion = nn.MSELoss()
        self.register_buffer("center", torch.zeros(self.model.bert.config.hidden_size))
        self.nu = 1e-3

    def training_step(self, batch, batch_idx):
        input_ids, mlm_labels = batch
        
        model_output = self.model(input_ids)
        
        # Access attributes from the ModelOutput object
        mlkp_logits = model_output.logits
        dist_outputs = model_output.embeddings
        
        mlkp_loss = self.mlkp_criterion(mlkp_logits.view(-1, self.hparams.vocab_size), mlm_labels.view(-1))
        vhm_loss = self.vhm_criterion(dist_outputs, self.center.repeat(dist_outputs.size(0), 1))
        loss = mlkp_loss + self.hparams.alpha * vhm_loss
        
        self.log_dict({'train_loss': loss, 'train_mlkp_loss': mlkp_loss, 'train_vhm_loss': vhm_loss}, 
                      prog_bar=True, on_step=True, on_epoch=True)
        
        with torch.no_grad():
            batch_center = torch.mean(dist_outputs, dim=0)
            self.center = (1 - self.nu) * self.center + self.nu * batch_center

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, mlm_labels = batch
        
        model_output = self.model(input_ids)

        mlkp_logits = model_output.logits
        dist_outputs = model_output.embeddings
        
        mlkp_loss = self.mlkp_criterion(mlkp_logits.view(-1, self.hparams.vocab_size), mlm_labels.view(-1))
        vhm_loss = self.vhm_criterion(dist_outputs, self.center.repeat(dist_outputs.size(0), 1))
        loss = mlkp_loss + self.hparams.alpha * vhm_loss
        self.log_dict({'val_loss': loss, 'val_mlkp_loss': mlkp_loss, 'val_vhm_loss': vhm_loss})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)