import pytorch_lightning as pl
import torch
import torch.nn as nn
from logadu.models.autoencoder import AutoEncoder

class AutoEncoderLightning(pl.LightningModule):
    def __init__(self, input_dim, hidden_size, num_layers, latent_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoEncoder(
            input_dim=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, latent_dim=latent_dim
        )
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # Unsupervised: we only need the sequences, not the labels
        sequences, _ = batch
        
        # Get the original and reconstructed representations
        original_repr, reconstructed_repr = self.model(sequences)
        
        # Loss is the difference between the original and reconstructed summary vectors
        loss = self.criterion(reconstructed_repr, original_repr)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, _ = batch
        original_repr, reconstructed_repr = self.model(sequences)
        loss = self.criterion(reconstructed_repr, original_repr)
        self.log('val_loss', loss, prog_bar=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)