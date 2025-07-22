# /logadu/logic/autoencoder_lightning.py

import pytorch_lightning as pl
import torch.nn as nn
import torch  # ADD THIS IMPORT

from logadu.models.autoencoder import AutoEncoder

class AutoEncoderLightning(pl.LightningModule):
    """
    PyTorch Lightning module for the AutoEncoder model.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, _ = batch
        reconstructed = self(sequences)
        loss = self.criterion(reconstructed, sequences)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, _ = batch
        reconstructed = self(sequences)
        loss = self.criterion(reconstructed, sequences)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        sequences, labels = batch
        reconstructed = self(sequences)
        loss = self.criterion(reconstructed, sequences)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # This line will now work because 'torch' has been imported
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)