import pytorch_lightning as pl
import torch
import numpy as np
import click
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

class MLLightningModule(pl.LightningModule):
    """
    A generic PyTorch Lightning wrapper for traditional scikit-learn models.
    
    This module leverages the PL Trainer for structured execution (data loading, hooks)
    but uses the sklearn API for fitting and predicting.
    """
    def __init__(self, model_name: str, model_params: dict):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters() # Saves model_name and model_params
        self.model = self._initialize_model(model_name, model_params)
        
        # Lists to store outputs from each step for epoch-end calculation
        self.test_step_predictions = []
        self.test_step_labels = []

    def _initialize_model(self, model_name, params):
        """Factory function to create an sklearn model instance."""
        if model_name == 'knn':
            return KNeighborsClassifier(**params)
        elif model_name == 'rf':
            return RandomForestClassifier(**params)
        elif model_name == 'pca':
            return PCA(**params)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    def on_train_start(self):
        """
        The core training logic for sklearn models.
        This hook is called once by the Trainer before the training loop begins.
        """
        click.echo(f"\n--- Fitting sklearn model: {self.hparams.model_name.upper()} ---")
        
        # Access the dataloader from the trainer
        train_loader = self.trainer.train_dataloader
        
        # Concatenate all batches into a single dataset for sklearn
        X_train = torch.cat([batch[0] for batch in train_loader]).numpy()
        y_train = torch.cat([batch[1] for batch in train_loader]).numpy()
        
        # For PCA (unsupervised), we must train only on normal data
        if self.hparams.model_name == 'pca':
            click.echo("Filtering for normal data to fit PCA...")
            X_train = X_train[y_train == 0]
        
        self.model.fit(X_train, y_train)
        click.secho("Model fitting complete.", fg="green")

    def training_step(self, batch, batch_idx):
        # Dummy training step, can be empty
        pass
    def validation_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        return None

    def test_step(self, batch, batch_idx):
        """
        Perform prediction on a single batch of test data.
        """
        X, y = batch
        X = X.numpy()
        
        if self.hparams.model_name == 'pca':
            # For PCA, "prediction" is calculating reconstruction error
            X_transformed = self.model.transform(X)
            X_reconstructed = self.model.inverse_transform(X_transformed)
            reconstruction_errors = np.mean((X - X_reconstructed)**2, axis=1)
            
            # This threshold logic would need to be passed or calculated
            # For simplicity, we'll need a better way to handle thresholds,
            # but for now, let's assume a simple logic.
            # A more robust implementation would fit the threshold in on_train_start.
            threshold = 0.1 # Placeholder - this needs to be properly fitted
            preds = (reconstruction_errors > threshold).astype(int)
        else:
            # For RF and KNN, it's a direct prediction
            preds = self.model.predict(X)
        
        self.test_step_predictions.append(torch.tensor(preds))
        self.test_step_labels.append(y)

    def on_test_epoch_end(self):
        """
        Called after all test batches are processed. Calculates and prints final metrics.
        """
        all_preds = torch.cat(self.test_step_predictions).cpu().numpy()
        all_labels = torch.cat(self.test_step_labels).cpu().numpy()

        click.echo("\n" + "="*55)
        click.secho(f"  Final Test Report for {self.hparams.model_name.upper()}", bold=True)
        click.echo("="*55)
        
        report = classification_report(
            all_labels, all_preds, target_names=['Normal', 'Anomalous'],
            digits=4, zero_division=0
        )
        click.echo(report)
        click.echo("="*55)

        self.test_step_predictions.clear()
        self.test_step_labels.clear()
