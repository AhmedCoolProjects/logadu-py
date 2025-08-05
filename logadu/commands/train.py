import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch
from logadu.logic.deeplog_datamodule import DeepLogDataModule
from logadu.logic.deeplog_lightning import DeepLogLightning
# ---------------- LOGBERT ----------------
from logadu.datamodules.logbert import LogBERTDataModule
from logadu.modellightning.logbert import LogBERTLightning
from logadu.models.logbert import LogBERT
# from logadu.trainer.deeplog import train_deeplog
# from logadu.logic.logrobust_datamodule import LogRobustDataModule
# from logadu.logic.logrobust_lightning import LogRobustLightning
# from logadu.logic.autoencoder_lightning import AutoEncoderLightning
# from logadu.logic.logbert_datamodule import LogBERTDataModule
# from logadu.logic.logbert_lightning import LogBERTLightning
# from logadu.logic.neurallog_lightning import NeuralLogLightning


# ADD THIS LINE TO ENABLE TENSOR CORES FOR FASTER TRAINING
torch.set_float32_matmul_precision('high') 

@click.command()
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option("--batch-size", default=128, help="Batch size for training.")
@click.option("--epochs", default=50, help="Number of epochs for training.")
@click.option("--learning-rate", default=0.001, help="Learning rate for optimizer.")
@click.option("--hidden-size", default=128, help="Hidden size for LSTM.")
@click.option("--num-layers", default=2, help="Number of LSTM layers.")
@click.option("--output-dir", default="models", help="Directory to save the trained model.")
@click.option("--wandb-project", required=True, help="W&B project name to log runs to.")
@click.option("--wandb-run-name", default=None, help="W&B run name.")
# --- ADD LOGROBUST-SPECIFIC OPTIONS ---
@click.option("--embedding-dim", default=128, help="Dimension of embeddings for semantic models (LogRobust, LogBERT).")
# --- ADD 'autoencoder' TO THE CHOICE LIST ---
@click.option("--model", required=True, type=click.Choice(['deeplog', 'logrobust', 'logbert', 'autoencoder', 'neurallog']), help="Model to train.")
# --- ADD AUTOENCODER-SPECIFIC OPTIONS ---
@click.option("--latent-dim", default=32, help="[AutoEncoder] Dimension of the bottleneck layer.")
@click.option("--alpha", default=1.0, help="[LogBERT] Weight for the VHM loss.")
@click.option("--num-attention-heads", default=4, help="[LogBERT] Number of attention heads.")
@click.option("--dataset-name", default="default", help="Name of the dataset for logging purposes.")
def train(dataset_file, model, batch_size, epochs, learning_rate, hidden_size, num_layers, 
          output_dir, wandb_project, wandb_run_name, embedding_dim, latent_dim, alpha, num_attention_heads,
          dataset_name):
    """Train a log anomaly detection model."""

    
    wandb_logger = WandbLogger(project=wandb_project, name=wandb_run_name, log_model="all")
    
    try:
        # --- MODEL DISPATCHER ---
        if model.lower() == "deeplog":
            # 1. Initialize the DataModule
            data_module = DeepLogDataModule(dataset_file=dataset_file, batch_size=batch_size)
            data_module.setup()
            
            # 2. Initialize the LightningModule
            lightning_model = DeepLogLightning(
                vocab_size=data_module.vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                embedding_dim=embedding_dim,
                learning_rate=learning_rate
            )
  
        elif model.lower() == "logbert":
            click.secho(f"Initializing self-supervised LogBERT training...", fg="yellow")
            data_module = LogBERTDataModule(dataset_file=dataset_file, batch_size=batch_size)
            data_module.setup()
            lightning_model = LogBERTLightning(
                vocab_size=len(data_module.vocab), embedding_dim=embedding_dim, hidden_size=hidden_size,
                num_layers=num_layers, num_attention_heads=num_attention_heads,
                alpha=alpha, learning_rate=learning_rate
            )
        
        else:
            raise click.UsageError("Invalid model specified.")

    # --- COMMON TRAINER LOGIC FOR ALL MODELS ---
        if data_module and lightning_model:
            checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=output_dir, filename=f'{model}-{dataset_name}-best-checkpoint')
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')

            trainer = pl.Trainer(
                max_epochs=epochs,
                callbacks=[checkpoint_callback, early_stopping_callback],
                logger=wandb_logger,
                default_root_dir=output_dir,
                accelerator="auto"
            )

            click.echo("\n--- Starting Training & Validation ---")
            trainer.fit(lightning_model, datamodule=data_module)

    finally:
        wandb.finish()