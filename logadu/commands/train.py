# /logadu/commands/train.py

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import wandb
import torch # ADD THIS



# Import all modules
from logadu.logic.deeplog_datamodule import DeepLogDataModule
from logadu.logic.deeplog_lightning import DeepLogLightning
from logadu.logic.logrobust_datamodule import LogRobustDataModule
from logadu.logic.logrobust_lightning import LogRobustLightning
from logadu.logic.autoencoder_lightning import AutoEncoderLightning
from logadu.logic.logbert_datamodule import LogBERTDataModule
from logadu.logic.logbert_lightning import LogBERTLightning
from logadu.logic.neurallog_lightning import NeuralLogLightning


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
def train(dataset_file, model, batch_size, epochs, learning_rate, hidden_size, num_layers, 
          output_dir, wandb_project, wandb_run_name, embedding_dim, latent_dim, alpha, num_attention_heads):
    """Train a log anomaly detection model."""

    
    wandb_logger = WandbLogger(project=wandb_project, name=wandb_run_name, log_model="all")
    
    try:
        # --- MODEL DISPATCHER ---
        if model.lower() == "deeplog":
            click.secho(f"Initializing DeepLog training...", fg="yellow")
            data_module = DeepLogDataModule(dataset_file=dataset_file, batch_size=batch_size)
            data_module.setup() 
            
            lightning_model = DeepLogLightning(
                input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                num_keys=data_module.num_classes, learning_rate=learning_rate
            )

        # --- LOGROBUST TRAINING ---
        elif model.lower() == "logrobust":
            if not dataset_file.endswith('_vectors.pt'):
                raise click.UsageError("For LogRobust, the input file must be a pre-computed vector file ending in '_vectors.pt'. "
                                       "Please run the 'represent' command first.")
            
            click.secho(f"Initializing LogRobust training on pre-computed vectors...", fg="yellow")
            
            # The DataModule now takes the vectorized file path
            data_module = LogRobustDataModule(vectorized_file=dataset_file, batch_size=batch_size)
            data_module.setup()

            # The Lightning model now needs input_dim instead of vocab_size
            lightning_model = LogRobustLightning(
                input_dim=data_module.input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                learning_rate=learning_rate
            )
            
            # Note: For full LogRobust, you would load pre-trained embeddings here
            # and assign them to lightning_model.model.embedding.weight
        # --- ADD THE NEW AUTOENCODER BLOCK ---
        elif model.lower() == "autoencoder":
            if not dataset_file.endswith('_vectors.pt'):
                raise click.UsageError("For AutoEncoder, the input file must be a pre-computed vector file ending in '_vectors.pt'. "
                                       "Please run the 'represent' command first.")
            
            click.secho(f"Initializing AutoEncoder training on pre-computed vectors...", fg="yellow")
            
            # Reuse the LogRobustDataModule
            data_module = LogRobustDataModule(vectorized_file=dataset_file, batch_size=batch_size)
            data_module.setup()

            lightning_model = AutoEncoderLightning(
                input_dim=data_module.input_dim,
                hidden_dim=hidden_size, # Reusing the --hidden-size parameter
                latent_dim=latent_dim,
                learning_rate=learning_rate
            )
        if model.lower() == "logbert":
            # Input must be an index-based sequence file for LogBERT
            if not dataset_file.endswith('_seq.csv') and not dataset_file.endswith('_seq_index.csv'):
                 raise click.UsageError("For LogBERT, the input file must be an index-based sequence CSV.")

            click.secho(f"Initializing self-supervised LogBERT training...", fg="yellow")
            
            data_module = LogBERTDataModule(dataset_file=dataset_file, batch_size=batch_size)
            data_module.setup() 
            
            lightning_model = LogBERTLightning(
                vocab_size=len(data_module.vocab),
                embedding_dim=embedding_dim, # Pass the parameter here
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                alpha=alpha,
                learning_rate=learning_rate
            )
        elif model.lower() == "neurallog":
            if not dataset_file.endswith('_neurallog.pt'):
                raise click.UsageError("For NeuralLog, the input file must be a pre-computed vector file from the 'represent --model neurallog' command.")

            click.secho(f"Initializing NeuralLog training...", fg="yellow")
            
            # We can reuse the same DataModule as LogRobust
            data_module = LogRobustDataModule(vectorized_file=dataset_file, batch_size=batch_size)
            data_module.setup()

            lightning_model = NeuralLogLightning(
                input_dim=data_module.input_dim,
                hidden_dim=hidden_size, # Reusing --hidden-size
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                learning_rate=learning_rate
            )
            
        else:
            raise click.UsageError("Invalid model specified.")
            
        # --- COMMON TRAINER LOGIC ---
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=output_dir, filename=f'{model}-best-checkpoint')
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')

        trainer = pl.Trainer(
            max_epochs=epochs, callbacks=[checkpoint_callback, early_stopping_callback],
            logger=wandb_logger, default_root_dir=output_dir, accelerator="auto"
        )
        
        click.echo("\n--- Starting Training & Validation ---")
        trainer.fit(lightning_model, datamodule=data_module)

        # click.echo("\n--- Starting Testing ---")
        # trainer.test(datamodule=data_module, ckpt_path='best')

    finally:
        wandb.finish()