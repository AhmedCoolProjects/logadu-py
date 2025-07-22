# /logadu/commands/train.py

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# ADD THIS IMPORT
from pytorch_lightning.loggers import WandbLogger
import wandb # ADD THIS IMPORT

# Import the new Lightning classes
from logadu.logic.deeplog_datamodule import DeepLogDataModule
from logadu.logic.deeplog_lightning import DeepLogLightning

@click.command()
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option("--model", default="deeplog", help="Model to train (default: deeplog)")
@click.option("--batch-size", default=128, help="Batch size for training (default: 128)")
@click.option("--epochs", default=50, help="Number of epochs for training (default: 50)")
@click.option("--learning-rate", default=0.001, help="Learning rate for optimizer (default: 0.001)")
@click.option("--hidden-size", default=128, help="Hidden size for LSTM (default: 128)")
@click.option("--num-layers", default=2, help="Number of LSTM layers (default: 2)")
@click.option("--output-dir", default="models", help="Directory to save the trained model (default: models)")
# --- ADD W&B OPTIONS ---
@click.option("--wandb-project", required=True, help="W&B project name to log runs to.")
@click.option("--wandb-run-name", default=None, help="W&B run name. If not provided, a random name is generated.")
def train(dataset_file, model, batch_size, epochs, learning_rate, hidden_size, num_layers, output_dir, wandb_project, wandb_run_name):
    """Train a log anomaly detection model using PyTorch Lightning and log to W&B."""
    if model.lower() == "deeplog":
        click.secho(f"Initializing DeepLog training with PyTorch Lightning...", fg="yellow")

        # 1. Initialize the W&B Logger
        # This will automatically log all hyperparameters, metrics, and save model checkpoints to W&B
        wandb_logger = WandbLogger(
            project=wandb_project, 
            name=wandb_run_name, 
            log_model="all" # "all" saves a checkpoint to W&B for every epoch that improves val_loss
        )

        try:
            # 2. Initialize the DataModule
            data_module = DeepLogDataModule(dataset_file=dataset_file, batch_size=batch_size)
            data_module.setup() 
            
            # 3. Initialize the LightningModule
            lightning_model = DeepLogLightning(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_keys=data_module.num_classes,
                learning_rate=learning_rate
            )

            # 4. Configure Callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath=output_dir,
                filename='deeplog-best-checkpoint',
                save_top_k=1,
                verbose=True,
                monitor='val_loss',
                mode='min'
            )
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')

            # 5. Initialize the Trainer, now with the logger
            trainer = pl.Trainer(
                max_epochs=epochs,
                callbacks=[checkpoint_callback, early_stopping_callback],
                logger=wandb_logger,  # PASS THE LOGGER HERE
                default_root_dir=output_dir,
                accelerator="auto"
            )

            # 6. Start Training
            click.echo("\n--- Starting Training & Validation ---")
            trainer.fit(lightning_model, datamodule=data_module)

            # 7. Start Testing
            click.echo("\n--- Starting Testing ---")
            trainer.test(datamodule=data_module, ckpt_path='best')

            click.secho("\nTraining, validation, and testing complete!", fg="green")

        finally:
            # Ensure that the W&B run is finished properly, even if an error occurs
            wandb.finish()

    else:
        raise click.UsageError(f"Model '{model}' is not supported. Currently only 'deeplog' is available.")