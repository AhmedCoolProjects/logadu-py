import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch
from pathlib import Path
# ---------------- DEEPLOG ----------------
from logadu.datamodules.deeplog import DeepLogDataModule
from logadu.modellightning.deeplog import DeepLogLightning

torch.set_float32_matmul_precision('high')  # Enable Tensor Cores for faster training

@click.command()
@click.argument("model", type=click.Choice(['deeplog']))
@click.argument("dataset_name", type=str)
@click.argument("window_size", type=int)
@click.option("--split-method", default=1, type=int, help="Which split type to use, 1: train/valid/test on squences, 2: train/valid/test log file, then sequencing with step size=1 for train, and step size=window size for valid and test.")
@click.option("--path", type=click.Path(exists=True), help="Path to the dataset file.")
@click.option("--epochs", default=50, help="Number of epochs for training.")
@click.option("--use-wandb", is_flag=True, help="Use Weights & Biases for logging.")
@click.option("--wandb-project", default="first_lad_in_apts", help="W&B project name to log runs to.")
def run(model, dataset_name, window_size, split_method, path, epochs, use_wandb, wandb_project):
    seq_type = None
    if model.lower() == "deeplog":
        seq_type = "index"
    else:
        raise ValueError(f"Unsupported model type: {model}")

    data_file = f"{path}/{dataset_name}/drain/{dataset_name}_{window_size}_1_seq_{seq_type}.csv"
    
    output_dir = f"{path}/{dataset_name}/models/{model.lower()}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if use_wandb:
        wandb_run_name = f"{model.lower()}-{dataset_name}-{window_size}-{seq_type}"
    
        wandb_logger = WandbLogger(project=wandb_project, name=wandb_run_name, log_model="all")
    
    try:
        if model.lower() == "deeplog":
            data_module = DeepLogDataModule(dataset_file=data_file, split_method=split_method)
            data_module.setup()
            
            lightning_model = DeepLogLightning(
                vocab_size=data_module.vocab_size,
                hidden_size=128,
                num_layers=2,   
                embedding_dim=128,
            )
        if data_module and lightning_model:
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                dirpath=output_dir,
                filename=f'{model}-{dataset_name}-{window_size}-{{epoch:02d}}-{{val_loss:.2f}}-best-checkpoint'
            )
            
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min'
            )
            
            if use_wandb:
                trainer = pl.Trainer(
                max_epochs=epochs,
                callbacks=[checkpoint_callback, early_stopping_callback],
                logger=wandb_logger,
                default_root_dir=output_dir,
                accelerator="auto"
            )
            else:
                trainer = pl.Trainer(
                    max_epochs=epochs,
                    callbacks=[checkpoint_callback, early_stopping_callback],
                    default_root_dir=output_dir,
                    accelerator="auto"
                )
            
            
            
            click.secho(f"Starting TRAIN and VALID for {model} on {dataset_name} with window size {window_size}...", fg="green")
            trainer.fit(lightning_model, data_module)
            
            click.secho(f"Starting TEST for {model} on {dataset_name} with window size {window_size}...", fg="green")
            trainer.test(datamodule=data_module, ckpt_path='best')
            click.secho(f"Model: {model}, Dataset: {dataset_name}, Window Size: {window_size}.", fg="blue")
    finally:
        if use_wandb:
            wandb.finish()
