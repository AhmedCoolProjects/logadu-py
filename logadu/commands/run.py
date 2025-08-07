import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.metrics import classification_report
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
@click.option("--n-splits", default=5, help="[Method 4] Number of folds for Time Series Cross-Validation.")
@click.option("--path", type=click.Path(exists=True), help="Path to the dataset file.")
@click.option("--epochs", default=50, help="Number of epochs for training.")
@click.option("--use-wandb", is_flag=True, help="Use Weights & Biases for logging.")
@click.option("--wandb-project", default="first_lad_in_apts", help="W&B project name to log runs to.")
def run(model, dataset_name, window_size, split_method, n_splits, path, epochs, use_wandb, wandb_project):
    seq_type = None
    if model.lower() == "deeplog":
        seq_type = "index"
    else:
        raise ValueError(f"Unsupported model type: {model}")

    if split_method == 1:
        data_file = f"{path}/{dataset_name}/drain/{dataset_name}_{window_size}_1_seq_{seq_type}.csv"
    elif split_method == 2 or split_method == 3 or split_method == 4:
        data_file = f"{path}/{dataset_name}/drain/{dataset_name}_merged.csv"
    
    output_dir = f"{path}/{dataset_name}/models/{split_method}/{model.lower()}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if use_wandb:
        wandb_run_name = f"{model.lower()}-{dataset_name}-{window_size}-{seq_type}"
    
        wandb_logger = WandbLogger(project=wandb_project, name=wandb_run_name, log_model="all")
    
    if split_method == 4:
        all_fold_preds, all_fold_labels = [], []
        for i in range(n_splits):
            click.secho(f"\n--- Starting Fold {i+1}/{n_splits} ---", fg="cyan", bold=True)
            data_module = DeepLogDataModule(
                dataset_file=data_file, split_method=4, window_size=window_size, n_splits=n_splits, fold_index=i)
            data_module.setup()
            
            # R-initializing the model for each fold to ensure no leakage
            lightning_model = DeepLogLightning(
                vocab_size=data_module.vocab_size,
                hidden_size=128,
                num_layers=2,
                embedding_dim=128,
            )
            
            trainer = pl.Trainer(
                max_epochs=epochs,
                accelerator="auto",
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')],
                enable_checkpointing=False
            )
            
            trainer.fit(lightning_model, datamodule=data_module, verbose=False)
            
            all_fold_preds.extend(lightning_model.test_step_predictions)
            all_fold_labels.extend(lightning_model.test_step_labels)
            
        # --- Aggregate results across folds ---
        click.secho("\n--- Aggregating Results Across Folds ---", fg="magenta", bold=True)
        final_preds = torch.cat(all_fold_preds).cpu().numpy()
        final_labels = torch.cat(all_fold_labels).cpu().numpy()
        
        click.echo("\n" + "="*60)
        click.secho(f"  Final Aggregated Time Series CV Report ({n_splits} Folds)", bold=True)
        click.echo("="*60)
        report = classification_report(final_labels, final_preds, target_names=['Normal', 'Anomalous'], digits=4)
        click.echo(report)
        click.echo("="*60)
    
    else:
    
        try:
            if model.lower() == "deeplog":
                data_module = DeepLogDataModule(dataset_file=data_file, split_method=split_method, window_size=window_size)
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
                    patience=15,
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
                click.secho(f"Model: {model}, Dataset: {dataset_name}, Window Size: {window_size}, Split Method: {split_method}", fg="blue")
        finally:
            if use_wandb:
                wandb.finish()
