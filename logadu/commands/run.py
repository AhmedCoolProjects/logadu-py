import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import os
import psutil
import gc
import wandb
# from sklearn.metrics import classification_report
import torch
from pathlib import Path
# ---------------- DEEPLOG ----------------
from logadu.datamodules.index import IndexDataModule
from logadu.modellightning.deeplog import DeepLogLightning

# ---------------- LOGBERT ----------------
# from logadu.modellightning.logbert import LogBERTLightning

# ---------------- ML ----------------
from logadu.modellightning.agg_vector_template import MLLightningModule
from logadu.datamodules.agg_vector_template import MLDataModuleFromMerged

# ---------------- LogRobust ----------------
from logadu.modellightning.logrobust import LogRobustLightning
from logadu.datamodules.vector_template import NoAggDataModule

# ----------------- LogCNN ----------------
from logadu.modellightning.logcnn import LogCNNLightning

# ----------------- PLELog ----------------
from logadu.modellightning.plelog import PLELogLightning

# ----------------- NeuralLog ---------------
from logadu.modellightning.neurallog import NeuralLogLightning


torch.set_float32_matmul_precision('high')  # Enable Tensor Cores for faster training

# logadu run pca Fox 5 --path .../implementation --vector-map-file .../vector.pt 
# logadu run knn Fox 5 --path .../implementation --vector-map-file .../vector.pt --k-neighbors 5

@click.command()
@click.argument("model", type=click.Choice(['deeplog', 'logbert', 'logrobust', 'logcnn', 'plelog', 'neurallog', 'pca', 'knn', 'rf']))
@click.argument("dataset_name", type=str)
@click.argument("window_size", type=int)
@click.option("--split-method", default=1, type=int, help="Which split type to use, 1: train/valid/test on squences, 2: train/valid/test log file, then sequencing with step size=1 for train, and step size=window size for valid and test.")
@click.option("--n-splits", default=5, help="[Method 4] Number of folds for Time Series Cross-Validation.")
@click.option("--path", type=click.Path(exists=True), help="Path to the dataset file.")
@click.option("--epochs", default=50, help="Number of epochs for training.")
@click.option("--k-neighbors", default=5, help="[KNN] Number of neighbors.")
@click.option("--n-estimators", default=100, help="[KNN] Number of estimators.")
@click.option("--n-components", default=0.95, help="[KNN] Number of components.")
@click.option("--topk", default=9, help="DeepLog and LogCNN: Top K most frequent templates to use for training.")
@click.option("--hidden-size", default=128, help="Hidden size for the LSTM layers in LogRobust and LogCNN.")
@click.option("--use-wandb", is_flag=True, help="Use Weights & Biases for logging.")
@click.option("--wandb-project", default="first_lad_in_apts", help="W&B project name to log runs to.")
def run(model, dataset_name, window_size, split_method, n_splits, path, epochs, k_neighbors, n_estimators, n_components, topk, hidden_size, use_wandb, wandb_project):
    
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    click.secho(f"Available memory: {available_gb:.2f} GB", fg="yellow")
    
    
    data_file = f"{path}/{dataset_name}/drain/{dataset_name}_merged.csv"
    
    output_dir = f"{path}/{dataset_name}/models/{split_method}/{model.lower()}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if use_wandb:
        wandb_run_name = f"{model.lower()}-{dataset_name}-{window_size}"
    
        wandb_logger = WandbLogger(project=wandb_project, name=wandb_run_name, log_model="all")
    
    # if split_method == 4:
    #     all_fold_preds, all_fold_labels = [], []
    #     for i in range(n_splits):
    #         click.secho(f"\n--- Starting Fold {i+1}/{n_splits} ---", fg="cyan", bold=True)
    #         data_module = DeepLogDataModule(
    #             dataset_file=data_file, split_method=4, window_size=window_size, n_splits=n_splits, fold_index=i)
    #         data_module.setup()
            
    #         # R-initializing the model for each fold to ensure no leakage
    #         lightning_model = DeepLogLightning(
    #             vocab_size=data_module.vocab_size,
    #             hidden_size=128,
    #             num_layers=2,
    #             embedding_dim=128,
    #         )
            
    #         trainer = pl.Trainer(
    #             max_epochs=epochs,
    #             accelerator="auto",
    #             callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')],
    #             enable_checkpointing=False
    #         )
            
    #         trainer.fit(lightning_model, datamodule=data_module, verbose=False)
            
    #         all_fold_preds.extend(lightning_model.test_step_predictions)
    #         all_fold_labels.extend(lightning_model.test_step_labels)
            
    #     # --- Aggregate results across folds ---
    #     click.secho("\n--- Aggregating Results Across Folds ---", fg="magenta", bold=True)
    #     final_preds = torch.cat(all_fold_preds).cpu().numpy()
    #     final_labels = torch.cat(all_fold_labels).cpu().numpy()
        
    #     click.echo("\n" + "="*60)
    #     click.secho(f"  Final Aggregated Time Series CV Report ({n_splits} Folds)", bold=True)
    #     click.echo("="*60)
    #     report = classification_report(final_labels, final_preds, target_names=['Normal', 'Anomalous'], digits=4)
    #     click.echo(report)
    #     click.echo("="*60)
    
    if True:
    
        try:
            if model.lower() == "deeplog":
                data_module = IndexDataModule(dataset_file=data_file, split_method=split_method, window_size=window_size)
                data_module.setup()
                
                lightning_model = DeepLogLightning(
                    vocab_size=data_module.vocab_size,
                    hidden_size=128,
                    num_layers=2,   
                    embedding_dim=128,
                    top_k=topk  # Use top_k templates
                )
            elif model.lower() == "logrobust":
                data_module = NoAggDataModule(
                    merged_file=data_file,
                    vector_map_file=f"{path}/{dataset_name}/drain/fasttext/{dataset_name}_templates_vectors.pt",
                    window_size=window_size,
                    batch_size=265,
                    num_workers=1,
                    aggregate=False  # No aggregation for LogRobust
                )
                data_module.setup()
                
                lightning_model = LogRobustLightning(
                    input_dim=data_module.input_dim,
                    hidden_size=hidden_size,
                    num_layers=2
                )
            elif model.lower() == "neurallog":
                N_HEADS = 8 # Number of attention heads. Must be a divisor of INPUT_DIMENSION.
                TRANSFORMER_HIDDEN_DIM = 2048 # Dimension of the feed-forward layer inside the transformer.
                N_LAYERS = 2 # Number of transformer encoder layers to stack.
                DROPOUT = 0.1
                LEARNING_RATE = 3e-5 # Transformers often benefit from smaller learning rates

                BATCH_SIZE = 128 * 4
            
                data_module = NoAggDataModule(
                    merged_file=data_file,
                    vector_map_file=f"{path}/{dataset_name}/drain/{dataset_name}_bert_vectors.pt",
                    window_size=window_size,
                    batch_size=BATCH_SIZE,
                    num_workers=1,
                    aggregate=False  # No aggregation for NeuralLog
                )
                data_module.setup()
                
                INPUT_DIM = data_module.input_dim
                if INPUT_DIM % N_HEADS != 0:
                    raise ValueError(f"INPUT_DIM ({INPUT_DIM}) must be divisible by N_HEADS ({N_HEADS}).")
                click.secho(f"Using NeuralLog with INPUT_DIM={INPUT_DIM}, N_HEADS={N_HEADS}", fg="green")
                
                lightning_model = NeuralLogLightning(
                    input_dim=INPUT_DIM,
                    n_head=N_HEADS,
                    hidden_dim=TRANSFORMER_HIDDEN_DIM,
                    n_layers=N_LAYERS,
                    learning_rate=LEARNING_RATE,
                    dropout=DROPOUT
                )
            elif model.lower() == "plelog":
                data_module = NoAggDataModule(
                    merged_file=data_file,
                    vector_map_file=f"{path}/{dataset_name}/drain/fasttext/{dataset_name}_templates_vectors.pt",
                    window_size=window_size,
                    num_workers=48,
                    aggregate=False,  # No aggregation for PLELog
                )
                data_module.setup()
                lightning_model = PLELogLightning(
                    input_dim=data_module.input_dim,
                )
                
            elif model.lower() == "logcnn":
                data_module = IndexDataModule(
                    dataset_file=data_file, 
                    split_method=split_method, 
                    window_size=window_size,
                    remove_duplicates=True,
                    label_type='next',
                    shuffle=False
                )
                data_module.setup()
                lightning_model = LogCNNLightning(
                    vocab_size=data_module.vocab_size,
                    embedding_dim=128,
                    hidden_size=128,
                    learning_rate=0.001,
                    top_k=topk
                )
            # elif model.lower() == "logbert":
            #     data_module = IndexDataModule(dataset_file=data_file, split_method=split_method, window_size=window_size)
            #     data_module.setup()
                
            #     lightning_model = LogBERTLightning(
            #         vocab_size=data_module.vocab_size,
            #     )
            elif model.lower() in ["pca", "knn", "rf"]:
                cpu_count = os.cpu_count() or 1
                # num_workers = max(1, (cpu_count * 2) // 3)
                num_workers = 8
                click.secho(f"Using {num_workers} workers for data loading.", fg="yellow")
                vector_map_file = f"{path}/{dataset_name}/drain/fasttext/{dataset_name}_templates_vectors.pt"
                data_module = MLDataModuleFromMerged(
                    merged_file=data_file,
                    vector_map_file=vector_map_file,
                    window_size=window_size,
                    num_workers=num_workers
                )
                model_params = {}
                if model.lower() == "pca":
                    model_params = {"n_components": n_components, 'random_state': 42}
                elif model.lower() == "knn":
                    model_params = {"n_neighbors": k_neighbors, "n_jobs": -1}
                elif model.lower() == "rf":
                    model_params = {"n_estimators": n_estimators, "random_state": 42, "n_jobs": -1}
                
                lightning_model = MLLightningModule(
                    model_name=model.lower(),
                    model_params=model_params,
                )

              
            if data_module and lightning_model and model.lower() in ['deeplog', 'logbert', 'logrobust', 'logcnn', 'plelog', 'neurallog']:
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
            
            elif data_module and lightning_model and model.lower() in ['pca', 'knn', 'rf']:
                trainer = pl.Trainer(
                    max_epochs=1,
                    accelerator="cpu", # sklearn models run on CPU
                    logger=False # Disable default logging for simplicity
                )

                click.secho(f"Starting TRAIN and VALID for {model} on {dataset_name} with window size {window_size}...", fg="green")
                trainer.fit(lightning_model, data_module)
                
                click.secho(f"Starting TEST for {model} on {dataset_name} with window size {window_size}...", fg="green")
                trainer.test(datamodule=data_module, ckpt_path='last')
                click.secho(f"Model: {model}, Dataset: {dataset_name}, Window Size: {window_size}, Split Method: {split_method}", fg="blue")
                
        finally:
            if use_wandb:
                wandb.finish()
