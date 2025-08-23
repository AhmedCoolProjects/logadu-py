import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import WandbLogger
import os
import psutil
# import gc
import time
# import wandb
# from sklearn.metrics import classification_report
import torch
from pathlib import Path
from logadu.datamodules.templates_dataloader import TemplatesDataLoader
from logadu.datamodules.raw_log_dataloader import RawLogDataModule


from logadu.modellightning.semisup.deeplog import DeepLogModel
from logadu.modellightning.sup.logcnn import LogCNNModel
from logadu.modellightning.sup.logrobust import LogRobustModel
from logadu.modellightning.sup.neurallog import NeuralLogModel
from logadu.modellightning.sup.ml import MLModels


# ---------------- DEEPLOG ----------------
# from logadu.datamodules.index import IndexDataModule
# from logadu.modellightning.deeplog import DeepLogLightning

# # ---------------- LOGBERT ----------------
# # from logadu.modellightning.logbert import LogBERTLightning

# # ---------------- ML ----------------
# from logadu.modellightning.agg_vector_template import MLLightningModule

# ---------------- LogRobust ----------------
# from logadu.modellightning.logrobust import LogRobustLightning
from logadu.datamodules.vector_template import NoAggDataModule

# ----------------- LogCNN ----------------
# from logadu.modellightning.logcnn import LogCNNLightning

# ----------------- PLELog ----------------
from logadu.modellightning.plelog import PLELogLightning

# ----------------- NeuralLog ---------------
# from logadu.modellightning.neurallog import NeuralLogLightning


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
    
    LOG_COLUMN_NAME = "Content"
    LABEL_COLUMN_NAME = "Label"
    TEMPLATE_COLUMN_NAME = "EventTemplate"
    EVENT_ID_COLUMN_NAME = "EventId"
    
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    click.secho(f"Available memory: {available_gb:.2f} GB", fg="yellow")
    
    gpath = Path(path)

    structured_file = gpath / "drain" / f"{dataset_name}_all_structured.csv"
    data_file = structured_file
    fasttext_vectors = gpath / "drain" / "fasttext" / f"{dataset_name}_vectors.pt"
    bert_vectors = gpath / "drain" / "bert" / f"{dataset_name}_vectors.pt"
    output_dir = gpath / "models" / dataset_name / f"{model.lower()}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # if use_wandb:
    #     wandb_run_name = f"{model.lower()}-{dataset_name}-{window_size}"
    
    #     wandb_logger = WandbLogger(project=wandb_project, name=wandb_run_name, log_model="all")
    
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
            # ==================================== Index Data Module ==========================
            if model.lower() == "deeplog":
                data_module = TemplatesDataLoader(csv_file_path=structured_file, window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                              col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME, 
                                              col_template_name=TEMPLATE_COLUMN_NAME,
                                              type='indexes')
                data_module.setup()
                
                lightning_model = DeepLogModel(
                    num_keys=data_module.vocab_size,
                    hidden_size=128,
                    num_layers=2,
                    g=9  # Use top_k templates
                )
            elif model.lower() == "logcnn":
                data_module = TemplatesDataLoader(csv_file_path=structured_file, window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                              col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME, 
                                              col_template_name=TEMPLATE_COLUMN_NAME,
                                              type='cnn_indexes')
                data_module.setup()
                lightning_model = LogCNNModel(
                    num_keys=data_module.vocab_size
                )
            # ==================================== FastText NOT Aggregated Vectors Data Module ==========================
            elif model.lower() == "logrobust":
                data_module = TemplatesDataLoader(csv_file_path=structured_file,
                                                  vector_map_path=fasttext_vectors,
                                                  window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                              col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME, 
                                              col_template_name=TEMPLATE_COLUMN_NAME,
                                              type='vectors')
                data_module.setup()
                
                lightning_model = LogRobustModel(
                    input_dim=data_module.input_dim
                )
            elif model.lower() == "neurallog":

                data_module = RawLogDataModule(csv_file_path=structured_file, window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                              col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME)
                data_module.setup()
                
                lightning_model = NeuralLogModel()
                
            elif model.lower() == "plelog":
                data_module = NoAggDataModule(
                    merged_file=structured_file,
                    vector_map_file=fasttext_vectors,
                    label_col=LABEL_COLUMN_NAME,
                    eventid_col=EVENT_ID_COLUMN_NAME,
                    content_col=LOG_COLUMN_NAME,
                    window_size=window_size,
                    num_workers=1,
                    aggregate=False,  # No aggregation for PLELog
                )
                data_module.setup()
                lightning_model = PLELogLightning(
                    input_dim=data_module.input_dim,
                )
                
            
            # elif model.lower() == "logbert":
            #     data_module = IndexDataModule(dataset_file=data_file, split_method=split_method, window_size=window_size)
            #     data_module.setup()
                
            #     lightning_model = LogBERTLightning(
            #         vocab_size=data_module.vocab_size,
            #     )
            # ==================================== FastText Aggregated Vectors Data Module ==========================
            elif model.lower() in ["pca", "knn", "rf"]:
                # cpu_count = os.cpu_count() or 1
                num_workers = 8
                click.secho(f"Using {num_workers} workers for data loading.", fg="yellow")
                
                data_module = TemplatesDataLoader(csv_file_path=structured_file,
                                                  vector_map_path=fasttext_vectors,
                                                  window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                              col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME, 
                                              col_template_name=TEMPLATE_COLUMN_NAME,
                                              num_workers=num_workers,
                                              type='agg_vector')
                
                model_params = {}
                if model.lower() == "pca":
                    model_params = {"n_components": n_components, 'random_state': 42}
                elif model.lower() == "knn":
                    model_params = {"n_neighbors": k_neighbors, "n_jobs": -1}
                elif model.lower() == "rf":
                    model_params = {"n_estimators": n_estimators, "random_state": 42, "n_jobs": -1}
                
                lightning_model = MLModels(
                    model_name=model.lower(),
                    **model_params
                )
                
                trainer = pl.Trainer(
                    max_epochs=1,
                    accelerator="cpu", # sklearn models run on CPU
                    enable_progress_bar=False,
                    logger=False # Disable default logging for simplicity
                )
                
                click.secho(f"Starting TRAIN and VALID for {model} on {dataset_name} with window size {window_size}...", fg="green")
                trainer.fit(lightning_model, data_module)
                
                click.secho(f"Starting TEST for {model} on {dataset_name} with window size {window_size}...", fg="green")
                trainer.test(datamodule=data_module, ckpt_path='last')
                click.secho(f"Model: {model}, Dataset: {dataset_name}, Window Size: {window_size}, Split Method: {split_method}", fg="blue")
                

              
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
                
                # if use_wandb:
                #     trainer = pl.Trainer(
                #     max_epochs=epochs,
                #     callbacks=[checkpoint_callback, early_stopping_callback],
                #     logger=wandb_logger,
                #     default_root_dir=output_dir,
                #     accelerator="auto"
                # )
                # else:
                trainer = pl.Trainer(
                    max_epochs=epochs,
                    callbacks=[checkpoint_callback, early_stopping_callback],
                    default_root_dir=output_dir,
                    accelerator="auto",
                    enable_progress_bar=False,
                )
                
                click.secho(f"Starting TRAIN and VALID for {model} on {dataset_name} with window size {window_size}...", fg="green")
                start_time_train = time.time()
                
                trainer.fit(lightning_model, data_module)
                end_time_train = time.time()
                train_duration = end_time_train - start_time_train
                train_duration_minutes = train_duration / 60
    
                # Print and log the training duration
                print(f"\nTotal Training Time: {train_duration:.2f} seconds, {train_duration_minutes:.2f} minutes")


                click.secho(f"Starting TEST for {model} on {dataset_name} with window size {window_size}...", fg="green")
                start_time_test = time.time()
                trainer.test(datamodule=data_module, ckpt_path='best')
                
                end_time_test = time.time()
                test_duration = end_time_test - start_time_test
                test_duration_minutes = test_duration / 60
                

                # Print and log the testing duration
                print(f"\nTotal Testing Time: {test_duration:.2f} seconds, {test_duration_minutes:.2f} minutes")

            # elif data_module and lightning_model and model.lower() in ['pca', 'knn', 'rf']:
            #     trainer = pl.Trainer(
            #         max_epochs=1,
            #         accelerator="cpu", # sklearn models run on CPU
            #         logger=False # Disable default logging for simplicity
            #     )

            #     click.secho(f"Starting TRAIN and VALID for {model} on {dataset_name} with window size {window_size}...", fg="green")
            #     trainer.fit(lightning_model, data_module)
                
            #     click.secho(f"Starting TEST for {model} on {dataset_name} with window size {window_size}...", fg="green")
            #     trainer.test(datamodule=data_module, ckpt_path='last')
            #     click.secho(f"Model: {model}, Dataset: {dataset_name}, Window Size: {window_size}, Split Method: {split_method}", fg="blue")
                
        finally:
            # if use_wandb:
            #     wandb.finish()
            click.secho("Cleaning up memory...", fg="yellow")
            
