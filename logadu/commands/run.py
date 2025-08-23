import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer
from pytorch_lightning.loggers import CSVLogger
import psutil
import torch
from pathlib import Path

# ----- Models -------
from logadu.dataloader.single_dataset_validation import SingleDatasetValidationDataLoader

from logadu.modellightning.sup.logrobust import LogRobustModel
from logadu.modellightning.semisup.logbert import LogBERTModel
from logadu.modellightning.semisup.deeplog import DeepLogModel
from logadu.modellightning.sup.logcnn import LogCNNModel
from logadu.modellightning.sup.ml import KNNModel, PCAModel, RFModel, OCSVMModel

torch.set_float32_matmul_precision('high')  # or 'medium'


@click.command()
@click.argument("model", type=click.Choice(['deeplog', 'logbert', 'logrobust', 'logcnn', 'plelog', 'neurallog', 'pca', 'knn', 'rf', 'ocsvm']))
@click.argument("paradigm", type=click.Choice(['supervised', 'semi', 'unsupervised']))
@click.argument("dataset_name", type=str)
@click.argument("window_size", type=int)
@click.option("--gpath", type=click.Path(exists=True), help="Path to the dataset file.")
@click.option("--epochs", default=25, help="Number of epochs for training.")
def run(model, paradigm, dataset_name, window_size, gpath, epochs):

    LOG_COLUMN_NAME = "Content"
    LABEL_COLUMN_NAME = "Label"
    TEMPLATE_COLUMN_NAME = "EventTemplate"
    EVENT_ID_COLUMN_NAME = "EventId"

    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    click.secho(f"Available memory: {available_gb:.2f} GB", fg="yellow")

    gpath = Path(gpath)

    structured_file = gpath / "drain" / f"{dataset_name}_all_structured.csv"
    fasttext_vectors = gpath / "drain" / \
        "fasttext" / f"{dataset_name}_vectors.pt"
    output_dir = gpath / "trained_models" / dataset_name / f"{model.lower()}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(save_dir=output_dir,
                       name=f"{model}-{dataset_name}-{window_size}")

    timer = Timer()

    logs_dir = Path(logger.log_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log_file = logs_dir / "run.log"

    is_DL = model.lower() in ["deeplog", "logbert",
                              "logrobust", "logcnn", "plelog", "neurallog"]

    if model.lower() == "logrobust":
        data_module = SingleDatasetValidationDataLoader(csv_file_path=structured_file,
                                                        vector_map_path=fasttext_vectors,
                                                        window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                                        col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME,
                                                        col_template_name=TEMPLATE_COLUMN_NAME,
                                                        paradigm="supervised",
                                                        type='seq_vectors_and_label')
        data_module.setup()

        click.secho(
            f"Input dimension for LogRobust: {data_module.input_dim}", fg="green")

        lightning_model = LogRobustModel(
            input_dim=data_module.input_dim,
        )

    if model.lower() == "logbert":
        data_module = SingleDatasetValidationDataLoader(csv_file_path=structured_file,
                                                        vector_map_path=fasttext_vectors,
                                                        window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                                        col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME,
                                                        col_template_name=TEMPLATE_COLUMN_NAME,
                                                        paradigm="semi",
                                                        type='indices')
        data_module.setup()

        click.secho(
            f"Input dimension for LogBERT: {data_module.vocab_size}", fg="green")

        lightning_model = LogBERTModel(
            vocab_size=data_module.vocab_size,
            max_len=window_size,
            r_threshold=3,
            top_g=10,
            log_file_path=str(run_log_file)


        )

    if model.lower() == "deeplog":
        data_module = SingleDatasetValidationDataLoader(csv_file_path=structured_file,
                                                        vector_map_path=fasttext_vectors,
                                                        window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                                        col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME,
                                                        col_template_name=TEMPLATE_COLUMN_NAME,
                                                        paradigm="semi",
                                                        type='indices')
        data_module.setup()

        click.secho(
            f"Input dimension for DeepLog: {data_module.vocab_size}", fg="green")

        lightning_model = DeepLogModel(
            vocab_size=data_module.vocab_size,
            h=window_size,
            log_file_path=str(run_log_file)

        )

    if model.lower() == "logcnn":
        data_module = SingleDatasetValidationDataLoader(csv_file_path=structured_file,
                                                        vector_map_path=fasttext_vectors,
                                                        window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                                        col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME,
                                                        col_template_name=TEMPLATE_COLUMN_NAME,
                                                        paradigm="supervised",
                                                        type='indices')
        data_module.setup()

        click.secho(
            f"Input dimension for LogCNN: {data_module.vocab_size}", fg="green")

        lightning_model = LogCNNModel(
            vocab_size=data_module.vocab_size,
            seq_len=window_size,
            log_file_path=str(run_log_file)

        )

    if model.lower() == "knn":
        data_module = SingleDatasetValidationDataLoader(csv_file_path=structured_file,
                                                        vector_map_path=fasttext_vectors,
                                                        window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                                        col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME,
                                                        col_template_name=TEMPLATE_COLUMN_NAME,
                                                        paradigm="supervised",
                                                        type='seq_vector_and_label')
        data_module.setup()

        knn_model = KNNModel(
            X_train_tensor=torch.cat(
                [data_module.train_dataset.tensors[0], data_module.val_dataset.tensors[0]]),
            y_train_tensor=torch.cat(
                [data_module.train_dataset.tensors[1], data_module.val_dataset.tensors[1]]),
            X_test_tensor=data_module.test_dataset.tensors[0],
            y_test_tensor=data_module.test_dataset.tensors[1]
        )

        knn_model.train()

    if model.lower() == "rf":
        data_module = SingleDatasetValidationDataLoader(csv_file_path=structured_file,
                                                        vector_map_path=fasttext_vectors,
                                                        window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                                        col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME,
                                                        col_template_name=TEMPLATE_COLUMN_NAME,
                                                        paradigm="supervised",
                                                        type='seq_vector_and_label')
        data_module.setup()

        rf_model = RFModel(
            X_train_tensor=torch.cat(
                [data_module.train_dataset.tensors[0], data_module.val_dataset.tensors[0]]),
            y_train_tensor=torch.cat(
                [data_module.train_dataset.tensors[1], data_module.val_dataset.tensors[1]]),
            X_test_tensor=data_module.test_dataset.tensors[0],
            y_test_tensor=data_module.test_dataset.tensors[1]
        )

        rf_model.train()

    if model.lower() == "pca":
        data_module = SingleDatasetValidationDataLoader(csv_file_path=structured_file,
                                                        vector_map_path=fasttext_vectors,
                                                        window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                                        col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME,
                                                        col_template_name=TEMPLATE_COLUMN_NAME,
                                                        paradigm=paradigm,
                                                        type='seq_vector_and_label')
        data_module.setup()

        pca_model = PCAModel(
            X_train_tensor=torch.cat(
                [data_module.train_dataset.tensors[0], data_module.val_dataset.tensors[0]]),
            X_test_tensor=data_module.test_dataset.tensors[0],
            y_test_tensor=data_module.test_dataset.tensors[1]
        )
        pca_model.train()

    if model.lower() == "ocsvm":
        dm = SingleDatasetValidationDataLoader(csv_file_path=structured_file,
                                               vector_map_path=fasttext_vectors,
                                               window_size=window_size, col_label_name=LABEL_COLUMN_NAME,
                                               col_eventid_name=EVENT_ID_COLUMN_NAME, col_content_name=LOG_COLUMN_NAME,
                                               col_template_name=TEMPLATE_COLUMN_NAME,
                                               paradigm=paradigm,
                                               type='seq_vector_and_label')
        dm.setup()

        _model = OCSVMModel(
            X_train_tensor=torch.cat(
                [dm.train_dataset.tensors[0], dm.val_dataset.tensors[0]]),
            X_test_tensor=dm.test_dataset.tensors[0],
            y_test_tensor=dm.test_dataset.tensors[1]
        )
        _model.train()

    if is_DL:
        # ===========================================
        # Training and Testing Callbacks
        # ===========================================

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=logger.log_dir,
            filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}'
        )

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min',
            verbose=True
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early_stopping_callback, timer],
            logger=logger,
            default_root_dir=output_dir,
            accelerator="auto",
            enable_progress_bar=True,
        )

        click.secho(
            f"Starting TRAIN and VALID for >>>>{model}<<<< on >>>>{dataset_name}<<<< with window size {window_size}...", fg="green")
        trainer.fit(lightning_model, data_module)

        # Call calibration method if available (supports different model names)
        if hasattr(lightning_model, "tune_r_g_on_val"):
            lightning_model.tune_r_g_on_val(
                data_module.val_dataloader(), target_fpr=0.01)
        elif hasattr(lightning_model, "tune_g_on_val"):
            lightning_model.tune_g_on_val(
                data_module.val_dataloader(), target_fpr=0.01)

        click.secho(
            f"Starting TEST for {model} on {dataset_name} with window size {window_size}...", fg="green")
        trainer.test(datamodule=data_module, ckpt_path='best')

        # ===========================================
        # Print Summary from the Callbacks
        # ===========================================

        # === Timings (append to the same run.log) ===
        train_duration_seconds = timer.time_elapsed("train")
        test_duration_seconds = timer.time_elapsed("test")
        train_duration_minutes = train_duration_seconds / 60
        test_duration_minutes = test_duration_seconds / 60

        print(
            f"\nTotal Training Time: {train_duration_seconds:.2f} seconds ({train_duration_minutes:.2f} minutes)")
        print(
            f"Total Testing Time: {test_duration_seconds:.2f} seconds ({test_duration_minutes:.2f} minutes)")

        try:
            with open(run_log_file, "a", encoding="utf-8") as f:
                f.write("\n=== Runtime Summary ===\n")
                f.write(
                    f"Training Time: {train_duration_seconds:.2f}s ({train_duration_minutes:.2f}m)\n")
                f.write(
                    f"Testing Time:  {test_duration_seconds:.2f}s ({test_duration_minutes:.2f}m)\n")
                f.write(
                    f"Best Checkpoint: {checkpoint_callback.best_model_path}\n")
                f.write(f"Metrics CSV:    {logger.log_dir}/metrics.csv\n")
        except Exception:
            pass

        print(f"Best model saved at: {checkpoint_callback.best_model_path}")
        print(f"Metrics saved in: {logger.log_dir}/metrics.csv")
