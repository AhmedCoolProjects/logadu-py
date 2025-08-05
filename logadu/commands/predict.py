import click
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, classification_report
import ast
import random

# ----------------- DEEPLOG -----------------
from logadu.logic.deeplog_lightning import DeepLogLightning
# ----------------- LOGBERT -----------------
from logadu.modellightning.logbert import LogBERTLightning
from logadu.datamodules.logbert import LogBERTDataModule

def _predict_forecasting_model(model, dataloader, model_type, top_k, anomaly_threshold, vocab=None):
    """
    Generic prediction function for forecasting-based models like DeepLog and LogBERT.
    
    Args:
        model: The trained PyTorch Lightning model.
        dataloader: DataLoader for the test set.
        model_type (str): 'deeplog' or 'logbert'.
        top_k (int): The 'g' parameter from the papers.
        anomaly_threshold (int): The 'r' parameter from the papers.
        vocab (dict, optional): Vocabulary needed for LogBERT's MASK token.
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting with {model_type.upper()}"):
            sequences, true_next_events, true_anomaly_labels = batch
            sequences = sequences.to(device)
            
            predicted_labels = torch.zeros(sequences.size(0), dtype=torch.long)

            if model_type == 'deeplog':
                # DeepLog predicts the next event
                logits = model(sequences)
                topk_preds = torch.topk(logits, k=top_k).indices
                is_in_topk = (true_next_events.to(device).unsqueeze(1) == topk_preds).any(dim=1)
                predicted_labels = (~is_in_topk).long()

            elif model_type == 'logbert':
                # LogBERT predicts masked events
                batch_size, seq_len = sequences.shape
                
                # Create a mask for each sequence in the batch
                for i in range(batch_size):
                    # Clone to avoid modifying the original sequence
                    masked_seq = sequences[i].clone()
                    original_keys = {}
                    
                    # Randomly mask some tokens (excluding PAD and [DIST])
                    for j in range(1, seq_len):
                        if masked_seq[j] == vocab['<pad>']: break
                        if random.random() < 0.15: # 15% mask probability
                            original_keys[j] = masked_seq[j].item()
                            masked_seq[j] = vocab['[MASK]']
                    
                    if not original_keys:
                        predicted_labels[i] = 0 # No masks, normal by default
                        continue

                    # Get model predictions for the single masked sequence
                    outputs = model.model(masked_seq.unsqueeze(0))
                    mlm_logits = outputs["mlm_logits"]
                    
                    anomalous_key_count = 0
                    for masked_idx, original_key_id in original_keys.items():
                        topk_preds = torch.topk(mlm_logits[0, masked_idx], k=top_k).indices
                        if original_key_id not in topk_preds:
                            anomalous_key_count += 1
                    
                    if anomalous_key_count > anomaly_threshold:
                        predicted_labels[i] = 1 # Anomaly

            all_predictions.extend(predicted_labels.cpu().tolist())
            all_true_labels.extend(true_anomaly_labels.tolist())

    return all_true_labels, all_predictions


@click.command()
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option("--model-checkpoint", required=True, type=click.Path(exists=True), 
              help="Path to the trained PyTorch Lightning model checkpoint (.ckpt).")
@click.option("--model-type", required=True, type=click.Choice(['deeplog', 'logbert']),
              help="The type of model architecture being used for prediction.")
@click.option("--top-k", default=9, help="Top-k candidates for a normal prediction (the 'g' parameter).")
@click.option("--anomaly-threshold", default=0, 
              help="Number of anomalous keys required to flag a sequence as an anomaly (the 'r' parameter). Default is 0, meaning >0 anomalous keys.")
def predict(dataset_file, model_checkpoint, model_type, top_k, anomaly_threshold):
    """
    Predict anomalies using a trained semi-supervised model (DeepLog, LogBERT).
    """
    click.echo(f"Loading model '{model_type}' from checkpoint: {model_checkpoint}")
    
    # 1. Load the trained model
    if model_type == 'deeplog':
        model = DeepLogLightning.load_from_checkpoint(model_checkpoint)
    elif model_type == 'logbert':
        model = LogBERTLightning.load_from_checkpoint(model_checkpoint)
    else:
        raise click.UsageError("Invalid model type specified for prediction.")

    # 2. Load and prepare the full dataset for testing
    df = pd.read_csv(dataset_file)
    df['sequence'] = df['sequence'].apply(ast.literal_eval)
    
    X_test = torch.tensor(df['sequence'].tolist(), dtype=torch.long)
    y_test_next_event = torch.tensor(df['next'].tolist(), dtype=torch.long) # Needed for DeepLog
    y_test_anomaly_label = torch.tensor(df['label'].tolist(), dtype=torch.long) # The ground truth
    
    test_dataset = TensorDataset(X_test, y_test_next_event, y_test_anomaly_label)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. Run prediction
    if model_type == 'deeplog':
        true_labels, predictions = _predict_forecasting_model(model, test_loader, 'deeplog', top_k, anomaly_threshold)
    elif model_type == 'logbert':
        # LogBERT needs the vocabulary to know the [MASK] token ID
        data_module = LogBERTDataModule(dataset_file=dataset_file)
        data_module.setup()
        true_labels, predictions = _predict_forecasting_model(model, test_loader, 'logbert', top_k, anomaly_threshold, vocab=data_module.vocab)
    
    # 4. Calculate and display metrics
    click.secho("\n--- Anomaly Detection Results ---", fg="green")
    
    # Create a detailed report string
    report = classification_report(true_labels, predictions, target_names=['Normal', 'Anomalous'], zero_division=0)
    click.echo(report)