# /logadu/commands/predict.py

import click
import torch
import pandas as pd
from logadu.logic.logbert_lightning import LogBERTLightning
from logadu.logic.logbert_datamodule import LogBERTDataModule
from sklearn.metrics import precision_recall_fscore_support

def predict_logbert(model, dataloader, g, r):
    """ Anomaly detection logic for LogBERT """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, _ = batch
            # Go through each sequence in the batch
            for i in range(input_ids.size(0)):
                seq = input_ids[i].tolist()
                
                # Find masked positions (from dataloader's masking)
                masked_indices = [idx for idx, token in enumerate(seq) if token == dataloader.vocab['[MASK]']]
                
                if not masked_indices:
                    predictions.append(0) # No masks, no anomaly detected
                    continue
                
                # Get model predictions
                logits = model(input_ids[i].unsqueeze(0).to(model.device))['mlm_logits']
                
                anomalous_keys = 0
                for masked_idx in masked_indices:
                    original_key = dataloader.test_sequences[i][masked_idx - 1] # -1 to account for [DIST]
                    top_g_preds = torch.topk(logits[0, masked_idx], g).indices
                    
                    if dataloader.vocab[str(original_key)] not in top_g_preds:
                        anomalous_keys += 1
                
                if anomalous_keys > r:
                    predictions.append(1) # Anomaly
                else:
                    predictions.append(0) # Normal

    return predictions

@click.command()
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option("--model-checkpoint", required=True, type=click.Path(exists=True), help="Path to the trained model checkpoint (.ckpt).")
@click.option("--g", default=10, help="Top-g candidates for normal prediction.")
@click.option("--r", default=1, help="Number of anomalous keys to flag a sequence as an anomaly.")
def predict(dataset_file, model_checkpoint, g, r):
    """Predict anomalies using a trained LogBERT model."""
    
    model = LogBERTLightning.load_from_checkpoint(model_checkpoint)
    
    # Setup dataloader on the full dataset
    data_module = LogBERTDataModule(dataset_file=dataset_file, mode='pretrain') # Use pretrain to mask test data
    data_module.setup()
    
    # We create a dataloader for the test set
    test_dataloader = DataLoader(data_module.test_dataset, batch_size=32, collate_fn=data_module._collate_fn)

    click.echo(f"Starting prediction with g={g} and r={r}...")
    predictions = predict_logbert(model, test_dataloader, g, r)
    
    true_labels = data_module.test_labels
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    click.secho(f"\n--- Prediction Results ---", fg="green")
    click.echo(f"Precision: {precision:.4f}")
    click.echo(f"Recall:    {recall:.4f}")
    click.echo(f"F1-Score:  {f1:.4f}")