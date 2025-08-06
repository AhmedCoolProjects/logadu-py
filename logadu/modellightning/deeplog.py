import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from sklearn.metrics import classification_report
from logadu.models.deeplog import DeepLog

class DeepLogLightning(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size, num_layers, embedding_dim, learning_rate=0.001, top_k=9):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = DeepLog(
            vocab_size=vocab_size,
            # hyper parameters
            hidden_size=hidden_size,
            num_layers=num_layers,
            embedding_dim=embedding_dim
        )
        self.top_k = top_k
        
        self.criterion = nn.CrossEntropyLoss() # loss function for classification tasks
        
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.vocab_size, top_k=self.top_k)
        self.test_step_predictions = []
        self.test_step_labels = []
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def forward(self, sequences):
        # PyTorch Lightning automatically moves the input `sequences` tensor to the correct device.
        batch = {'sequential': sequences}
        return self.model(batch).probabilities
    
    def training_step(self, batch, batch_idx):
        sequences, next_events = batch
        batch_dict = {'sequential': sequences}
        
        logits = self.model(batch_dict).logits # logits are the raw scores before applying softmax
        loss = self.criterion(logits, next_events)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, next_events = batch
        batch_dict = {'sequential': sequences}
        
        output = self.model(batch_dict)
        
        loss = self.criterion(output.logits, next_events)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log(f'val_acc_top{self.top_k}', self.val_accuracy(output.logits, next_events))
    
    def test_step(self, batch, batch_idx):
        sequences, next_events, anomaly_labels = batch
        
        probabilities = self(sequences)
        
        # Get the top-k predicted next events
        topk_preds = torch.topk(probabilities, k=self.top_k).indices

        # Check if the true next event is within the top-k predictions
        is_in_topk = (next_events.unsqueeze(1) == topk_preds).any(dim=1)

        # --- Anomaly Logic ---
        # The predicted label is 1 (Anomaly) if the true next event is NOT in the top-k
        predicted_labels = (~is_in_topk).long()

        # Store the predictions and true labels for this batch
        self.test_step_predictions.append(predicted_labels)
        self.test_step_labels.append(anomaly_labels)
    
    def on_test_epoch_end(self):
        """
        This hook is called after the test loop finishes.
        Perfect place to calculate and print final metrics.
        """
        # Concatenate all predictions and labels from all test batches
        all_preds = torch.cat(self.test_step_predictions).cpu().numpy()
        all_labels = torch.cat(self.test_step_labels).cpu().numpy()

        # Generate and print the detailed classification report
        click.echo("\n" + "="*55)
        click.secho(f"  DeepLog Test Results (top_k={self.top_k})", fg="blue", bold=True)
        click.echo("="*55)
        
        report = classification_report(
            all_labels,
            all_preds,
            target_names=['Normal', 'Anomalous'],
            digits=4,
            zero_division=0
        )
        click.echo(report)
        click.echo("="*55)

        # Clear the lists for the next potential run
        self.test_step_predictions.clear()
        self.test_step_labels.clear()
