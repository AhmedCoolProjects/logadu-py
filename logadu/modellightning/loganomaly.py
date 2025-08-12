import pytorch_lightning as pl
import torch
from logadu.models.loganomaly import LogAnomalyModel
import torch.nn as nn
import click
import torch.nn.functional as F

class LogAnomalyLightning(pl.LightningModule):
    def __init__(self, semantic_input_dim, num_templates, hidden_size=128, num_layers=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = LogAnomalyModel(
            semantic_input_dim=self.hparams.semantic_input_dim,
            num_templates=self.hparams.num_templates,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers
        )

        # Loss functions for the two streams
        self.quantitative_loss = nn.CrossEntropyLoss()
        self.sequential_loss = nn.CosineEmbeddingLoss() # Good for comparing vectors

        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        # Your dataloader should provide the input sequences and the true "next" log
        seq_vectors, count_vectors, next_log_index, next_log_vector = batch
        
        # Get model predictions
        pred_logits, pred_vector = self.model(seq_vectors, count_vectors)
        
        # Calculate losses
        quant_loss = self.quantitative_loss(pred_logits, next_log_index)
        # For cosine loss, we need a target tensor of 1s (indicating they should be similar)
        target = torch.ones(seq_vectors.size(0), device=self.device)
        seq_loss = self.sequential_loss(pred_vector, next_log_vector, target)
        
        # Combine the losses (you can add a weight alpha)
        total_loss = quant_loss + seq_loss
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('quant_loss', quant_loss)
        self.log('seq_loss', seq_loss)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        # In testing, we calculate the "anomaly score"
        seq_vectors, count_vectors, next_log_index, next_log_vector = batch
        
        pred_logits, pred_vector = self.model(seq_vectors, count_vectors)
        
        # Anomaly score can be the prediction error
        quant_error = F.cross_entropy(pred_logits, next_log_index, reduction='none')
        seq_error = 1 - F.cosine_similarity(pred_vector, next_log_vector) # 1 - similarity = distance
        
        anomaly_score = quant_error + seq_error
        
        self.test_step_outputs.append({'scores': anomaly_score, 'labels': (your_true_anomaly_labels)})

    def on_test_epoch_end(self):
        # Here you would collect all scores and labels, then use a threshold
        # to calculate precision, recall, and F1-score.
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)