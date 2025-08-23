import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from sklearn.metrics import classification_report
from transformers import AutoModel

# We need a PositionalEncoding module for the Transformer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, embedding_dim)
        # Pytorch Transformer expects (sequence_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2)


class NeuralLogModel(pl.LightningModule):
    """
    PyTorch Lightning implementation of the NeuralLog model.
    """
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        # Transformer Encoder parameters
        d_model: int = 768, # BERT's hidden size
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        learning_rate: float = 1e-5
    ):
        super(NeuralLogModel, self).__init__()
        self.save_hyperparameters()

        # 1. BERT Model for log message embedding
        self.bert = AutoModel.from_pretrained(bert_model_name)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Important!
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Final Classifier
        self.classifier = nn.Linear(d_model, 2) # 2 classes: Normal/Anomalous

        self.loss_fn = nn.CrossEntropyLoss()
        self.test_step_outputs = []

    # This is an alternative forward method for the NeuralLogModel class
    # that strictly follows the paper's description of averaging.

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids shape: (batch_size, window_size, max_log_len)
        # attention_mask shape: (batch_size, window_size, max_log_len)
        
        # --- Step 1: Get semantic vectors from BERT via AVERAGING ---
        batch_size, window_size, max_log_len = input_ids.shape
        
        input_ids_reshaped = input_ids.view(-1, max_log_len)
        attention_mask_reshaped = attention_mask.view(-1, max_log_len)
        
        bert_output = self.bert(input_ids=input_ids_reshaped, attention_mask=attention_mask_reshaped)
        
        # Paper Deviation 1: Average the token embeddings instead of using [CLS]
        # To do this correctly, we must ignore the padding tokens.
        last_hidden = bert_output.last_hidden_state
        input_mask_expanded = attention_mask_reshaped.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        log_vectors = sum_embeddings / sum_mask # This is the mean pooling

        log_sequence_vectors = log_vectors.view(batch_size, window_size, self.hparams.d_model)
        # shape: (batch_size, window_size, d_model)
        
        # --- Step 2: Pass through Transformer Encoder (same as before) ---
        seq_with_pos = self.pos_encoder(log_sequence_vectors)
        transformer_output = self.transformer_encoder(seq_with_pos)
        # shape: (batch_size, window_size, d_model)
        
        # --- Step 3: Classify with POOLING ---
        # Paper Deviation 2: Average the sequence outputs instead of taking the first one
        # This is a simple implementation of the "pooling" step mentioned in the paper.
        sequence_embedding = torch.mean(transformer_output, dim=1)
        
        logits = self.classifier(sequence_embedding)
        # shape: (batch_size, 2)
        
        return logits

    # The training, validation, testing, and optimizer logic is identical
    # to LogCNN as it's a standard supervised classification task.
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        input_ids, attention_mask, anomaly_labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, anomaly_labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        input_ids, attention_mask, anomaly_labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, anomaly_labels)
        self.log('val_loss', loss)
        
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == anomaly_labels).item() / (len(anomaly_labels) * 1.0)
        self.log('val_accuracy', acc)

    def test_step(self, batch: tuple, batch_idx: int):
        input_ids, attention_mask, anomaly_labels = batch
        logits = self(input_ids, attention_mask)
        predicted_labels = torch.argmax(logits, dim=1)
        self.test_step_outputs.append({'preds': predicted_labels, 'targets': anomaly_labels})

    def on_test_epoch_end(self):
        # This is identical to the LogCNN implementation
        if not self.test_step_outputs: return
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs]).cpu().numpy()
        self.test_step_outputs.clear()
        
        report_dict = classification_report(all_targets, all_preds, target_names=['Normal (0)', 'Anomalous (1)'], zero_division=0, output_dict=True, digits=4)
        self.log_dict({
            'test_accuracy': report_dict['accuracy'],
            'test_precision_anomaly': report_dict['Anomalous (1)']['precision'],
            'test_recall_anomaly': report_dict['Anomalous (1)']['recall'],
            'test_f1_anomaly': report_dict['Anomalous (1)']['f1-score'],
            'test_macro_f1': report_dict['macro avg']['f1-score']
        })
        
        print("\n" + "="*60 + "\nTest Set Classification Report:\n" + "="*60)
        print(classification_report(all_targets, all_preds, target_names=['Normal (0)', 'Anomalous (1)'], zero_division=0, digits=4))
        print("="*60)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return Adam(self.parameters(), lr=self.hparams.learning_rate)