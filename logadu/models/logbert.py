import torch.nn as nn
import torch
from transformers import BertModel, BertConfig
from logadu.utils.output import ModelOutput


class LogBERT(nn.Module):
    """
    LogBERT model implemented using the standard Hugging Face Transformers library.
    This approach is more robust, maintainable, and allows leveraging pre-trained models.
    """
    def __init__(self, model_name='bert-base-uncased', vocab_size=None):
        super(LogBERT, self).__init__()
        
        # If a vocab_size is provided, we initialize a BERT model from scratch.
        # This is useful for a pure index-based approach without pre-training.
        if vocab_size:
            self.bert_config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=128,          # Smaller embedding size for efficiency
                num_hidden_layers=2,      # A shallow model is often enough
                num_attention_heads=2,    # Fewer heads
                is_decoder=False
            )
            self.bert = BertModel(config=self.bert_config)
        else:
            # If no vocab_size, load a powerful pre-trained BERT model.
            # This is the standard approach for semantic tasks.
            self.bert = BertModel.from_pretrained(model_name)
        
        # Get the hidden dimension from the BERT model's config
        hidden_dim = self.bert.config.hidden_size
        
        # Head for the Masked Log Key Prediction (MLM) task
        # It must predict a score for each word in the vocabulary.
        self.mlm_head = nn.Linear(hidden_dim, vocab_size if vocab_size else self.bert.config.vocab_size)

    def forward(self, x):
        """
        The forward pass of the model, now returning a standardized ModelOutput object.
        """
        attention_mask = (x > 0).long()
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # --- Prepare outputs for the ModelOutput class ---
        
        # 1. Logits for Masked Log Key Prediction (MLKP)
        mlm_logits = self.mlm_head(sequence_output)
        
        # 2. Probabilities for MLKP (useful for prediction/inference)
        mlm_probabilities = torch.softmax(mlm_logits, dim=-1)
        
        # 3. Embeddings for Volume of Hypersphere Minimization (VHM)
        # This is the representation of the entire sequence from the [DIST] token
        dist_output = sequence_output[:, 0, :]

        # The loss is calculated in the LightningModule, so we return None here.
        return ModelOutput(
            logits=mlm_logits,
            probabilities=mlm_probabilities,
            loss=None,
            embeddings=dist_output
        )