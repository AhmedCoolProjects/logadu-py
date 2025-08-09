import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class LogBERT_HF(nn.Module):
    def __init__(self, vocab_size, window_size, hidden_size=256, num_layers=2, num_heads=4):
        super().__init__()

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 2,
            max_position_embeddings=window_size,
        )

        self.bert = BertModel(config)
        self.mlkp_head = nn.Linear(hidden_size, vocab_size)

        # Center for VHM
        self.register_buffer("center", torch.zeros(hidden_size))

    def forward(self, input_ids, mask_positions=None):
        outputs = self.bert(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state

        logits = None
        if mask_positions is not None:
            masked_hidden = hidden_states[mask_positions]
            logits = self.mlkp_head(masked_hidden)

        dist_embed = hidden_states[:, 0, :]  # [CLS] token
        return logits, dist_embed
