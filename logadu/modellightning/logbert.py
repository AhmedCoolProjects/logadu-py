import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from logadu.models.logbert import LogBERT_HF

class LogBERTLightning(pl.LightningModule):
    def __init__(self, vocab_size, window_size, hidden_size=256, num_layers=2, num_heads=4,
                 alpha=0.5, mask_ratio=0.15, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = LogBERT_HF(
            vocab_size=vocab_size,
            window_size=window_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )

    def mask_input(self, input_ids):
        """ Randomly mask tokens for MLKP """
        batch_size, seq_len = input_ids.size()
        mask = torch.rand((batch_size, seq_len), device=input_ids.device) < self.hparams.mask_ratio
        mask[:, 0] = False  # don't mask [CLS]
        
        target_tokens = input_ids[mask]
        masked_input = input_ids.clone()
        masked_input[mask] = 1  # [MASK] token id
        
        mask_positions = mask.nonzero(as_tuple=True)
        return masked_input, mask_positions, target_tokens

    def compute_loss(self, logits, targets, dist_embed):
        mlkp_loss = F.cross_entropy(logits, targets)

        # Initialize center if not set
        if self.model.center.sum() == 0:
            self.model.center = dist_embed.mean(dim=0).detach()
        vhm_loss = torch.mean(torch.sum((dist_embed - self.model.center) ** 2, dim=1))

        return mlkp_loss + self.hparams.alpha * vhm_loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        masked_input, mask_positions, target_tokens = self.mask_input(input_ids)
        logits, dist_embed = self.model(masked_input, mask_positions)
        loss = self.compute_loss(logits, target_tokens, dist_embed)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        masked_input, mask_positions, target_tokens = self.mask_input(input_ids)
        logits, dist_embed = self.model(masked_input, mask_positions)
        loss = self.compute_loss(logits, target_tokens, dist_embed)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
