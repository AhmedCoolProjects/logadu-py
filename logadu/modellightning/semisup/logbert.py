import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple
from sklearn.metrics import classification_report  # pip install scikit-learn
from pathlib import Path
from datetime import datetime
from logadu.modellightning.positional_encoding import SinusoidalPositionalEncoding


class LogBERTModel(pl.LightningModule):
    """
    LogBERT:
      - Input: integer log-key sequences of length L (window size)
      - Prepend a special DIST token
      - Self-supervision: masked log-key prediction (MLKP) + distance-to-center (VHM)
    """

    def __init__(
        self,
        # number of unique log keys (no specials)
        vocab_size: int,
        max_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        # fraction of tokens to mask (not counting DIST)
        mask_ratio: float = 0.15,
        alpha_vhm: float = 1.0,          # weight for VHM loss
        center_momentum: float = 0.9,    # EMA for the VHM center
        top_g: int = 9,                  # test-time: candidate set size
        r_threshold: int = 1,            # test-time: misses >= r => anomalous
        lr: float = 1e-3,
        log_file_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Special tokens added after the base vocabulary
        self.base_vocab = vocab_size
        self.mask_id = vocab_size        # MASK
        self.dist_id = vocab_size + 1    # DIST (sequence-level token)
        self.total_vocab = vocab_size + 2

        # Embedding + positional encoding
        self.embed = nn.Embedding(self.total_vocab, d_model)
        self.posenc = SinusoidalPositionalEncoding(
            d_model, max_len=max_len + 1)  # +1 for DIST

        # Transformer encoder (BERT-like encoder stack)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Prediction head over *base* vocab (we never predict specials)
        self.mlm_head = nn.Linear(d_model, self.base_vocab)

        # VHM center (EMA-updated)
        self.register_buffer("center", torch.zeros(d_model))
        self.center_momentum = center_momentum

        # Store a few hyperparams for convenience
        self.mask_ratio = mask_ratio
        self.alpha_vhm = alpha_vhm
        self.top_g = top_g
        self.r_threshold = r_threshold
        self.lr = lr

        # Test metrics (confusion counts)
        self.test_tp = 0
        self.test_fp = 0
        self.test_tn = 0
        self.test_fn = 0

        # Test bookkeeping for full report
        self._test_preds = []
        self._test_labels = []

        self.log_file_path = log_file_path

    # ---------- Helpers ----------

    def _prepend_dist(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend DIST token id to each sequence. x: [B, L] -> [B, L+1]"""
        B = x.size(0)
        dist_col = torch.full((B, 1), self.dist_id,
                              dtype=torch.long, device=x.device)
        return torch.cat([dist_col, x], dim=1)

    def _random_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask a fraction of positions 1..T-1 (skip position 0 which is DIST).
        Returns (masked_input, mask_bool) matching x's shape.
        """
        B, T = x.shape
        mask = torch.zeros_like(x, dtype=torch.bool)
        if T > 1:
            inner = torch.rand(B, T - 1, device=x.device) < self.mask_ratio
            # make sure each sequence has at least one mask (stability)
            need_one = inner.sum(dim=1) == 0
            if need_one.any():
                idx = torch.randint(
                    0, T - 1, (need_one.sum(),), device=x.device)
                inner[need_one, idx] = True
            mask[:, 1:] = inner

        x_masked = x.clone()
        x_masked[mask] = self.mask_id
        return x_masked, mask

    def _encode(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """Embed + posenc + transformer. seq_ids: [B, T] -> hidden: [B, T, D]"""
        h = self.embed(seq_ids)
        h = self.posenc(h)
        h = self.encoder(h)
        return h

    # ---------- Losses ----------
    def _mlkp_loss(self, h: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy over masked positions only.
        h: [B, T, D], targets: [B, T] original token ids, mask: [B, T] bool
        """
        if mask.any():
            logits = self.mlm_head(h[mask])                       # [M, V]
            gold = targets[mask].clamp_max(
                self.base_vocab - 1)   # ensure in 0..V-1
            return F.cross_entropy(logits, gold)
        else:
            return torch.tensor(0.0, device=h.device)

    def _vhm_loss(self, h_dist: torch.Tensor) -> torch.Tensor:
        # Use a detached snapshot so backward doesn't track/require the buffer
        center_target = self.center.detach().unsqueeze(0).expand_as(h_dist)
        return F.mse_loss(h_dist, center_target)

    # ---------- Lightning flow ----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoded states; position 0 is the DIST token."""
        x = self._prepend_dist(x)
        return self._encode(x)

    def training_step(self, batch, batch_idx):
        seq = batch["seq"]
        seq_with_dist = self._prepend_dist(seq)
        masked_input, mask = self._random_mask(seq_with_dist)

        h = self._encode(masked_input)
        h_dist = h[:, 0, :]

        mlkp = self._mlkp_loss(h, seq_with_dist, mask)
        vhm = self._vhm_loss(h_dist)
        loss = mlkp + self.alpha_vhm * vhm

        # ---- DEFER center update until after backward ----
        # Cache the batch mean (detached) for a safe post-backward EMA update
        self._center_batch_mean = h_dist.detach().mean(dim=0)

        self.log_dict({"train_mlkp": mlkp, "train_vhm": vhm,
                      "train_loss": loss}, prog_bar=True)
        return loss

    def on_after_backward(self) -> None:
        # Perform the EMA update AFTER gradients are computed
        if hasattr(self, "_center_batch_mean"):
            with torch.no_grad():
                m = self.center_momentum
                # same EMA as before, but now it's post-backward -> safe
                self.center.mul_(m).add_(self._center_batch_mean * (1 - m))
            del self._center_batch_mean

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        seq = batch["seq"]

        seq_with_dist = self._prepend_dist(seq)
        masked_input, mask = self._random_mask(seq_with_dist)

        h = self._encode(masked_input)
        h_dist = h[:, 0, :]

        mlkp = self._mlkp_loss(h, seq_with_dist, mask)
        vhm = self._vhm_loss(h_dist)
        loss = mlkp + self.alpha_vhm * vhm

        self.log_dict({"val_mlkp": mlkp, "val_vhm": vhm,
                      "val_loss": loss}, prog_bar=True)
        return loss

    # ---------- Detection utilities ----------
    @torch.no_grad()
    def _seq_is_anomalous(self, seq: torch.Tensor) -> bool:
        """
        Decide anomaly via top-g and r:
        - Randomly mask positions
        - If the true key is NOT in top-g at >= r positions -> anomalous
        """
        seq = seq.unsqueeze(0)  # [1, L]
        seq_with_dist = self._prepend_dist(seq)   # [1, L+1]
        masked_input, mask = self._random_mask(seq_with_dist)

        h = self._encode(masked_input)           # [1, L+1, D]
        logits = self.mlm_head(h)                # [1, L+1, V]
        probs = F.softmax(logits, dim=-1)        # [1, L+1, V]

        gold = seq_with_dist
        masked_positions = mask[0].nonzero(as_tuple=False).squeeze(-1)  # [M]
        misses = 0
        for pos in masked_positions.tolist():
            gold_id = int(gold[0, pos].item())
            # skip specials (shouldn’t be masked)
            if gold_id >= self.base_vocab:
                continue
            top_g_ids = probs[0, pos].topk(self.top_g).indices.tolist()
            if gold_id not in top_g_ids:
                misses += 1

        return misses >= self.r_threshold  # flag if misses reach r or more

    def on_test_start(self) -> None:
        self.test_tp = self.test_fp = self.test_tn = self.test_fn = 0
        self._test_preds = []
        self._test_labels = []

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        batch must contain:
          - 'seq': LongTensor [B, L]
          - 'label': LongTensor [B]  (0 = normal, 1 = anomalous)
        """
        seq = batch["seq"]
        labels = batch["label"]
        B = seq.size(0)

        preds = []
        for i in range(B):
            is_anom = self._seq_is_anomalous(seq[i])
            preds.append(1 if is_anom else 0)
        preds = torch.tensor(preds, device=seq.device)

        # Collect for full classification report
        self._test_preds.extend(preds.tolist())
        self._test_labels.extend(labels.tolist())

        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        self.test_tp += tp
        self.test_fp += fp
        self.test_tn += tn
        self.test_fn += fn

    def on_test_epoch_end(self) -> None:
        # Scalar metrics as before
        tp, fp, tn, fn = self.test_tp, self.test_fp, self.test_tn, self.test_fn
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        self.log_dict({"test_precision": precision,
                      "test_recall": recall, "test_f1": f1}, prog_bar=True)

        # ---- Full classification report (0 = normal, 1 = anomalous) ----
        y_true = self._test_labels
        y_pred = self._test_preds

        # 1) Dict form for programmatic metrics (accuracy, macro-F1)
        report_dict = classification_report(
            y_true, y_pred,
            labels=[0, 1],
            target_names=["normal", "anomalous"],
            digits=4,
            zero_division=0,
            output_dict=True
        )
        accuracy = report_dict['accuracy']
        macro_f1 = report_dict['macro avg']['f1-score']

        # Optionally log as numeric metrics to the trainer
        self.log_dict({
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1
        })

        # 2) String form for pretty print
        report = classification_report(
            y_true, y_pred,
            labels=[0, 1],
            target_names=["normal", "anomalous"],
            digits=4,
            zero_division=0
        )

        # Print to console
        self.print("\n=== Test Classification Report ===\n" + report)

        # Log to TensorBoard, if available
        try:
            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_text"):
                self.logger.experiment.add_text(
                    "test/classification_report", f"```\n{report}\n```", global_step=self.global_step)
        except Exception:
            pass

        # 3) Also append to a plain-text log file next to checkpoints
        try:
            # Prefer the explicit path if provided; otherwise derive from logger directory
            if self.log_file_path:
                log_path = Path(self.log_file_path)
            else:
                base_dir = Path(
                    self.trainer.logger.log_dir) if self.trainer and self.trainer.logger else Path(".")
                log_path = base_dir / "logs" / "run.log"

            log_path.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now().isoformat(timespec="seconds")

            # Format accuracy and macro-F1 with comma decimal separator
            accuracy_str = f"{accuracy:.4f}".replace('.', ',')
            macro_f1_str = f"{macro_f1:.4f}".replace('.', ',')

            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write(f"=== Test Summary ({now}) ===\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall:    {recall:.4f}\n")
                f.write(f"F1:        {f1:.4f}\n")
                f.write(
                    f"Accuracy & Macro F1: {accuracy_str} & {macro_f1_str}\n")
                f.write("\nClassification Report:\n")
                f.write(report)
                f.write("\n")
        except Exception:
            # Don't break the run for logging errors
            pass

    # ---------- Optimizer ----------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # Add inside LogBERTLightning
    @torch.no_grad()
    def _count_misses_for_seq(self, seq: torch.Tensor, g: int) -> int:
        """Return number of masked positions where gold ∉ top-g."""
        seq1 = seq.unsqueeze(0)                         # [1, L]
        seq_with_dist = self._prepend_dist(seq1)        # [1, L+1]
        masked_input, mask = self._random_mask(seq_with_dist)

        h = self._encode(masked_input)                  # [1, L+1, D]
        logits = self.mlm_head(h)                       # [1, L+1, V]
        probs = torch.softmax(logits, dim=-1)

        gold = seq_with_dist
        positions = mask[0].nonzero(as_tuple=False).squeeze(-1).tolist()
        misses = 0
        for pos in positions:
            gid = int(gold[0, pos].item())
            if gid >= self.base_vocab:
                continue
            top_ids = probs[0, pos].topk(g).indices.tolist()
            if gid not in top_ids:
                misses += 1
        return misses

    @torch.no_grad()
    def tune_r_g_on_val(self, val_loader, g_grid=(10, 20, 30), r_grid=(3, 4, 5), target_fpr: float = 0.01):
        """
        Paper-faithful calibration: choose (g, r) for the count rule using normal-only val.
        Picks the lowest FPR; among ties prefers larger r (more conservative) then larger g.
        """
        self.eval()
        best = None  # (fpr, -r, -g, g, r)
        for g in g_grid:
            for r in r_grid:
                fp = tn = 0
                for batch in val_loader:
                    seqs = batch["seq"].to(self.device)
                    for i in range(seqs.size(0)):
                        misses = self._count_misses_for_seq(seqs[i], g=g)
                        is_anom = (misses >= r)  # paper rule
                        if is_anom:
                            fp += 1
                        else:
                            tn += 1
                fpr = fp / (fp + tn + 1e-8)
                cand = (abs(fpr - target_fpr), -r, -g, g, r, fpr)
                if (best is None) or (cand < best):
                    best = cand

        _, _, _, g_best, r_best, fpr_best = best
        self.top_g = int(g_best)
        self.r_threshold = int(r_best)
        self.print(
            f"[Calibration] Set top_g={self.top_g}, r_threshold={self.r_threshold} (val FPR≈{fpr_best:.3%})")
