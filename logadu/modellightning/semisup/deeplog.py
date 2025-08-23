from typing import Dict, Tuple, Optional, Iterable
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from datetime import datetime
from pathlib import Path


class DeepLogModel(pl.LightningModule):
    """
    DeepLog (indices version):
      - Input: window of log-key indices (length h == window_size)
      - Target: next log-key index
      - Train (semi-supervised): normal-only windows
      - Detect: normal if gold next ∈ top-g; else anomalous
    """

    def __init__(
        self,
        vocab_size: int,
        h: int,                 # window_size
        emb_dim: int = 128,
        hidden_dim: int = 128,       # α in the paper
        num_layers: int = 2,         # L
        dropout: float = 0.1,
        top_g: int = 9,              # g (default per paper)
        lr: float = 1e-3,
        log_file_path: str = None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.h = h
        self.top_g = top_g
        self.lr = lr

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

        # test-time bookkeeping
        self._test_preds: list[int] = []
        self._test_labels: list[int] = []

        self.log_file_path = log_file_path

    # ----- core forward -----
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, h] integers
        return: logits for next key, shape [B, vocab_size]
        """
        x = self.embed(seq)            # [B, h, D]
        out, _ = self.lstm(x)          # [B, h, H]
        last = out[:, -1, :]           # [B, H]
        logits = self.head(last)       # [B, V]
        return logits

    # ----- training/validation -----
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        seq = batch["seq"].long()      # [B, h]
        nxt = batch["next"].long()     # [B]
        logits = self(seq)
        loss = F.cross_entropy(logits, nxt)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        seq = batch["seq"].long()
        nxt = batch["next"].long()
        logits = self(seq)
        loss = F.cross_entropy(logits, nxt)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    # ----- test: top-g membership => anomaly -----
    @torch.no_grad()
    def _is_anomalous_window(self, seq: torch.Tensor, gold_next: int) -> bool:
        """
        seq: [h], gold_next: int
        rule: anomalous if gold_next ∉ top-g(next-logits)
        """
        seq = seq.unsqueeze(0)           # [1, h]
        logits = self(seq)               # [1, V]
        top_ids = logits.topk(self.top_g, dim=-1).indices[0].tolist()
        return int(gold_next) not in top_ids

    def on_test_start(self) -> None:
        self._test_preds = []
        self._test_labels = []

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        seq = batch["seq"].long()        # [B, h]
        nxt = batch["next"].long()       # [B]
        y = batch["label"].long()        # [B]  (0 normal, 1 anomalous)

        B = seq.size(0)
        preds = []
        for i in range(B):
            is_anom = self._is_anomalous_window(seq[i], int(nxt[i].item()))
            preds.append(1 if is_anom else 0)

        self._test_preds.extend(preds)
        self._test_labels.extend(y.tolist())

    def on_test_epoch_end(self) -> None:

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
                f.write(
                    f"Accuracy & Macro F1: {accuracy_str} & {macro_f1_str}\n")
                f.write("\nClassification Report:\n")
                f.write(report)
                f.write("\n")
        except Exception:
            # Don't break the run for logging errors
            pass

    # ----- optimizer -----
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def tune_g_on_val(self, val_loader, g_grid: Iterable[int] = (10, 20, 30), target_fpr: float = 0.01):
        """
        Choose g for the top-g rule using normal-only validation to target a small FPR.
        Keeps the rest of the method unchanged.
        """
        self.eval()
        best = None  # (|fpr - target|, g, fpr)
        for g in g_grid:
            fp = tn = 0
            for batch in val_loader:
                seq = batch["seq"].to(self.device)
                nxt = batch["next"].to(self.device)
                # validation labels are normal-only in your setup
                for i in range(seq.size(0)):
                    is_anom = self._is_anomalous_window(
                        seq[i], int(nxt[i].item()))
                    if is_anom:
                        fp += 1
                    else:
                        tn += 1
            fpr = fp / (fp + tn + 1e-8)
            cand = (abs(fpr - target_fpr), g, fpr)
            if (best is None) or (cand < best):
                best = cand
        _, g_best, fpr_best = best
        self.top_g = int(g_best)
        self.print(
            f"[Calibration] Set top_g={self.top_g} (val FPR≈{fpr_best:.3%})")
