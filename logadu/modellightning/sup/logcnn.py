from typing import Dict, Tuple, Optional, Iterable
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from datetime import datetime
from pathlib import Path
from torchmetrics.classification import Accuracy


class LogCNNModel(pl.LightningModule):
    """
    Supervised LogCNN (indices):
      Input:  seq  -> LongTensor [B, L]   (log-key indices)
      Target: label -> LongTensor [B]     (0 normal, 1 anomalous)

    Architecture (paper):
      Embedding (logkey2vec) -> 3 x Conv1d(k=3,4,5) -> LeakyReLU -> max-over-time
      -> concat -> Dropout(0.5) -> Linear -> Softmax
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,          # your window_size
        emb_dim: int = 128,         # codebook size in paper
        num_filters: int = 128,     # filters per kernel (paper Table I)
        kernel_sizes: Iterable[int] = (3, 4, 5),
        dropout: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,  # set >0 if you want L2
        leaky_relu_slope: float = 0.1,
        num_classes: int = 2,       # normal/abnormal
        log_file_path: str = None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embed = nn.Embedding(vocab_size, emb_dim)

        # Conv blocks: Conv1d over sequence (time) with channels = emb_dim
        convs = []
        for k in kernel_sizes:
            convs.append(
                nn.Conv1d(
                    in_channels=emb_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=0,    # "VALID" in paper
                    bias=True,
                )
            )
        self.convs = nn.ModuleList(convs)

        self.act = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            num_filters * len(tuple(kernel_sizes)), num_classes)

        # buffers for test-time report
        self._test_preds = []
        self._test_labels = []

        # Accuracy via torchmetrics; use multiclass with 2 classes
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)

        self.log_file_path = log_file_path

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, L] -> logits: [B, num_classes]
        """
        x = self.embed(seq)                 # [B, L, D]
        x = x.transpose(1, 2)               # [B, D, L] for Conv1d

        # conv -> lrelu -> max-over-time pool
        feats = []
        for conv in self.convs:
            y = conv(x)                     # [B, C, L-k+1]
            y = self.act(y)
            y = F.max_pool1d(y, kernel_size=y.size(2)).squeeze(2)  # [B, C]
            feats.append(y)

        h = torch.cat(feats, dim=1)         # [B, C * len(kernels)]
        h = self.dropout(h)
        logits = self.fc(h)                  # [B, num_classes]
        return logits

    # ------- Lightning hooks -------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        seq = batch["seq"].long()
        label = batch["label"].long()
        logits = self(seq)
        loss = F.cross_entropy(logits, label)
        pred = logits.argmax(dim=-1)

        self.train_acc.update(pred, label)
        # Log loss per step and accuracy per epoch
        self.log("train_loss", loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        seq = batch["seq"].long()
        label = batch["label"].long()
        logits = self(seq)
        loss = F.cross_entropy(logits, label)
        pred = logits.argmax(dim=-1)

        self.val_acc.update(pred, label)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.val_acc.reset()

    @torch.no_grad()
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        seq = batch["seq"].long()
        label = batch["label"].long()
        logits = self(seq)
        pred = logits.argmax(dim=-1)
        self._test_preds.extend(pred.tolist())
        self._test_labels.extend(label.tolist())

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

        # cleanup
        self._test_preds.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
