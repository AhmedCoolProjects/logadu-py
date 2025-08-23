import pandas as pd
import torch
import numpy as np
from typing import Optional, List, Dict
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
import click
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class SlidingWindowPremappedDataset(Dataset):
    """
    A Dataset for NeuralLog that creates sliding windows over EventIds
    and uses a pre-computed map to look up the corresponding raw log content.
    
    This is highly memory-efficient for large datasets.
    """
    def __init__(self, event_id_sequence: np.ndarray, label_sequence: np.ndarray, window_size: int, eventid_to_content_map: Dict):
        if event_id_sequence.shape[0] != label_sequence.shape[0]:
            raise ValueError("EventId and label sequence length mismatch")
        
        self.event_id_sequence = event_id_sequence
        self.label_sequence = label_sequence
        self.window_size = window_size
        self.eventid_to_content_map = eventid_to_content_map
        self.num_windows = len(self.event_id_sequence) - window_size + 1
        
        if self.num_windows <= 0:
            raise ValueError("Not enough log events to form a single window.")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[List[str], int]:
        j = idx + self.window_size
        
        # 1. Get the window of EventIds
        event_id_window = self.event_id_sequence[idx:j]
        
        # 2. Look up the representative raw content for each EventId
        # Use a default empty string if an ID is somehow not in the map
        window_logs = [self.eventid_to_content_map.get(eid, "") for eid in event_id_window]
        
        # 3. Determine the sequence label
        seq_label = int(any(self.label_sequence[idx:j]))
        
        return window_logs, seq_label

class RawLogDataModule(pl.LightningDataModule):
    """
    DataModule for Raw that uses an efficient EventId-to-Content mapping.
    """
    def __init__(
        self,
        csv_file_path: str,
        col_content_name: str,
        col_label_name: str,
        col_eventid_name: str,
        window_size: int,
        bert_model_name: str = 'bert-base-uncased',
        batch_size: int = 16,
        max_log_len: int = 128,
        num_workers: int = 0,
        random_state: int = 42
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.bert_model_name)
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.eventid_to_content_map: Optional[Dict] = None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        click.secho("Setting up NeuralLogDataModule with EventId mapping...", fg="cyan")
        
        # 1. Load the full dataset from CSV
        df = pd.read_csv(
            self.hparams.csv_file_path,
            usecols=[self.hparams.col_content_name, self.hparams.col_label_name, self.hparams.col_eventid_name],
            low_memory=False
        )
        df[self.hparams.col_content_name] = df[self.hparams.col_content_name].astype(str)

        # 2. --- CRITICAL STEP: Create the EventId -> Content map ---
        # Keep the first occurrence of each unique EventId to get a representative content
        df_unique = df.drop_duplicates(subset=[self.hparams.col_eventid_name])
        self.eventid_to_content_map = dict(zip(df_unique[self.hparams.col_eventid_name], df_unique[self.hparams.col_content_name]))
        click.secho(f"Created map with {len(self.eventid_to_content_map)} unique EventIds.", fg="green")

        # 3. Get the full sequence of EventIds and labels
        event_id_sequence = df[self.hparams.col_eventid_name].to_numpy()
        label_sequence = df[self.hparams.col_label_name].to_numpy(dtype=int)
        
        # 4. Create the main sliding window dataset using the map
        full_dataset = SlidingWindowPremappedDataset(
            event_id_sequence=event_id_sequence,
            label_sequence=label_sequence,
            window_size=self.hparams.window_size,
            eventid_to_content_map=self.eventid_to_content_map
        )
        
        # 5. Perform supervised train/val/test split on window indices
        num_windows = len(full_dataset)
        indices = np.arange(num_windows)
        
        click.echo("Calculating window labels for stratified splitting...")
        window_labels = np.array([full_dataset[i][1] for i in range(num_windows)], dtype=np.int8)

        stratify_labels = window_labels if np.unique(window_labels).size > 1 else None

        idx_train_val, idx_test = train_test_split(
            indices, test_size=0.2, random_state=self.hparams.random_state, stratify=stratify_labels
        )
        stratify_train_val = window_labels[idx_train_val] if stratify_labels is not None else None
        idx_train, idx_val = train_test_split(
            idx_train_val, test_size=0.1, random_state=self.hparams.random_state, stratify=stratify_train_val
        )
        
        self.train_dataset = Subset(full_dataset, idx_train)
        self.val_dataset = Subset(full_dataset, idx_val)
        self.test_dataset = Subset(full_dataset, idx_test)
        
        click.secho(
            f"Setup complete: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}",
            fg="green"
        )
        
    def _collate_fn(self, batch: List[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # This function remains unchanged as it already handles batches of raw text
        batch_logs, batch_labels = zip(*batch)
        all_logs_flat = [log for seq in batch_logs for log in seq]
        
        tokenized = self.tokenizer(
            all_logs_flat, padding='max_length', truncation=True,
            max_length=self.hparams.max_log_len, return_tensors='pt'
        )
        
        batch_size = len(batch_logs)
        window_size = self.hparams.window_size
        input_ids = tokenized['input_ids'].view(batch_size, window_size, -1)
        attention_mask = tokenized['attention_mask'].view(batch_size, window_size, -1)
        labels = torch.tensor(batch_labels, dtype=torch.long)
        
        return input_ids, attention_mask, labels

    # --- Standard DataLoader methods ---
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, collate_fn=self._collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, collate_fn=self._collate_fn, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, collate_fn=self._collate_fn, pin_memory=True)