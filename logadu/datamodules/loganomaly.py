import click
import pytorch_lightning as pl
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

class LogAnomalyDataModule(pl.LightningDataModule):
    """
    A specific DataModule for the LogAnomaly model.

    It creates a dataset that yields tuples of five tensors for each sample:
    1.  seq_vectors (input): A sequence of semantic vectors for the window.
    2.  count_vectors (input): A sequence of cumulative count vectors for the window.
    3.  next_log_index (target): The integer index of the log template that comes after the window.
    4.  next_log_vector (target): The semantic vector of the log that comes after the window.
    5.  anomaly_label (ground truth): The binary anomaly label (0 or 1) for the entire window + next log.
    """
    def __init__(self, merged_file: str, vector_map_file: str, 
                 window_size: int, batch_size: int = 256, 
                 num_workers: int = 4):
        super().__init__()
        self.merged_file = merged_file
        self.vector_map_file = vector_map_file
        self.window_size = window_size # This is the input sequence length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.semantic_input_dim = None
        self.num_templates = None
        self._data_prepared = False

    def setup(self, stage: str = None):
        if self._data_prepared:
            return
        
        # --- 1. Load Input Files and Create Mappings ---
        click.echo("--- Loading raw data and building mappings ---")
        df = pd.read_csv(self.merged_file, low_memory=False)
        vector_map = torch.load(self.vector_map_file)
        self.semantic_input_dim = next(iter(vector_map.values())).shape[0]

        # Create a mapping from EventId to a unique integer index (0, 1, 2, ...)
        # This is crucial for the count vectors.
        unique_templates = df['EventId'].unique()
        self.template_map = {template: i for i, template in enumerate(unique_templates)}
        self.num_templates = len(unique_templates)
        
        # --- 2. Generate Sequences and All Required Tensors ---
        click.echo(f"--- Generating dual-stream sequences (window_size={self.window_size}) ---")
        
        all_seq_inputs = []
        all_quant_inputs = []
        all_next_log_indices = []
        all_next_log_vectors = []
        all_anomaly_labels = []

        # The loop iterates up to the last possible window
        iterator = range(len(df) - self.window_size)
        for i in tqdm(iterator, desc="Processing windows"):
            # The full window includes the target log event
            full_window_df = df.iloc[i : i + self.window_size + 1]
            event_ids = full_window_df['EventId'].tolist()
            
            input_event_ids = event_ids[:-1]
            target_event_id = event_ids[-1]

            # --- Create Sequential Input ---
            seq_input = torch.stack([
                vector_map.get(eid, torch.zeros(self.semantic_input_dim)) for eid in input_event_ids
            ])
            all_seq_inputs.append(seq_input)
            
            # --- Create Quantitative Input (Cumulative Counts) ---
            count_sequence = []
            current_counts = torch.zeros(self.num_templates)
            for eid in input_event_ids:
                template_idx = self.template_map.get(eid)
                if template_idx is not None:
                    current_counts[template_idx] += 1
                count_sequence.append(current_counts.clone()) # Use clone!
            quant_input = torch.stack(count_sequence)
            all_quant_inputs.append(quant_input)

            # --- Create Targets ---
            target_idx = self.template_map.get(target_event_id, -1) # Use -1 for unseen templates
            all_next_log_indices.append(target_idx)
            
            target_vec = vector_map.get(target_event_id, torch.zeros(self.semantic_input_dim))
            all_next_log_vectors.append(target_vec)

            # --- Create Final Anomaly Label for Evaluation ---
            anomaly_label = 1 if full_window_df['label'].any() else 0
            all_anomaly_labels.append(anomaly_label)

        # --- 3. Convert lists to final tensors ---
        X_seq = torch.stack(all_seq_inputs)
        X_quant = torch.stack(all_quant_inputs)
        y_idx = torch.tensor(all_next_log_indices, dtype=torch.long)
        y_vec = torch.stack(all_next_log_vectors)
        y_anomaly = torch.tensor(all_anomaly_labels, dtype=torch.long)
        
        # --- 4. Perform Chronological Split ---
        click.echo("--- Performing chronological train-val-test split ---")
        dataset_size = len(X_seq)
        train_end_idx = int(dataset_size * 0.7)  # 70% for training
        val_end_idx = int(dataset_size * 0.85) # 15% for validation, 15% for testing

        self.train_dataset = TensorDataset(
            X_seq[:train_end_idx], X_quant[:train_end_idx], 
            y_idx[:train_end_idx], y_vec[:train_end_idx], 
            y_anomaly[:train_end_idx]
        )
        self.val_dataset = TensorDataset(
            X_seq[train_end_idx:val_end_idx], X_quant[train_end_idx:val_end_idx], 
            y_idx[train_end_idx:val_end_idx], y_vec[train_end_idx:val_end_idx], 
            y_anomaly[train_end_idx:val_end_idx]
        )
        self.test_dataset = TensorDataset(
            X_seq[val_end_idx:], X_quant[val_end_idx:], 
            y_idx[val_end_idx:], y_vec[val_end_idx:], 
            y_anomaly[val_end_idx:]
        )
        
        click.secho(f"Data setup complete. Found {self.num_templates} unique templates.", fg="green")
        self._data_prepared = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)