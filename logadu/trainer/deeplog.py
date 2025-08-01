import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import ast
from tqdm import tqdm
from logadu.models.deeplog import DeepLog



def _load_and_split_data(file_path):
    """ Loads the CSV, splits into train/test, and prepares tensors. """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Safely convert string representation of lists
    df['sequence'] = df['sequence'].apply(ast.literal_eval)

    # Split the dataset into training (70%) and testing (30%) sets
    # We use stratify on the 'label' to ensure both sets have a similar anomaly ratio
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    
    # --- CRITICAL STEP FOR DEEPLOG ---
    # The final training set must only contain NORMAL sequences
    train_df_normal = train_df[train_df['label'] == 0].copy()
    
    print(f"Total data: {len(df)} sequences.")
    print(f"Training on: {len(train_df_normal)} NORMAL sequences.")
    print(f"Testing on: {len(test_df)} sequences (includes normal and anomalous).")

    # Determine vocabulary size from the entire dataset to handle all possible keys
    all_keys = set(df['next'].unique())
    for seq in df['sequence']:
        all_keys.update(seq)
    vocab_size = max(all_keys) + 1
    print(f"Calculated vocabulary size: {vocab_size}")

    # Convert to PyTorch Tensors
    X_train = torch.tensor(train_df_normal['sequence'].tolist(), dtype=torch.long)
    y_train = torch.tensor(train_df_normal['next'].tolist(), dtype=torch.long)
    
    X_test = torch.tensor(test_df['sequence'].tolist(), dtype=torch.long)
    y_test_next_event = torch.tensor(test_df['next'].tolist(), dtype=torch.long)
    y_test_anomaly_label = torch.tensor(test_df['label'].tolist(), dtype=torch.long)

    return (X_train, y_train), (X_test, y_test_next_event, y_test_anomaly_label), vocab_size

def train_deeplog(dataset_file, epochs, batch_size, top_k, output_model_path):
    """
    Trains and evaluates a DeepLog model on the provided sequence dataset.
    """
    # 1. Load and Prepare Data
    train_data, test_data, vocab_size = _load_and_split_data(dataset_file)
    X_train, y_train = train_data
    X_test, y_test_next_event, y_test_anomaly_label = test_data

    # 2. Set up training environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize and Train Model
    model = DeepLog(vocab_size=vocab_size, criterion=nn.CrossEntropyLoss()).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    print("\n--- Starting Model Training ---")
    model.train()
    for epoch in range(epochs):
        for seq, next_event in train_loader:
            batch = {'sequential': seq, 'label': next_event}
            optimizer.zero_grad()
            output = model(batch, device=device)
            loss = output.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}] finished.")
    
    model.save(output_model_path)
    print(f"Trained model saved to: {output_model_path}")

    # --- 4. Evaluation Phase ---
    print("\n--- Starting Evaluation ---")
    model.eval()
    
    # Create a DataLoader for the test set
    test_dataset = TensorDataset(X_test, y_test_next_event, y_test_anomaly_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for seq, next_event, anomaly_label in tqdm(test_loader, desc="Evaluating"):
            batch = {'sequential': seq}
            output = model(batch, device=device)
            
            # Get the top-k predictions from the model
            topk_preds = torch.topk(output.probabilities, k=top_k).indices

            # Check if the true next event is in the top-k predictions
            # .any(dim=1) checks this for every sequence in the batch
            is_in_topk = (next_event.to(device).unsqueeze(1) == topk_preds).any(dim=1)

            # Anomaly is when the true event is NOT in the top-k predictions
            predicted_labels = (~is_in_topk).long().cpu()

            all_predictions.extend(predicted_labels.tolist())
            all_true_labels.extend(anomaly_label.tolist())

    # --- 5. Calculate and Print Metrics ---
    print("\n--- Evaluation Results ---")
    # Using zero_division=0 handles cases where a class has no predictions
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_predictions, average='binary', zero_division=0
    )
    
    print(f"Top-k                : {top_k}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"F1-Score             : {f1:.4f}")

