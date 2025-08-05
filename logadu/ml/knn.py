import click
import torch
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  classification_report
from tqdm import tqdm


def _load_and_aggregate_data(vector_file_path):
    """
    Load the vector sequences from .pt file and aggregating the vectors in the sequence
    to a single feature vector per sequence.
    
    ---
    Returns: Input Vector (X) and Labels (y) as numpy arrays.
    """
    click.echo(f"--- Step 1: Loading and Aggregating Data ---")
    click.echo(f"Loading event vectors from: {vector_file_path}")
    data = torch.load(vector_file_path)
    sequences = data['sequences']
    labels = [lbl.item() for lbl in data['labels']]
    click.echo(f"Found {len(sequences)} sequences. Now aggregating...")
    feature_vectors = [torch.mean(seq, dim=0).numpy() for seq in tqdm(sequences, desc="Aggregating sequences")]
    X = np.array(feature_vectors)
    y = np.array(labels)
    click.secho(f"Data preparation complete. Feature matrix shape: {X.shape}", fg="green")
    return X, y

def run_knn(vector_file, k_neighbors, n_splits):
    """
    Trains and evaluates a KNN model using Time Series Cross-Validation and
    provides a detailed classification report.
    """
    # 1. Load and prepare the feature matrix
    X, y = _load_and_aggregate_data(vector_file)

    # 2. Set up Time Series Cross-Validation
    click.echo(f"\n--- Step 2: Setting up Time Series Cross-Validation with {n_splits} splits ---")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # --- NEW: Lists to store predictions and true labels from all folds ---
    all_true_labels = np.array([])
    all_predictions = np.array([])

    # 3. Loop through each fold for training and evaluation
    click.echo(f"\n--- Step 3: Running Cross-Validation ---")
    fold_iterator = tqdm(list(tscv.split(X)), desc="Cross-Validation Folds")
    
    for fold, (train_index, test_index) in enumerate(fold_iterator):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize and train the model for this fold
        knn_model = KNeighborsClassifier(n_neighbors=k_neighbors, n_jobs=-1)
        knn_model.fit(X_train, y_train)
        
        # Predict on the current fold's test data
        predictions = knn_model.predict(X_test)
        
        # --- NEW: Append the results of this fold for final report ---
        all_true_labels = np.append(all_true_labels, y_test)
        all_predictions = np.append(all_predictions, predictions)

    # --- 4. Generate and display the final, detailed classification report ---
    click.echo("\n" + "="*55)
    click.secho(f"  Time Series Cross-Validation (KNN) - Final Report", bold=True)
    click.echo("="*55)
    
    click.echo(f"  Hyperparameters:")
    click.echo(f"    - K-Neighbors        : {k_neighbors}")
    click.echo(f"    - Number of Folds    : {n_splits}")
    click.echo(f"-"*55)
    click.secho(f"  Aggregated Performance Metrics Across All Folds:", bold=True)
    
    # Define class labels for the report
    target_names = ['Normal', 'Anomalous']
    
    # Use classification_report to get the detailed metrics
    report = classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=target_names, 
        digits=4,
        zero_division=0
    )
    
    click.echo(report)
    click.echo("="*55)
