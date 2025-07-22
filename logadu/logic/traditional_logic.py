# /logadu/logic/traditional_logic.py

import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import joblib
import os
import click
from sklearn.ensemble import RandomForestClassifier # ADD THIS
from sklearn.model_selection import train_test_split # ADD THIS
from sklearn.neighbors import KNeighborsClassifier

def _prepare_data_from_vectors(vector_file):
    """ Loads and prepares data for sklearn models by averaging sequence vectors. """
    click.echo(f"Loading and preparing data from {vector_file}...")
    data = torch.load(vector_file)
    sequences = data['sequences']
    labels = [lbl.item() for lbl in data['labels']]

    # --- Feature Engineering: Average vectors in each sequence ---
    feature_vectors = [torch.mean(seq, dim=0).numpy() for seq in sequences]
    
    X = np.array(feature_vectors)
    y = np.array(labels)
    
    click.echo(f"Data prepared. Shape of feature matrix X: {X.shape}")
    return X, y

def evaluate_pca(vector_file, output_dir):
    """ Trains and evaluates a PCA model for anomaly detection. """
    X, y = _prepare_data_from_vectors(vector_file)
    
    # Split data: train PCA only on normal data
    X_normal = X[y == 0]
    
    # We don't need a massive training set for PCA
    X_train_normal, _ = train_test_split(X_normal, train_size=min(len(X_normal)-1, 20000), random_state=42)

    click.echo(f"Fitting PCA model on {len(X_train_normal)} normal samples...")
    # Let PCA determine the number of components to explain 95% of the variance
    pca = PCA(n_components=0.95, random_state=42)
    pca.fit(X_train_normal)

    click.echo(f"PCA fitted. Number of components chosen: {pca.n_components_}")

    # --- Anomaly Detection ---
    # Calculate reconstruction error for the entire dataset
    click.echo("Calculating reconstruction error for all samples...")
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    reconstruction_errors = np.mean((X - X_reconstructed)**2, axis=1)
    
    # Set anomaly threshold: A common heuristic is to use a high percentile of the
    # reconstruction errors from the normal training data.
    train_reconstruction_errors = np.mean((X_train_normal - pca.inverse_transform(pca.transform(X_train_normal)))**2, axis=1)
    threshold = np.percentile(train_reconstruction_errors, 99)
    click.echo(f"Anomaly threshold (99th percentile of normal error) set to: {threshold:.6f}")

    # Predict anomalies
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # --- Evaluate ---
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary')
    click.secho("\n--- PCA Evaluation Results ---", fg="green")
    click.echo(f"Precision: {precision:.4f}")
    click.echo(f"Recall:    {recall:.4f}")
    click.echo(f"F1-Score:  {f1:.4f}")
    
    # --- Save the trained model ---
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "pca_model.joblib")
    joblib.dump(pca, model_path)
    click.echo(f"Saved trained PCA model to: {model_path}")
    


# ... (keep the existing _prepare_data_from_vectors and evaluate_pca functions)

def evaluate_rf(vector_file, output_dir):
    """ Trains and evaluates a Random Forest model for anomaly detection. """
    X, y = _prepare_data_from_vectors(vector_file)
    
    # --- Data Splitting (Crucial for Supervised Models) ---
    # We split the data into a training set (70%) and a testing set (30%).
    # `stratify=y` is very important: it ensures the test set has the same
    # percentage of anomalies as the training set, which is critical for imbalanced data.
    click.echo("Splitting data into training (70%) and testing (30%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    click.echo(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # --- Model Training ---
    # n_estimators=100 is a good default number of trees.
    # n_jobs=-1 tells sklearn to use all available CPU cores, which speeds up training.
    click.echo("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    click.echo("Model training complete.")

    # --- Prediction ---
    click.echo("Making predictions on the test set...")
    predictions = rf.predict(X_test)
    
    # --- Evaluation ---
    # We evaluate the model's performance ONLY on the unseen test data.
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
    click.secho("\n--- Random Forest Evaluation Results ---", fg="green")
    click.echo(f"Precision: {precision:.4f}")
    click.echo(f"Recall:    {recall:.4f}")
    click.echo(f"F1-Score:  {f1:.4f}")
    
    # --- Save the trained model ---
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "random_forest_model.joblib")
    joblib.dump(rf, model_path)
    click.echo(f"Saved trained Random Forest model to: {model_path}")
    
def evaluate_knn(vector_file, output_dir):
    """ Trains and evaluates a K-Nearest Neighbors model for anomaly detection. """
    X, y = _prepare_data_from_vectors(vector_file)
    
    # --- Data Splitting (Same as for Random Forest) ---
    click.echo("Splitting data into training (70%) and testing (30%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    click.echo(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # --- Model Training ---
    # n_neighbors=5 is a common and effective default for 'k'.
    # This is a key hyperparameter you could expose as a CLI option later.
    click.echo("Training K-Nearest Neighbors model (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    # For KNN, "fit" is very fast as it's a "lazy" learner that just stores the data.
    knn.fit(X_train, y_train)
    click.echo("Model training complete.")

    # --- Prediction ---
    # This step can be slow for KNN, as it calculates distances to the training data.
    click.echo("Making predictions on the test set...")
    predictions = knn.predict(X_test)
    
    # --- Evaluation ---
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
    click.secho("\n--- K-Nearest Neighbors (KNN) Evaluation Results ---", fg="green")
    click.echo(f"Precision: {precision:.4f}")
    click.echo(f"Recall:    {recall:.4f}")
    click.echo(f"F1-Score:  {f1:.4f}")
    
    # --- Save the trained model ---
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "knn_model.joblib")
    joblib.dump(knn, model_path)
    click.echo(f"Saved trained KNN model to: {model_path}")