from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import click
import time
from typing import Union
import numpy as np

class KNNModel:
    def __init__(self, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
        self.X_train = X_train_tensor.cpu().numpy()
        self.y_train = y_train_tensor.cpu().numpy()
        self.X_test = X_test_tensor.cpu().numpy()
        self.y_test = y_test_tensor.cpu().numpy()

        click.secho("Training data shape: {}".format(self.X_train.shape), fg="green")
        click.secho("Testing data shape: {}".format(self.X_test.shape), fg="green")
    
    def train(self):
        self.model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        start_fit_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        end_fit_time = time.time()
        fit_duration = end_fit_time - start_fit_time
        click.secho(f"KNN model trained successfully in {fit_duration:.4f} seconds!", fg="green")
        self._evaluate()

    def _evaluate(self):
        start_eval_time = time.time()
        y_pred = self.model.predict(self.X_test)
        end_eval_time = time.time()
        eval_duration = end_eval_time - start_eval_time
        click.secho(f"KNN model evaluation completed in {eval_duration:.4f} seconds!", fg="green")
        
        report_dict = classification_report(
            self.y_test,
            y_pred,
            labels=[0, 1],
            target_names=['Normal (0)', 'Anomalous (1)'],
            zero_division=0,
            output_dict=True,
            digits=4
        )
        print("\n" + "-"*20 + " Metrics " + "-"*20)
        accuracy = report_dict['accuracy']
        macro_f1 = report_dict['macro avg']['f1-score']
        anomalous_f1 = report_dict['Anomalous (1)']['f1-score']
        anomalous_pre = report_dict['Anomalous (1)']['precision']
        anomalous_recal = report_dict['Anomalous (1)']['recall']
        accuracy_str = f"{accuracy:.4f}".replace('.', ',')
        macro_f1_str = f"{macro_f1:.4f}".replace('.', ',')
        anomalous_f1_str = f"{anomalous_f1:.4f}".replace('.', ',')
        anomalous_pre_str = f"{anomalous_pre:.4f}".replace('.', ',')
        anomalous_recal_str = f"{anomalous_recal:.4f}".replace('.', ',')
        print(f"Test Accuracy & Macro F1-Score: {accuracy_str} & {macro_f1_str}\n")
        print(f"Anomalous: {anomalous_pre_str} & {anomalous_recal_str} & {anomalous_f1_str}")

        print("-" * (49)) # Match the length of the header
        
        print("\n" + "="*60 + "\nTest Set Classification Report:\n" + "="*60)
        
        print(classification_report(
            self.y_test,
            y_pred,
            digits=4,
            target_names=['Normal (0)', 'Anomalous (1)'],
            zero_division=0,
            labels=[0, 1]
        ))
        print("="*60)
        
class RFModel:
    def __init__(self, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
        self.X_train = X_train_tensor.cpu().numpy()
        self.y_train = y_train_tensor.cpu().numpy()
        self.X_test = X_test_tensor.cpu().numpy()
        self.y_test = y_test_tensor.cpu().numpy()

        click.secho("Training data shape: {}".format(self.X_train.shape), fg="green")
        click.secho("Testing data shape: {}".format(self.X_test.shape), fg="green")

    def train(self):
        # Initialize the model with common, robust parameters
        self.model = RandomForestClassifier(
            n_estimators=100, # Number of trees in the forest
            random_state=42,  # For reproducibility
            n_jobs=-1         # Use all available CPU cores
        )
        
        # Time the fitting process
        start_fit_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        end_fit_time = time.time()
        fit_duration = end_fit_time - start_fit_time
        
        click.secho(f"Random Forest model trained successfully in {fit_duration:.4f} seconds!", fg="green")
        
        # Immediately evaluate the model after training
        self._evaluate()

    def _evaluate(self):
        start_eval_time = time.time()
        y_pred = self.model.predict(self.X_test)
        end_eval_time = time.time()
        eval_duration = end_eval_time - start_eval_time
        
        click.secho(f"Random Forest model evaluation completed in {eval_duration:.4f} seconds!", fg="green")
        
        # Generate classification report as a dictionary
        report_dict = classification_report(
            self.y_test,
            y_pred,
            labels=[0, 1],
            target_names=['Normal (0)', 'Anomalous (1)'],
            zero_division=0,
            output_dict=True,
            digits=4
        )
        
        # Print the custom metrics summary
        print("\n" + "-"*20 + " Metrics " + "-"*20)
        accuracy = report_dict['accuracy']
        macro_f1 = report_dict['macro avg']['f1-score']
        anomalous_f1 = report_dict['Anomalous (1)']['f1-score']
        anomalous_pre = report_dict['Anomalous (1)']['precision']
        anomalous_recal = report_dict['Anomalous (1)']['recall']
        accuracy_str = f"{accuracy:.4f}".replace('.', ',')
        macro_f1_str = f"{macro_f1:.4f}".replace('.', ',')
        anomalous_f1_str = f"{anomalous_f1:.4f}".replace('.', ',')
        anomalous_pre_str = f"{anomalous_pre:.4f}".replace('.', ',')
        anomalous_recal_str = f"{anomalous_recal:.4f}".replace('.', ',')
        print(f"Test Accuracy & Macro F1-Score: {accuracy_str} & {macro_f1_str}\n")
        print(f"Anomalous: {anomalous_pre_str} & {anomalous_recal_str} & {anomalous_f1_str}")
        print("-" * (49))
        
        # Print the full, formatted report
        print("\n" + "="*60 + "\nTest Set Classification Report:\n" + "="*60)
        print(classification_report(
            self.y_test,
            y_pred,
            digits=4,
            labels=[0, 1],
            target_names=['Normal (0)', 'Anomalous (1)'],
            zero_division=0
        ))
        print("="*60)
    
class OCSVMModel:
    def __init__(
        self,
        X_train_tensor,
        X_test_tensor,
        y_test_tensor,
        *,
        kernel: str = "rbf",
        nu: float = 0.05,          # expected fraction of anomalies (upper bound on training outliers)
        gamma: Union[str, float] = "scale"
    ):
        # Store raw arrays
        self.X_train = X_train_tensor.cpu().numpy()
        self.X_test  = X_test_tensor.cpu().numpy()
        self.y_test  = y_test_tensor.cpu().numpy().astype(int)

        # Flatten to 2D if sequences are 3D (N, W, D) or similar
        self.X_train = self._ensure_2d(self.X_train)
        self.X_test  = self._ensure_2d(self.X_test)

        # Keep params
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma

        click.secho(f"Training data shape: {self.X_train.shape}", fg="green")
        click.secho(f"Testing data shape:  {self.X_test.shape}",  fg="green")
    
    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1) if X.ndim > 2 else X
    
    def train(self):
        # Scale features (crucial for OCSVM with RBF/poly)
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        t0 = time.time()
        self.X_train_std = self.scaler.fit_transform(self.X_train)
        t1 = time.time()

        # Fit One-Class SVM on TRAIN ONLY (one-class)
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.model.fit(self.X_train_std)
        t2 = time.time()

        click.secho(f"Scaler fit in {(t1 - t0):.4f}s; OC-SVM trained in {(t2 - t1):.4f}s.", fg="green")
        self._evaluate()
        
    def _evaluate(self):
        # Transform test with the same scaler
        X_test_std = self.scaler.transform(self.X_test)

        t0 = time.time()
        # sklearn OCSVM outputs +1 for inliers (normal) and -1 for outliers (anomalies)
        pred_inlier_outlier = self.model.predict(X_test_std)
        # Convert to your convention: 0=normal, 1=anomaly
        y_pred = (pred_inlier_outlier == -1).astype(int)

        # Anomaly score (higher = more anomalous): negative signed distance to boundary
        # decision_function > 0 => inlier; < 0 => outlier
        decision_scores = self.model.decision_function(X_test_std)  # shape (N,)
        anomaly_scores = -decision_scores

        t1 = time.time()
        click.secho(f"OC-SVM evaluation completed in {(t1 - t0):.4f} seconds!", fg="green")

        # Report
        report_dict = classification_report(
            self.y_test,
            y_pred,
            labels=[0, 1],
            target_names=['Normal (0)', 'Anomalous (1)'],
            zero_division=0,
            output_dict=True,
            digits=4
        )

        print("\n" + "-"*20 + " Metrics " + "-"*20)
        accuracy = report_dict['accuracy']
        macro_f1 = report_dict['macro avg']['f1-score']
        anomalous_f1 = report_dict['Anomalous (1)']['f1-score']
        anomalous_pre = report_dict['Anomalous (1)']['precision']
        anomalous_recal = report_dict['Anomalous (1)']['recall']
        accuracy_str = f"{accuracy:.4f}".replace('.', ',')
        macro_f1_str = f"{macro_f1:.4f}".replace('.', ',')
        anomalous_f1_str = f"{anomalous_f1:.4f}".replace('.', ',')
        anomalous_pre_str = f"{anomalous_pre:.4f}".replace('.', ',')
        anomalous_recal_str = f"{anomalous_recal:.4f}".replace('.', ',')
        print(f"Test Accuracy & Macro F1-Score: {accuracy_str} & {macro_f1_str}\n")
        print(f"Anomalous: {anomalous_pre_str} & {anomalous_recal_str} & {anomalous_f1_str}")
        print("-" * 49)

        print("\n" + "="*60 + "\nTest Set Classification Report:\n" + "="*60)
        print(classification_report(
            self.y_test,
            y_pred,
            digits=4,
            labels=[0, 1],
            target_names=['Normal (0)', 'Anomalous (1)'],
            zero_division=0
        ))
        print("="*60)

class PCAModel:
    def __init__(self, X_train_tensor, X_test_tensor, y_test_tensor):
        self.X_train = X_train_tensor.cpu().numpy()
        self.X_test = X_test_tensor.cpu().numpy()
        self.y_test = y_test_tensor.cpu().numpy()

        click.secho("Training data shape: {}".format(self.X_train.shape), fg="green")
        click.secho("Testing data shape: {}".format(self.X_test.shape), fg="green")
    
    def train(self):
        self.model = PCA(n_components=0.95, random_state=42)
        start_fit_time = time.time()
        self.model.fit(self.X_train)
        end_fit_time = time.time()
        fit_duration = end_fit_time - start_fit_time
        click.secho(f"PCA model trained successfully in {fit_duration:.4f} seconds!", fg="green")
        self._evaluate()

    def _evaluate(self):
        X_train_reconstructed = self.model.inverse_transform(self.model.transform(self.X_train))
        reconstruction_errors_normal = np.mean((self.X_train - X_train_reconstructed)**2, axis=1)
        THRESHOLD_PERCENTILE = 95
        threshold = np.percentile(reconstruction_errors_normal, THRESHOLD_PERCENTILE)
        click.secho(f"Anomaly threshold (at {THRESHOLD_PERCENTILE}th percentile) is: {threshold:.6f}", fg="yellow")
        
        start_eval_time = time.time()
        X_test_reconstructed = self.model.inverse_transform(self.model.transform(self.X_test))
        reconstruction_errors_test = np.mean((self.X_test - X_test_reconstructed)**2, axis=1)
        y_pred = (reconstruction_errors_test > threshold).astype(int)
        
        end_eval_time = time.time()
        eval_duration = end_eval_time - start_eval_time
        click.secho(f"PCA model evaluation completed in {eval_duration:.4f} seconds!", fg="green")

        report_dict = classification_report(
            self.y_test,
            y_pred,
            labels=[0, 1],
            target_names=['Normal (0)', 'Anomalous (1)'],
            zero_division=0,
            output_dict=True,
            digits=4
        )
        print("\n" + "-"*20 + " Metrics " + "-"*20)
        accuracy = report_dict['accuracy']
        macro_f1 = report_dict['macro avg']['f1-score']
        anomalous_f1 = report_dict['Anomalous (1)']['f1-score']
        anomalous_pre = report_dict['Anomalous (1)']['precision']
        anomalous_recal = report_dict['Anomalous (1)']['recall']
        accuracy_str = f"{accuracy:.4f}".replace('.', ',')
        macro_f1_str = f"{macro_f1:.4f}".replace('.', ',')
        anomalous_f1_str = f"{anomalous_f1:.4f}".replace('.', ',')
        anomalous_pre_str = f"{anomalous_pre:.4f}".replace('.', ',')
        anomalous_recal_str = f"{anomalous_recal:.4f}".replace('.', ',')
        print(f"Test Accuracy & Macro F1-Score: {accuracy_str} & {macro_f1_str}\n")
        print(f"Anomalous: {anomalous_pre_str} & {anomalous_recal_str} & {anomalous_f1_str}")
        print("-" * (49)) # Match the length of the header
        
        print("\n" + "="*60 + "\nTest Set Classification Report:\n" + "="*60)
        
        print(classification_report(
            self.y_test,
            y_pred,
            digits=4,
            labels=[0, 1],
            target_names=['Normal (0)', 'Anomalous (1)'],
            zero_division=0
        ))
        print("="*60)
        