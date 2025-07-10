import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

def create_one_hot_encoding(structured_log_path: str, output_path: str):
    """
    Reads structured log data (e.g., from Drain), performs one-hot encoding
    on the 'EventId' column as required for DeepLog, and saves the result.

    Args:
        structured_log_path (str):
            Path to the structured log CSV file. This file must contain
            an 'EventId' column that represents the template for each log line.
        output_path (str):
            Path to save the resulting one-hot encoded numpy array (e.g., 'features.npy').
    """
    print(f"Step 1: Reading structured log file from '{structured_log_path}'...")
    try:
        df = pd.read_csv(structured_log_path)
    except FileNotFoundError:
        print(f"Error: The file '{structured_log_path}' was not found.")
        print("Please ensure the path to your structured log file is correct.")
        return

    print("Successfully loaded the CSV file. Here are the first 5 rows:")
    print(df.head())

    if 'EventId' not in df.columns:
        print("Error: 'EventId' column not found in the CSV file.")
        print("Please ensure your CSV has a column named 'EventId'.")
        return

    # The 'EventId' column contains the template identifiers we need to encode.
    # Scikit-learn's OneHotEncoder expects a 2D array, so we reshape it.
    event_ids = df[['EventId']]
    print(f"\nStep 2: Extracting the 'EventId' column for encoding. Found {len(event_ids)} log entries.")

    # Initialize the OneHotEncoder.
    # - sparse_output=False creates a standard (dense) numpy array.
    # - handle_unknown='ignore' ensures that if we encounter a new EventId during
    #   testing (that wasn't seen during training), it's encoded as a vector of all zeros.
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    print("\nStep 3: Fitting the encoder to the data and transforming it into one-hot vectors...")
    # The fit_transform method learns the unique EventIds and creates the one-hot matrix.
    one_hot_encoded_matrix = encoder.fit_transform(event_ids)

    # The shape of the matrix is (number of log lines, number of unique templates).
    num_logs, num_unique_templates = one_hot_encoded_matrix.shape
    print("Encoding complete.")
    print(f" -> Shape of the resulting matrix: {one_hot_encoded_matrix.shape}")
    print(f" -> This corresponds to {num_logs} log entries and {num_unique_templates} unique templates.")

    print("\nHere is a sample of the first 5 encoded vectors:")
    print(one_hot_encoded_matrix[:5])

    print(f"\nStep 4: Saving the one-hot encoded matrix to '{output_path}'...")
    np.save(output_path, one_hot_encoded_matrix)
    print("File saved successfully.")

    # You can also inspect the unique templates (categories) the encoder found
    # print("\nUnique templates discovered by the encoder:")
    # print(encoder.categories_[0])


def create_dummy_data(file_path: str):
    """Creates a dummy structured.csv file for demonstration purposes."""
    print(f"Creating a dummy data file at '{file_path}' for demonstration.")
    dummy_data = {
        'LineId': [1, 2, 3, 4, 5, 6, 7],
        'Content': [
            "Trojaned version of file '/bin/diff' detected.",
            "ossec: Manager started.",
            "Trojaned version of file '/usr/bin/diff' detected.",
            "<TIMESTAMP> status <*> wazuh-dashboard:amd64 4.5.2-1",
            "ossec: Manager started.",
            "<TIMESTAMP> status <*> telnet:amd64 0.17-44build1",
            "Trojaned version of file '/bin/diff' detected."
        ],
        'EventId': ['331a4559', 'e1a1f4c3', '331a4559', 'b4ac99bb', 'e1a1f4c3', '8a1531a5', '331a4559'],
        'EventTemplate': [
            'Trojaned version of file <*> detected.',
            'ossec: Manager started.',
            'Trojaned version of file <*> detected.',
            '<TIMESTAMP> status <*> wazuh-dashboard:amd64 <*>',
            'ossec: Manager started.',
            '<TIMESTAMP> status <*> telnet:amd64 <*>',
            'Trojaned version of file <*> detected.'
        ]
    }
    df_dummy = pd.DataFrame(dummy_data)
    df_dummy.to_csv(file_path, index=False)


if __name__ == '__main__':
    # --- Configuration ---
    # This is the path to your structured log file from Drain, Spell, etc.
    # We will create a dummy file here for a runnable example.
    INPUT_CSV_PATH = 'drain_structured.csv'

    # This is the path where the final numpy array will be saved.
    OUTPUT_NPY_PATH = 'deeplog_one_hot_features.npy'

    # --- Execution ---
    # 1. Create a dummy file to run the script.
    #    In your real workflow, you would replace this with your actual data file.
    create_dummy_data(INPUT_CSV_PATH)

    # 2. Run the main encoding function.
    print("\n" + "="*50)
    print("Starting One-Hot Encoding Process for DeepLog")
    print("="*50)
    create_one_hot_encoding(
        structured_log_path=INPUT_CSV_PATH,
        output_path=OUTPUT_NPY_PATH
    )
    print("="*50)

    # --- Verification (Optional) ---
    # You can load the file back to verify its contents.
    if os.path.exists(OUTPUT_NPY_PATH):
        print(f"\nVerification: Loading '{OUTPUT_NPY_PATH}' back into memory.")
        loaded_features = np.load(OUTPUT_NPY_PATH)
        print(f"Successfully loaded matrix with shape: {loaded_features.shape}")

    # --- Cleanup ---
    # Clean up the dummy files created for the demonstration.
    if os.path.exists(INPUT_CSV_PATH):
        os.remove(INPUT_CSV_PATH)
    if os.path.exists(OUTPUT_NPY_PATH):
        os.remove(OUTPUT_NPY_PATH)