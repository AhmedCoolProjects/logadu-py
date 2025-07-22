# In logadu/sequencing.py

import pandas as pd
from tqdm import tqdm
import pickle
import os
from pathlib import Path
import numpy as np

def generate_sequences(log_file, method, session_col=None, window_size=10, step_size=1):
    """
    Generates sequences from structured log data using either session-based or window-based grouping.

    Args:
        log_file (str): Path to the structured log CSV file.
        
        method (str): Grouping method, either 'session' or 'window'.
        session_col (str, optional): The column name for session grouping. Required if method is 'session'.
        window_size (int, optional): The size of the sliding window. Required if method is 'window'.
        step_size (int, optional): The step size for the sliding window. Required if method is 'window'.
    """
    print(f"Generating sequences using '{method}' method...")


    df = pd.read_csv(log_file)
    _parent = Path(log_file).parent
    _name = Path(log_file).stem
    _name = _name.split('_')[0]
    output_dir = _parent / 'sequences'
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'label' not in df.columns or 'EventId' not in df.columns:
        raise ValueError("Input CSV must contain 'label' and 'EventId' columns.")

    session_data = {}

    # TODO: not verified yet
    if method == 'session':
        if not session_col:
            raise ValueError("session_col must be provided for 'session' method.")
        session_data = _group_by_session(df, session_col)
    
    elif method == 'window':
        session_data = _group_by_window(df, window_size, step_size)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'session' or 'window'.")

    print(f"Generated {len(session_data)} sequences.")
    
    sequences_file = os.path.join(output_dir, f'{_name}_{window_size}_{step_size}.pkl')
    with open(sequences_file, 'wb') as f:
        pickle.dump(session_data, f)
    
    print(f"Sequences saved to {sequences_file}")

def _group_by_session(df, session_col):
    """ Helper function for session-based grouping. """
    session_data = {}
    sessions = df.groupby(session_col)
    for session_id, session_df in tqdm(sessions, desc="Grouping by session"):
        event_id_sequence = session_df['EventId'].tolist()
        # Label is 1 if any log in the session is abnormal
        session_label = 1 if 'abnormal' in session_df['Label'].unique() else 0
        session_data[session_id] = {"sequence": event_id_sequence, "label": session_label}
    return session_data

def _group_by_window(df, window_size, step_size):
    
    """ Helper function for sliding window-based grouping. """
    session_data = {}
    event_ids = df['EventId'].tolist()
    labels = df['label'].tolist()
    
    num_events = len(event_ids)
    for i in tqdm(range(0, num_events - window_size + 1, step_size), desc="Generating sliding windows"):
        window_sequence = event_ids[i : i + window_size]
        window_labels = labels[i : i + window_size]
        
        # Label is 1 if any log in the window is abnormal
        session_label = 1 if any(window_labels) else 0
        session_id = f"window_{i}" # Use the window start index as a unique ID
        session_data[session_id] = {"sequence": window_sequence, "label": session_label}
    return session_data



def create_sequential_vectors(file_path, is_dir, window_size=10, step_size=1, nbr_indexes=True):
    """
    Reads a log file and converts it into sequential vectors for model training.

    Args:
        file_path (str): The path to the merged CSV log file.
        window_size (int): The number of historical log keys (h) to use for
                           predicting the next one.
        step_size (int): The step size for the sliding window.
        is_dir (bool): if it's a dir, then we need to process all files in the directory and apply the same logic to each file.
        number_indexes (bool): If True, the EventIds will be replaced with a unique incremental index for each unique EventId. Then we save the mapping in a pickle file.

    Returns:
        None: The function saves the generated sequences and labels to a new CSV file. With the same name as the input file but with '{window_size}_{step_size}_seq' appended to the name.
    """
    
    if is_dir:
        # If it's a directory, process all files in the directory
        for file in Path(file_path).glob('*_merged.csv'):
            print(f"Processing file: {file}")
            create_seq_vectors_for_file(str(file), window_size, step_size, nbr_indexes=nbr_indexes)
    else:
        # If it's a single file, process that file
        create_seq_vectors_for_file(file_path, window_size, step_size, nbr_indexes=nbr_indexes)

def create_seq_vectors_for_file(file_path, window_size=10, step_size=1, nbr_indexes=True):
    """
    Create sequential vectors from a single log file.
    """
    _name = Path(file_path).name
    print(f"Loading data from {_name}...")
    
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the CSV file is in the same directory as the script or provide the full path.")
        return None, None
    
    

    # Extract the sequence of EventIds, which are the "log keys"
    # The paper notes that these should be treated as categorical values.
    log_key_sequence = df['EventId'].tolist()
    labels = df['label'].tolist()
    
    if nbr_indexes:
        # Create a mapping from EventId to a unique index
        unique_event_ids = sorted(set(log_key_sequence))
        event_id_to_index = {event_id: idx for idx, event_id in enumerate(unique_event_ids)}
        
        # Replace EventIds with their corresponding indices
        log_key_sequence = [event_id_to_index[event_id] for event_id in log_key_sequence]
        
        # Save the mapping to a pickle file
        mapping_file = file_path.replace('_merged.csv', f'_{window_size}_{step_size}_mapping.pkl')
        with open(mapping_file, 'wb') as f:
            pickle.dump(event_id_to_index, f)
        
        print(f"Saved EventId to index mapping to {mapping_file}")
    else:
        # If nbr_indexes is False, we keep the EventIds as they are
        event_id_to_index = {event_id: idx for idx, event_id in enumerate(set(log_key_sequence))}
        print("Using EventIds as they are without indexing.")
    
    print(f"Successfully loaded {len(log_key_sequence)} log events.")

    # Create sequences using a sliding window approach
    X = []
    y = []
    L = []
    
    # Iterate through the sequence to create input/output pairs
    # The loop stops when there are not enough elements left to form a full sequence and a label
    for i in range(0, len(log_key_sequence) - window_size, step_size):
        # The input sequence is a window of 'window_size' log keys
        sequence = log_key_sequence[i:i + window_size]
        
        # The label is the log key immediately following the sequence
        _next = log_key_sequence[i + window_size]
        
        X.append(sequence)
        y.append(_next)
        # in L we will add 0 if there's no 1 in the sequence, otherwise 1
        L.append(1 if 1 in labels[i:i + window_size] else 0)

    print(f"Generated {len(X)} sequences with a window size of {window_size}.")
    
    # --- Save to CSV ---
    print("\n--- Saving to CSV file ---")

    # Create a new DataFrame to hold the sequences and labels
    # The 'sequences' column will contain the list of EventIds for each sample
    train_df = pd.DataFrame({
        'sequences': list(X),
        'next': y,
        'label': L
    })

    # Define the output file name
    output_filename = file_path.replace('_merged', f'_{window_size}_{step_size}_seq')

    # Save the DataFrame to a CSV file, without writing the DataFrame index
    train_df.to_csv(output_filename, index=False)

    print(f"Successfully saved {len(train_df)} rows to {output_filename}")