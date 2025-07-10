# In logadu/sequencing.py

import pandas as pd
from tqdm import tqdm
import pickle
import os

def generate_sequences(structured_log_file, output_dir, method, session_col=None, window_size=None, step_size=None):
    """
    Generates sequences from structured log data using either session-based or window-based grouping.

    Args:
        structured_log_file (str): Path to the structured log CSV file.
        output_dir (str): The directory to save the output files.
        method (str): Grouping method, either 'session' or 'window'.
        session_col (str, optional): The column name for session grouping. Required if method is 'session'.
        window_size (int, optional): The size of the sliding window. Required if method is 'window'.
        step_size (int, optional): The step size for the sliding window. Required if method is 'window'.
    """
    print(f"Generating sequences using '{method}' method...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(structured_log_file)
    
    # Ensure 'Label' column exists and fill NaNs if necessary
    if 'Label' not in df.columns:
        df['Label'] = 'normal' # Assume normal if no labels are present
    else:
        df['Label'] = df['Label'].fillna('normal')

    session_data = {}

    if method == 'session':
        if not session_col:
            raise ValueError("session_col must be provided for 'session' method.")
        session_data = _group_by_session(df, session_col)
    
    elif method == 'window':
        if not window_size or not step_size:
            raise ValueError("window_size and step_size must be provided for 'window' method.")
        session_data = _group_by_window(df, window_size, step_size)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'session' or 'window'.")

    print(f"Generated {len(session_data)} sequences.")
    
    sequences_file = os.path.join(output_dir, 'log_sequences.pkl')
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
    labels = (df['Label'] == 'abnormal').astype(int).tolist()
    
    num_events = len(event_ids)
    for i in tqdm(range(0, num_events - window_size + 1, step_size), desc="Generating sliding windows"):
        window_sequence = event_ids[i : i + window_size]
        window_labels = labels[i : i + window_size]
        
        # Label is 1 if any log in the window is abnormal
        session_label = 1 if any(window_labels) else 0
        session_id = f"window_{i}" # Use the window start index as a unique ID
        session_data[session_id] = {"sequence": window_sequence, "label": session_label}
    return session_data