# In logadu/sequencing.py

import pandas as pd
from tqdm import tqdm
import pickle
import os
from pathlib import Path

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