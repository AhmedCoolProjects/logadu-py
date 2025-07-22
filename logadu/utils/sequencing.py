# /logadu/utils/sequencing.py

import pandas as pd
import pickle
import os
import click
from pathlib import Path

# --- MODIFIED FUNCTION SIGNATURE ---
def create_sequential_vectors(file_path, is_dir, window_size=10, step_size=1, output_format='index'):
    """
    Reads a log file and converts it into sequential vectors for model training.
    """
    
    if is_dir:
        for file in Path(file_path).glob('*_merged.csv'):
            print(f"Processing file: {file}")
            create_seq_vectors_for_file(str(file), window_size, step_size, output_format)
    else:
        create_seq_vectors_for_file(file_path, window_size, step_size, output_format)

# --- MODIFIED FUNCTION SIGNATURE AND LOGIC ---
def create_seq_vectors_for_file(file_path, window_size=10, step_size=1, output_format='index'):
    """
    Create sequential vectors from a single log file, either as integer indexes or text templates.
    """
    _name = Path(file_path).name
    print(f"Loading data from {_name}...")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    source_sequence = []
    
    # --- CONDITIONAL LOGIC BASED ON OUTPUT FORMAT ---
    if output_format == 'index':
        click.echo("Generating sequences of integer indexes from 'EventId' column.")
        # Extract the sequence of EventIds
        log_key_sequence = df['EventId'].tolist()
        
        # Create and save a mapping from EventId to a unique index
        unique_event_ids = sorted(list(df['EventId'].unique()))
        event_id_to_index = {event_id: idx for idx, event_id in enumerate(unique_event_ids)}
        
        # Replace EventIds with their corresponding indices
        source_sequence = [event_id_to_index[event_id] for event_id in log_key_sequence]
        
        # Save the mapping to a pickle file
        mapping_file = file_path.replace('_merged.csv', f'_{window_size}_{step_size}_mapping.pkl')
        with open(mapping_file, 'wb') as f:
            pickle.dump(event_id_to_index, f)
        print(f"Saved EventId to index mapping to {mapping_file}")

    elif output_format == 'template':
        click.echo("Generating sequences of text templates from 'EventTemplate' column.")
        if 'EventTemplate' not in df.columns:
            click.secho("Error: 'EventTemplate' column not found in the input file.", fg="red")
            return
        # Use the EventTemplate strings directly as the source sequence
        source_sequence = df['EventTemplate'].tolist()

    labels = df['label'].tolist()
    print(f"Successfully loaded {len(source_sequence)} log events.")

    # --- SLIDING WINDOW LOGIC (REMAINS THE SAME) ---
    X = []
    y = []
    L = []
    
    for i in range(0, len(source_sequence) - window_size, step_size):
        sequence = source_sequence[i:i + window_size]
        _next = source_sequence[i + window_size]
        
        X.append(sequence)
        y.append(_next)
        L.append(1 if 1 in labels[i:i + window_size] else 0)

    print(f"Generated {len(X)} sequences with a window size of {window_size}.")
    
    # --- Save to CSV ---
    print("\n--- Saving to CSV file ---")

    # The 'sequences' column can hold either lists of ints or lists of strings
    # For LogRobust, we use the EventSequence and Label column names
    if output_format == 'template':
        output_df = pd.DataFrame({'EventSequence': X, 'Label': L})
    else:
        output_df = pd.DataFrame({'sequences': X, 'next': y, 'label': L})

    # Define a clearer output file name based on the format
    output_suffix = f'_{window_size}_{step_size}_seq_{output_format}.csv'
    output_filename = file_path.replace('_merged.csv', output_suffix)

    output_df.to_csv(output_filename, index=False)
    print(f"Successfully saved {len(output_df)} rows to {output_filename}")