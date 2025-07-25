# In logadu/merging.py

import pandas as pd
import os

def merge_logs(in_dir, file_name, parser):
    """
    Merges the structured output from the parser with the original labeled log file.

    This function assumes that the rows in both files correspond to each other 
    (i.e., row N in structured_file corresponds to row N in original_file).

    Args:
        in_dir (str): the directory containing the structured and original log files.
        file_name (str): the name of the file with no extension.
        parser (str): the name of used parser (drain, spell, tf_tree)
    """
    print("Starting the merging process...")
    
    org_file_path = os.path.join(in_dir, f"{file_name}.csv")
    structured_file_path = os.path.join(in_dir, parser, f"{file_name}_structured.csv")

    try:
        df_structured = pd.read_csv(structured_file_path)
        df_original = pd.read_csv(org_file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check file paths.")
        return

    # --- Sanity Check ---
    if len(df_structured) != len(df_original):
        print("Warning: The number of rows in the structured file and the original file do not match.")
        print(f"Structured file has {len(df_structured)} rows.")
        print(f"Original file has {len(df_original)} rows.")
        print("Proceeding, but this may lead to incorrect merges.")

    
        
    # Create the merged DataFrame by adding the new columns from the structured file to the original file
    merged_df = df_original.copy()
    if 'EventId' in df_structured.columns:
        merged_df['EventId'] = df_structured['EventId']
    else:
        print("Warning: 'EventId' column not found in structured file. Merging without it.")
    if 'EventTemplate' in df_structured.columns:
        merged_df['EventTemplate'] = df_structured['EventTemplate']
    else:
        print("Warning: 'EventTemplate' column not found in structured file. Merging without it.")
    if 'LineId' in df_structured.columns:
        merged_df['LineId'] = df_structured['LineId']
    else:
        print("Warning: 'LineId' column not found in structured file. Merging without it.")

    # Save the merged file
    output_path = os.path.join(in_dir, parser, f"{file_name}_merged.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"Successfully merged files. Output saved to: {output_path}")
    print(f"Merged file contains columns: {merged_df.columns.tolist()}")