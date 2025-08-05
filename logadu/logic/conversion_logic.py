import click
import pandas as pd
import torch
import pickle
import ast

def convert_indexes_to_vectors(index_seq_file, mapping_file, template_vectors_file):
    """
    Main logic to convert sequences of indexes to sequences of vectors.
    """
    # --- Step 1: Load all required files ---
    click.echo("Loading input files...")
    df_sequences = pd.read_csv(index_seq_file)
    df_sequences['sequence'] = df_sequences['sequence'].apply(ast.literal_eval)

    with open(mapping_file, 'rb') as f:
        eventid_to_index_map = pickle.load(f)

    template_vector_map = torch.load(template_vectors_file)
    click.echo("All files loaded.")

    # --- Step 2: Invert the mapping to get Index -> EventID ---
    # The .pkl file maps EventID -> Index. We need the reverse for our lookup.
    click.echo("Inverting EventID-to-Index map...")
    index_to_eventid_map = {index: eventid for eventid, index in eventid_to_index_map.items()}

    # --- Step 3: Iterate and convert sequences ---
    vectorized_sequences = []
    labels = []
    with click.progressbar(df_sequences.itertuples(), length=len(df_sequences), label="Converting sequences") as bar:
        for row in bar:
            index_sequence = row.sequence
            
            # This list will hold the vectors for the current sequence
            sequence_of_vectors = []
            
            for index in index_sequence:
                # The two-step lookup process:
                # 1. Look up the EventID from the index
                event_id = index_to_eventid_map.get(index)
                
                if event_id:
                    # 2. Look up the vector from the EventID
                    vector = template_vector_map.get(event_id)
                    if vector is not None:
                        sequence_of_vectors.append(vector)
                
                # If an index or event_id is not found, it is skipped.
                # This can happen if the mapping files are from a different dataset version.
            
            # We only add sequences that are not empty after conversion
            if sequence_of_vectors:
                vectorized_sequences.append(torch.stack(sequence_of_vectors))
                labels.append(torch.tensor(row.label, dtype=torch.long))

    # --- Step 4: Save the new vectorized sequence file ---
    output_data = {
        'sequences': vectorized_sequences,
        'labels': labels
    }
    
    # Create a descriptive output filename
    output_filename = index_seq_file.replace('_seq_index.csv', '_vectors_converted.pt')
    torch.save(output_data, output_filename)
    
    click.echo(f"\nSuccessfully saved {len(vectorized_sequences)} converted vector sequences to: {output_filename}")