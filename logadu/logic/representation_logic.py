# /logadu/logic/representation_logic.py

import pandas as pd
import torch
from gensim.models import KeyedVectors
import numpy as np
import ast
import click

def _load_word_embeddings(file_path):
    """ Loads pre-trained word embeddings using gensim. """
    click.echo("Loading word embeddings model... (This may take a few minutes)")
    # limit=500000 can be used for faster loading during testing
    word_vectors = KeyedVectors.load_word2vec_format(file_path, binary=False)
    click.echo("Word embeddings model loaded.")
    return word_vectors

def _vectorize_template(template_text, word_vectors):
    """ Converts a single event template string into an aggregated semantic vector. """
    words = template_text.split()
    # Find vectors for words that exist in the embedding model's vocabulary
    vectors = [word_vectors[word] for word in words if word in word_vectors.key_to_index]

    if not vectors:
        # If no words are found in the vocab, return a zero vector
        return np.zeros(word_vectors.vector_size, dtype=np.float32)
    
    # Aggregate by taking the mean of the word vectors
    return np.mean(vectors, axis=0)

def generate_semantic_vectors(template_seq_file, word_embeddings_file):
    """ Main logic to create and save semantic vector sequences. """
    df = pd.read_csv(template_seq_file)
    df['EventSequence'] = df['EventSequence'].apply(ast.literal_eval)

    word_vectors = _load_word_embeddings(word_embeddings_file)

    vectorized_sequences = []
    labels = []
    
    with click.progressbar(df.itertuples(), length=len(df), label="Vectorizing sequences") as bar:
        for row in bar:
            # For each sequence (a list of template strings)
            sequence_of_vectors = [
                _vectorize_template(template, word_vectors) for template in row.EventSequence
            ]
            vectorized_sequences.append(torch.tensor(np.array(sequence_of_vectors), dtype=torch.float32))
            labels.append(torch.tensor(row.Label, dtype=torch.long))

    # --- Save the processed data ---
    # Saving as a dictionary in a single .pt file is efficient
    output_data = {
        'sequences': vectorized_sequences,
        'labels': labels
    }
    
    output_filename = template_seq_file.replace('.csv', '_vectors.pt')
    torch.save(output_data, output_filename)
    
    click.echo(f"Saved {len(vectorized_sequences)} vectorized sequences to: {output_filename}")