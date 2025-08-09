import click
import pandas as pd
import torch
import numpy as np
from gensim.models import KeyedVectors

def _load_word_embeddings(file_path):
    """
    Loads the pre-trained FastText word embeddings model from a .vec file.
    This is a time and memory-intensive operation.
    """
    click.echo(f"Loading word embeddings model from: {file_path}")
    click.secho("This may take several minutes and consume significant RAM...", fg="yellow")
    
    # Use gensim's KeyedVectors to load the model. binary=False is for .vec text format.
    word_vectors = KeyedVectors.load_word2vec_format(file_path, binary=False)
    
    click.secho("Word embeddings model loaded successfully.", fg="green")
    return word_vectors

def _vectorize_template(template_text, word_vectors):
    """
    Converts a single log template string into one aggregated semantic vector.
    
    Args:
        template_text (str): The log template, e.g., "User * logged out".
        word_vectors (KeyedVectors): The loaded FastText model.

    Returns:
        np.ndarray: A single 300-dimensional vector representing the template.
    """
    # 1. Tokenize the template into words
    words = template_text.split()
    
    # 2. Look up the vector for each word, ignoring words not in the vocabulary
    vectors = [word_vectors[word] for word in words if word in word_vectors.key_to_index]

    # 3. Handle cases where no words were found in the vocabulary
    if not vectors:
        # Return a zero vector of the correct dimension if the template is empty or all words are unknown
        return np.zeros(word_vectors.vector_size, dtype=np.float32)
    
    # 4. Aggregate the word vectors by taking their mean to get a single template vector
    return np.mean(vectors, axis=0)

def vectorize_templates_from_file(vectorizer_path, output_file, temp_path):
    """
    Loads a CSV of unique templates, vectorizes them, and saves a lookup map.
    """
    df = pd.read_csv(temp_path)
    
    if 'EventId' not in df.columns or 'EventTemplate' not in df.columns:
        raise click.UsageError("Input CSV must contain 'EventId' and 'EventTemplate' columns.")
        
    word_vectors = _load_word_embeddings(vectorizer_path)

    template_vector_map = {}

    with click.progressbar(df.itertuples(), length=len(df), label="Vectorizing templates") as bar:
        for row in bar:
            event_id = row.EventId
            template_text = row.EventTemplate
            
            vector = _vectorize_template(template_text, word_vectors)
            
            template_vector_map[event_id] = torch.tensor(vector, dtype=torch.float32)

    torch.save(template_vector_map, output_file)
