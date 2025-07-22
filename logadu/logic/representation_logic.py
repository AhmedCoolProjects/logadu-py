# /logadu/logic/representation_logic.py

import pandas as pd
import torch
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
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
    
    

def generate_neurallog_vectors(raw_seq_file):
    """
    Generates semantic vectors for raw log message sequences using a pre-trained BERT model.
    """
    df = pd.read_csv(raw_seq_file)
    df['EventSequence'] = df['EventSequence'].apply(ast.literal_eval)
    
    click.echo("Loading pre-trained BERT model and tokenizer ('bert-base-cased')...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    model.eval() # Set model to evaluation mode
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    click.echo(f"Using device: {device}")

    vectorized_sequences = []
    labels = []

    with click.progressbar(df.itertuples(), length=len(df), label="Vectorizing raw sequences") as bar:
        for row in bar:
            raw_text_sequence = row.EventSequence
            
            # Tokenize and get BERT embeddings for the batch of messages in the sequence
            # We get the embedding of the [CLS] token as the representation for each message
            inputs = tokenizer(raw_text_sequence, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # Use the hidden state of the [CLS] token for each message
                sequence_of_vectors = outputs.last_hidden_state[:, 0, :].cpu()
            
            vectorized_sequences.append(sequence_of_vectors)
            labels.append(torch.tensor(row.Label, dtype=torch.long))

    # --- Save the processed data ---
    output_data = {'sequences': vectorized_sequences, 'labels': labels}
    output_filename = raw_seq_file.replace('.csv', '_vectors_neurallog.pt')
    torch.save(output_data, output_filename)
    click.echo(f"Saved {len(vectorized_sequences)} NeuralLog vectors to: {output_filename}")