import click
import pandas as pd
import torch
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import re


# You can expand this list with domain-specific words if needed
STOP_WORDS = {
    'a', 'an', 'the', 'in', 'on', 'is', 'are', 'am', 'it', 'of', 'for', 'to',
    'and', 'or', 'with', 'as', 'by', 'at', 'from', 'about', 'be', 'was', 'were'
}

def _load_word_embeddings(file_path):
    """ Loads the pre-trained FastText word embeddings model from a .vec file. """
    click.echo(f"Loading word embeddings model from: {file_path}")
    click.secho("This may take several minutes and consume significant RAM...", fg="yellow")
    word_vectors = KeyedVectors.load_word2vec_format(file_path, binary=False)
    click.secho("Word embeddings model loaded successfully.", fg="green")
    return word_vectors

def _clean_and_tokenize_template(template_text: str) -> list[str]:
    """
    NEW FUNCTION: Cleans and tokenizes a single log template string.
    - Splits camelCase and PascalCase words.
    - Removes non-alphabetic characters.
    - Converts to lowercase.
    - Removes stop words.
    """
    if not isinstance(template_text, str):
        return []
    
    # 1. Split camelCase, e.g., "BlockManager" -> "Block Manager"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', template_text)
    
    # 2. Remove all non-alphabetic characters (keeps spaces for splitting)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 3. Convert to lowercase and split into words
    words = text.lower().split()
    
    # 4. Remove stop words and single-character tokens
    cleaned_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    
    return cleaned_words

def _vectorize_template_with_tfidf(cleaned_words: list[str], word_vectors: KeyedVectors, idf_weights: dict) -> np.ndarray:
    """
    Converts a list of cleaned words into one TF-IDF weighted semantic vector.
    
    Args:
        cleaned_words (list[str]): The pre-cleaned and tokenized words of the template.
        word_vectors (KeyedVectors): The loaded FastText model.
        idf_weights (dict): A dictionary mapping words to their learned IDF scores.

    Returns:
        np.ndarray: A single vector representing the template.
    """
    if not cleaned_words:
        return np.zeros(word_vectors.vector_size, dtype=np.float32)

    # Filter for words that exist in the word embedding vocabulary
    valid_words = [word for word in cleaned_words if word in word_vectors.key_to_index]
    
    if not valid_words:
        return np.zeros(word_vectors.vector_size, dtype=np.float32)

    vectors = [word_vectors[word] for word in valid_words]
    weights = [idf_weights.get(word, 1.0) for word in valid_words]

    return np.average(vectors, axis=0, weights=weights)

def vectorize_templates_from_file(vectorizer_path, output_file, temp_path):
    """
    Loads a CSV of unique templates, cleans them, vectorizes them using TF-IDF weighted
    FastText embeddings, and saves a lookup map.
    """
    df = pd.read_csv(temp_path, usecols=['EventId', 'EventTemplate'])

    if 'EventId' not in df.columns or 'EventTemplate' not in df.columns:
        raise click.UsageError("Input CSV must contain 'EventId' and 'EventTemplate' columns.")

    # --- NEW: Step 1 - Clean all templates first ---
    click.echo("Cleaning and tokenizing all templates...")
    tqdm.pandas(desc="Cleaning Templates")
    df['CleanedTokens'] = df['EventTemplate'].progress_apply(_clean_and_tokenize_template)

    # --- Step 2 - Fit TF-IDF on the cleaned tokens ---
    click.echo("Fitting TF-IDF vectorizer on cleaned tokens...")
    # The tokenizer is now an identity function because the input is already tokenized
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    
    tfidf.fit(df['CleanedTokens'])
    
    idf_weights = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    click.secho(f"TF-IDF vocabulary size: {len(idf_weights)}", fg="green")

    # --- Step 3: Load Word Embeddings ---
    word_vectors = _load_word_embeddings(vectorizer_path)

    template_vector_map = {}

    # --- Step 4: Vectorize each template using the cleaned tokens and weights ---
    for row in tqdm(df.itertuples(), total=len(df), desc="Vectorizing templates"):
        event_id = row.EventId
        cleaned_tokens = row.CleanedTokens
        
        vector = _vectorize_template_with_tfidf(cleaned_tokens, word_vectors, idf_weights)
        
        template_vector_map[event_id] = torch.tensor(vector, dtype=torch.float32)

    torch.save(template_vector_map, output_file)
    click.secho(f"\nSuccessfully created and saved semantic vector map to: {output_file}", fg="green")
    
    