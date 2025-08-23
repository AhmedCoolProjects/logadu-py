import click
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import torch

STOP_WORDS = {
    'a', 'an', 'the', 'in', 'on', 'is', 'are', 'am', 'it', 'of', 'for', 'to',
    'and', 'or', 'with', 'as', 'by', 'at', 'from', 'about', 'be', 'was', 'were'
}


def fasttext_tfidf(vectorizer_path, temp_path, output_path, col_template_name, col_eventid_name):

    df = pd.read_csv(temp_path, usecols=[col_eventid_name, col_template_name], low_memory=False)

    if col_eventid_name not in df.columns or col_template_name not in df.columns:
        raise click.UsageError(f"Input CSV must contain '{col_eventid_name}' and '{col_template_name}' columns.")

    df['CleanedTemplate'] = df['EventTemplate'].progress_apply(_tokenize)
    # --- Step 1 - Fit TF-IDF on the cleaned tokens ---
    click.echo("Fitting TF-IDF vectorizer on cleaned tokens...")
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    tfidf.fit(df['CleanedTemplate'])
    idf_weights = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    click.secho(f"TF-IDF vocabulary size: {len(idf_weights)}", fg="green")

    # --- Step 3: Load Word Embeddings ---
    word_vectors = _load_word_embeddings(vectorizer_path)
    template_vector_map = {}

    # --- Step 4: Vectorize each template using the cleaned tokens and weights ---
    for row in tqdm(df.itertuples(), total=len(df), desc="Vectorizing templates"):
        event_id = getattr(row, col_eventid_name)
        template_content = getattr(row, 'CleanedTemplate')
        if not template_content:
            vector = np.zeros(word_vectors.vector_size, dtype=np.float32)
        else:
            valid_words = [word for word in template_content if word in word_vectors.key_to_index]
            if not valid_words:
                vector = np.zeros(word_vectors.vector_size, dtype=np.float32)
            else:
                vectors = [word_vectors[word] for word in valid_words]
                weights = [idf_weights.get(word, 1.0) for word in valid_words]
                vector = np.average(vectors, axis=0, weights=weights)
                
        template_vector_map[event_id] = torch.tensor(vector, dtype=torch.float32)

    torch.save(template_vector_map, output_path)
    click.secho(f"\nSuccessfully created and saved semantic vector map to: {output_path}", fg="green")


def _load_word_embeddings(vectorizer_path):
    """ Loads the pre-trained FastText word embeddings model from a .vec file. """
    click.secho("Loading word embedding FastText may take significant RAM...", fg="yellow")
    word_vectors = KeyedVectors.load_word2vec_format(vectorizer_path, binary=False)
    click.secho("Word embeddings model loaded successfully.", fg="green")
    return word_vectors

def _tokenize(template_content: str) -> list[str]:

    if not isinstance(template_content, str):
        return []
    
    # 1. Split camelCase, e.g., "BlockManager" -> "Block Manager"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', template_content)
    # 2. Remove all non-alphabetic characters (keeps spaces for splitting)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 3. Convert to lowercase and split into words
    words = text.lower().split()
    # 4. Remove stop words and single-character tokens
    cleaned_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    return cleaned_words