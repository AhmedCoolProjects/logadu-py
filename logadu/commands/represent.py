# /logadu/commands/represent.py

import click
from pathlib import Path
from logadu.logic.representation_logic import generate_semantic_vectors

@click.command()
@click.argument("template_seq_file", type=click.Path(exists=True))
@click.option("--word-embeddings-file", required=True, type=click.Path(exists=True),
              help="Path to the pre-trained word embeddings file (e.g., crawl-300d-2M.vec).")
def represent(template_seq_file, word_embeddings_file):
    """
    Generate and save semantic vector representations from text sequences.
    
    Takes a CSV file with 'EventSequence' and 'Label' columns and converts it
    into a PyTorch file containing tensors ready for training LogRobust.
    """
    click.secho(f"Starting semantic representation for: {Path(template_seq_file).name}", fg="yellow")
    
    generate_semantic_vectors(template_seq_file, word_embeddings_file)
    
    click.secho("Semantic vector file created successfully.", fg="green")