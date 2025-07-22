# /logadu/commands/represent.py

import click
from pathlib import Path
from logadu.logic.representation_logic import generate_semantic_vectors, generate_neurallog_vectors

@click.command()
@click.argument("input_file")
@click.option("--model", required=True, type=click.Choice(['logrobust', 'neurallog']), help="The target model for representation.")
@click.option("--word-embeddings-file", type=click.Path(exists=True),
              help="[LogRobust] Path to the pre-trained word embeddings file.")
def represent(input_file, model, word_embeddings_file):
    click.secho(f"Starting semantic representation for: {Path(input_file).name}", fg="yellow")

    if model.lower() == 'logrobust':
        if not word_embeddings_file:
            raise click.UsageError("--word-embeddings-file is required for LogRobust.")
        generate_semantic_vectors(input_file, word_embeddings_file)
    elif model.lower() == 'neurallog':
        generate_neurallog_vectors(input_file)

    click.secho("Semantic vector file created successfully.", fg="green")