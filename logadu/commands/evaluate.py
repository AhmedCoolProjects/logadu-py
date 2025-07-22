# /logadu/commands/evaluate.py

import click
from logadu.logic.traditional_logic import evaluate_pca, evaluate_rf

@click.command()
@click.argument("vector_file", type=click.Path(exists=True))
@click.option("--model", required=True, type=click.Choice(['pca', 'rf']), help="Traditional model to evaluate.")
@click.option("--output-dir", default="models", help="Directory to save the trained model.")
def evaluate(vector_file, model, output_dir):
    """
    Evaluate traditional ML models (PCA, RandomForest) on pre-computed vectors.
    """
    if not vector_file.endswith('_vectors.pt'):
        raise click.UsageError("Input for this command must be a pre-computed vector file ending in '_vectors.pt'.")

    if model.lower() == 'pca':
        evaluate_pca(vector_file, output_dir)
    elif model.lower() == 'rf':
        evaluate_rf(vector_file, output_dir)
    else:
        click.echo(f"Model '{model}' not yet implemented.")