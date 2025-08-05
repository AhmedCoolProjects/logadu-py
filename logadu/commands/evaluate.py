import click
from logadu.ml.knn import run_knn

@click.command()
@click.argument("vector_file", type=click.Path(exists=True))
@click.option("--model", default="knn", type=click.Choice(['knn']), help="Type of model to evaluate.")
@click.option("--k-neighbors", "-k", default=5, help="Number of neighbors (k) for the KNN model.")
@click.option("--n-splits", default=5, help="Number of folds for Time Series Cross-Validation.")
def evaluate(vector_file, model, k_neighbors, n_splits):
    """
    Evaluate the specified model on the given vector file.
    """
    if model == "knn":
        run_knn(vector_file, k_neighbors, n_splits)