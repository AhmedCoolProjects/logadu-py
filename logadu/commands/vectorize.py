import click
from pathlib import Path
from logadu.logic.vectorization_logic import vectorize_templates_from_file


@click.command()
@click.argument("vectorizer", type=click.Choice(['fasttext', 'bert']))
@click.argument("vectorizer_path", type=click.Path(exists=True))
@click.option("--dataset", type=str, help="Dataset name to vectorize.")
@click.option("--gpath", type=click.Path(exists=True))
def vectorize(vectorizer, vectorizer_path, dataset, gpath):
    """
    Vectorize log templates using the specified word embeddings (e.g., FastText, BERT).
    """
    # temp_path = gpath + dataset + drain + dataset + "_templates.csv"
    temp_path = Path(gpath) / dataset / "drain" / f"{dataset}_templates.csv"
    if not temp_path.exists():
        click.secho(f"Error: The file {temp_path} does not exist.", fg="red")
        return
    output_dir = Path(temp_path).parent / vectorizer
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{Path(temp_path).stem}_vectors.pt"

    vectorize_templates_from_file(vectorizer_path, output_file, temp_path)
