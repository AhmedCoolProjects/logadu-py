import click
from pathlib import Path
from logadu.vectorization.fasttext import fasttext_tfidf

# TODO: give it a list of dataset names, so we don't waste time loading the vectorizer for each dataset
@click.command()
@click.argument("vectorizer", type=click.Choice(['fasttext', 'bert']))
@click.argument("vectorizer_path", type=click.Path(exists=True))
@click.option("--dataset", type=str, help="Dataset name to vectorize.")
@click.option("--gpath", type=click.Path(exists=True))
def vectorize(vectorizer, vectorizer_path, dataset, gpath):
    """
    Vectorize log templates using the specified word embeddings (e.g., FastText, BERT).
    """

    temp_path = Path(gpath) / f"{dataset}_all_templates.csv"
    if not temp_path.exists():
        click.secho(f"Error: The file {temp_path} does not exist.", fg="red")
        return
    output_dir = Path(gpath) / vectorizer
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_vectors.pt"

    fasttext_tfidf(vectorizer_path=vectorizer_path, temp_path=temp_path, output_path=output_path, col_template_name="EventTemplate", col_eventid_name="EventId")
