import click
from pathlib import Path
from logadu.logic.vectorization_logic import vectorize_templates_from_file


# /home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod/drain/santos_all_templates.csv
# /home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/crawl-300d-2M.vec
# logadu vectorize fasttext /home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/crawl-300d-2M.vec --dataset santos --gpath /home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/temp/prod/drain

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
    output_file = output_dir / f"{dataset}_vectors.pt"

    vectorize_templates_from_file(vectorizer_path, output_file, temp_path)
