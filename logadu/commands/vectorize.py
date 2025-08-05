import click
from pathlib import Path
from logadu.logic.vectorization_logic import vectorize_templates_from_file


@click.command()
@click.argument("template_file", type=click.Path(exists=True))
@click.argument("word_embeddings_file", type=click.Path(exists=True))
@click.option("--output-dir", default="fasttext", help="The folder name we save the vectorized templates to.")
@click.option("--vectorizer", default="fasttext", type=click.Choice(['fasttext', 'bert']), help="Type of vectorizer to use.")
def vectorize(template_file, word_embeddings_file, output_dir, vectorizer):
    """
    Vectorize log templates using the specified word embeddings.
    """
    output_path = Path(template_file).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{Path(template_file).stem}_vectors.pt"

    vectorize_templates_from_file(template_file, word_embeddings_file, output_file, vectorizer)
