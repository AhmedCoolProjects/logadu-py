import click
from pathlib import Path
from logadu.logic.conversion_logic import convert_indexes_to_vectors

@click.command()
@click.argument("index_seq_file", type=click.Path(exists=True))
@click.option("--mapping-file", required=True, type=click.Path(exists=True),
help="Path to the .pkl file mapping EventIDs to indexes.")
@click.option("--template-vectors-file", required=True, type=click.Path(exists=True),
help="Path to the .pt file mapping EventIDs to semantic vectors (from 'vectorize-templates').")
def convert(index_seq_file, mapping_file, template_vectors_file):
    """
    Convert an index-based sequence file into a semantic vector sequence file. This command is a bridge that allows reusing index-based data for semantic models.
    It uses the mapping files to look up the corresponding semantic vector for each index.
    """
    click.secho(f"Starting conversion of: {Path(index_seq_file).name}", fg="yellow")

    convert_indexes_to_vectors(index_seq_file, mapping_file, template_vectors_file)

    click.secho("Conversion to semantic vector sequences complete.", fg="green")