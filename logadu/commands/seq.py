from logadu.utils.sequencing import generate_sequences, create_sequential_vectors
import click
from pathlib import Path

@click.command()
@click.argument("log_file", type=click.Path(exists=True))
# @click.option("--method", type=click.Choice(['session', 'window']), required=True,
            #   help="Method to generate sequences: 'session' for session-based or 'window' for sliding window-based grouping.")
# @click.option("--session_col", type=str, help="Column name for session grouping (required if method is 'session').")
@click.option("--window_size", type=int, default=10,
              help="Size of the sliding window (required if method is 'window').")
@click.option("--step_size", type=int, default=1,
              help="Step size for the sliding window (required if method is 'window').")
@click.option("--nbr_indexes", is_flag=True, default=True,
              help="If set, replaces EventIds with unique incremental indexes and saves the mapping in a pickle file.")
def seq(log_file, window_size, step_size, nbr_indexes):
    """Generate sequences from structured log data."""
    
    _name = Path(log_file).name
    
    is_dir = Path(log_file).is_dir()
    if is_dir:
        click.echo(f"Processing all files in directory: {_name}")
    else:
        click.echo(f"Processing file: {_name}")
        

    create_sequential_vectors(log_file, is_dir, window_size, step_size, nbr_indexes)
    click.echo(f"Sequences generated and saved")

# Example usage in command line:
# logadu seq /path/to/structured_log.csv --method window --window_size 10 --step_size 1