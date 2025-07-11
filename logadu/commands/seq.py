from logadu.utils.sequencing import generate_sequences 
import click
from pathlib import Path

@click.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--method", type=click.Choice(['session', 'window']), required=True,
              help="Method to generate sequences: 'session' for session-based or 'window' for sliding window-based grouping.")
@click.option("--session_col", type=str, help="Column name for session grouping (required if method is 'session').")
@click.option("--window_size", type=int, default=10,
              help="Size of the sliding window (required if method is 'window').")
@click.option("--step_size", type=int, default=1,
              help="Step size for the sliding window (required if method is 'window').")
def seq(log_file, method, session_col, window_size, step_size):
    """Generate sequences from structured log data."""
    if method == 'session' and not session_col:
        raise click.UsageError("session_col must be provided for 'session' method.")
    
    if method == 'window' and (window_size is None or step_size is None):
        raise click.UsageError("window_size and step_size must be provided for 'window' method.")
    
    generate_sequences(log_file, method, session_col, window_size, step_size)
    click.echo(f"Sequences generated using '{method}' method and saved to {Path(log_file).parent / 'sequences'}")
    
# Example usage in command line:
# logadu seq /path/to/structured_log.csv --method window --window_size 10 --step_size 1