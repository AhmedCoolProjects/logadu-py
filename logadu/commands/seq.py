# /logadu/commands/seq.py

from logadu.utils.sequencing import create_sequential_vectors
import click
from pathlib import Path

@click.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--window_size", type=int, default=10, help="Size of the sliding window.")
@click.option("--step_size", type=int, default=1, help="Step size for the sliding window.")
# --- MODIFIED OPTION ---
# Replaced --nbr_indexes with a more explicit --output-format option
@click.option("--output-format", 
              type=click.Choice(['index', 'template', 'raw']), 
              default='index', 
              show_default=True,
              help="Output format: 'index', 'template' text, or 'raw' log messages.")
def seq(log_file, window_size, step_size, output_format):
    """Generate sequences from structured log data."""
    
    _name = Path(log_file).name
    
    is_dir = Path(log_file).is_dir()
    if is_dir:
        click.echo(f"Processing all files in directory: {_name}")
    else:
        click.echo(f"Processing file: {_name}")
        
    # Pass the new option to the logic function
    create_sequential_vectors(log_file, is_dir, window_size, step_size, output_format)
    click.echo(f"Sequences generated and saved")