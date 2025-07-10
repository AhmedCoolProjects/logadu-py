import click
from logadu.parsers.merging import merge_logs
from pathlib import Path

@click.command()
@click.argument("in_dir", type=click.Path(exists=True))
@click.argument("file_name", type=str)
@click.option("--parser", default="drain", help="Parser used for structured log files (drain, spell, ft-tree)")
def merge(in_dir, file_name, parser):
    """Merge structured log files with original labeled log files."""
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise click.UsageError(f"{in_dir} is not a valid directory.")

    if not file_name:
        raise click.UsageError("file_name must be provided.")

    merge_logs(in_dir=str(in_dir), file_name=file_name, parser=parser)
    click.echo(f"Successfully merged {file_name} logs using {parser} parser.")


# Example usage in command line:
# python -m logadu.commands.merge /path/to/logs file_name --parser drain