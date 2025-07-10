import click
from logadu.commands.parse import parse
from logadu.commands.merge import merge

@click.group()
def cli():
    """LogADU - Advanced Log Analysis and Processing"""
    pass

cli.add_command(merge)
cli.add_command(parse)
