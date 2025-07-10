import click
from logadu.commands.parse import parse

@click.group()
def cli():
    """LogADU - Advanced Log Analysis and Processing"""
    pass

cli.add_command(parse)