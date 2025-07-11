import click
from logadu.commands.parse import parse
from logadu.commands.merge import merge
from logadu.commands.seq import seq
from logadu.commands.train import train

@click.group()
def cli():
    """LogADU - Advanced Log Analysis and Processing"""
    pass

cli.add_command(parse)
cli.add_command(merge)
cli.add_command(seq)
cli.add_command(train)
