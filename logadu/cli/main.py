import click
from logadu.commands.parse import parse
from logadu.commands.merge import merge
from logadu.commands.seq import seq
from logadu.commands.train import train

@click.group()
def cli():
    """LogADU - Advanced Log Analysis and Processing"""
    pass

# Log parsing
cli.add_command(parse)
cli.add_command(merge)
# Feature Extraction
## Sequence vectors
cli.add_command(seq)
## Quantitative vectors
# cli.add_command(quant)
## Semantic vectors
# cli.add_command(semantic)
# Training
cli.add_command(train)
