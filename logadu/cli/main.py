import click
from logadu.commands.parse import parse
from logadu.commands.merge import merge
from logadu.commands.seq import seq
from logadu.commands.train import train
from logadu.commands.represent import represent
from logadu.commands.predict import predict
from logadu.commands.evaluate import evaluate

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
cli.add_command(represent) # ADD THIS
# Training
cli.add_command(train)
# Prediction
cli.add_command(predict)
# Evaluation
cli.add_command(evaluate)
