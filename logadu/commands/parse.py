import click
from logadu.parsers.ft_tree import parse_logs
from logadu.parsers.spell import SpellParser
from logadu.parsers.drain import DrainParser
from pathlib import Path
import os

@click.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--depth", default=4, help="Depth parameter for Drain parser")
@click.option("--max-children", default=100, help="Max children for Drain parser")
@click.option("--parser", default="ft-tree", help="Parser to use")
@click.option("--leaf-num", default=4, help="Maximum children before pruning")
@click.option("--short-threshold", default=5, help="Minimum log length")
@click.option("--log-format", help="Log format for Spell parser", default="<Content>")
@click.option("--tau", help="Threshold for Spell parser", default=0.5, type=float)
@click.option("--keep-parameters/--no-parameters", default=True, help="Include extracted parameters in output")
def parse(log_file, parser, leaf_num, short_threshold, log_format, tau, keep_parameters, depth, max_children):
    """Parse log files using specified parser"""
    if parser == "ft-tree":
        log_file_name = Path(log_file).name
        output_dir = Path(log_file).parent / 'ft_tree'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Remove file extension for output names
        parse_logs(
            input_path=log_file,
            output_template=output_dir / (log_file_name.split('.')[0] + '.template'),
            output_freq=output_dir / (log_file_name.split('.')[0] + '.fre'),
            leaf_num=leaf_num,
            short_threshold=short_threshold
        )
        click.echo(f"Successfully parsed logs using FT-Tree. Templates saved to ")
    elif parser == "spell":
        if log_format is None:
            raise click.UsageError("--log-format is required for Spell parser")
        spell = SpellParser(tau=tau, keep_para=keep_parameters)
        output_dir = Path(log_file).parent / 'spell'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        spell.parse_file(
            input_path=str(log_file),
            output_dir=output_dir,
            log_format=log_format
        )
        click.echo(f"Successfully parsed logs using Spell parser. Output saved to {Path(log_file).parent}/spell")
    elif parser == "drain":
        output_dir = Path(log_file).parent / 'drain'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        drain = DrainParser(
            depth=depth,
            sim_threshold=tau,
            max_children=max_children,
            keep_para=keep_parameters
        )
        drain.parse_file(
            input_path=str(log_file),
            output_dir=str(output_dir),
            log_format=log_format
        )
    else:
        click.echo(f"Parser {parser} not recognized", err=True)


# EXAMPLE:
# ogadu parse ../../x.log --parser drain --no-parameters