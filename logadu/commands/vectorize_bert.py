import click
from pathlib import Path
from logadu.logic.vectorization_bert import vectorization_with_bert


# /home/ahmed.bargady/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/ahmed.bargady/datasets/AITv2/implementation/Fox/drain/Fox_merged.csv

@click.command()
@click.argument("gpath", type=click.Path(exists=True))
@click.argument("dataname", type=str)
def vectorizebert(gpath, dataname):
    """
    Vectorize log templates using the specified word embeddings (e.g., FastText, BERT).
    """
    
    structured_path = Path(gpath) / f"{dataname}_all_structured.csv"
    if not structured_path.exists():
        click.secho(f"Error: The file {structured_path} does not exist.", fg="red")
        return
    output_dir = Path(gpath) / "bert"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataname}_vectors.pt"
    
    log_column = "Content"
    event_id_column = "EventId"
    model_name = "bert-base-uncased"  # You can change this to any other BERT model

    vectorization_with_bert(structured_path, output_file, log_column, event_id_column, model_name, batch_size=128)