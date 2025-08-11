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
    base_path = Path(gpath) / dataname / "drain"
    merged_csv = base_path / f"{dataname}_merged.csv"
    output_path = base_path / f"{dataname}_bert_vectors.pt"
    
    log_column = "content"
    event_id_column = "EventId"
    model_name = "bert-base-uncased"  # You can change this to any other BERT model

    vectorization_with_bert(merged_csv, output_path, log_column, event_id_column, model_name, batch_size=128)