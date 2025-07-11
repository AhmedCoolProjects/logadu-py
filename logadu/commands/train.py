import click
from logadu.data.trainer import train_deeplog


@click.command()
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option("--model", default="deeplog", help="Model to train (default: deeplog)")
@click.option("--batch-size", default=32, help="Batch size for training (default: 32)")
@click.option("--epochs", default=100, help="Number of epochs for training (default: 10)")
@click.option("--learning-rate", default=0.001, help="Learning rate for optimizer (default: 0.001)")
@click.option("--hidden-size", default=128, help="Hidden size for LSTM (default: 128)")
@click.option("--num-layers", default=2, help="Number of LSTM layers (default: 2)")
@click.option("--output-dir", default="models", help="Directory to save the trained model (default: models)")
@click.option("--sliding-size", default=9, help="Sliding window size for sequence generation (default: 9)")
def train(dataset_file, model, batch_size, epochs, learning_rate, hidden_size, num_layers, output_dir, sliding_size):
    """Train a log anomaly detection model."""
    if model.lower() == "deeplog":
        
        train_deeplog(
            dataset_file,
            batch_size,
            hidden_size,
            num_layers,
            learning_rate,
            epochs,
            output_dir,
            sliding_size
        )
    else:
        raise click.UsageError(f"Model '{model}' is not supported. Currently only 'deeplog' is available.")

# example with minimal arguments:
# logadu train /home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/dataset/data/spell/sequences/LINUX24_10_1.pkl --model deeplog --output-dir /home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/models