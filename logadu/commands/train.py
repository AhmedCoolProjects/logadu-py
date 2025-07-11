import click
from logadu.data.trainer import train_deeplog


@click.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--model", default="deeplog", help="Model to train (default: deeplog)")
@click.option("--batch-size", default=32, help="Batch size for training (default: 32)")
@click.option("--epochs", default=10, help="Number of epochs for training (default: 10)")
@click.option("--learning-rate", default=0.001, help="Learning rate for optimizer (default: 0.001)")
@click.option("--hidden-size", default=128, help="Hidden size for LSTM (default: 128)")
@click.option("--num-layers", default=2, help="Number of LSTM layers (default: 2)")
@click.option("--output-dir", default="models", help="Directory to save the trained model (default: models)")
def train(dataset, model, batch_size, epochs, learning_rate, hidden_size, num_layers, output_dir):
    """Train a log anomaly detection model."""
    if model.lower() == "deeplog":
        args = click.get_current_context().obj
        args.dataset_file = dataset
        args.batch_size = batch_size
        args.epochs = epochs
        args.learning_rate = learning_rate
        args.hidden_size = hidden_size
        args.num_layers = num_layers
        args.output_dir = output_dir
        
        train_deeplog(args)
    else:
        raise click.UsageError(f"Model '{model}' is not supported. Currently only 'deeplog' is available.")

# example with minimal arguments:
# logadu train /home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/dataset/data/spell/sequences/LINUX24_10_1.pkl --model deeplog --output-dir /home/ahmed.bargady/lustre/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/data/github/logs-ad-ultimate/logadu-package/models