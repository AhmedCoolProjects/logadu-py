import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import click

def vectorization_with_bert(input_merged_csv, output_path, log_column, event_id_column, model_name, batch_size):
    """
    Vectorizes unique log templates from a CSV file using a pre-trained BERT model
    and saves the result as a torch tensor file mapping EventId to vector.
    """
    # --- 1. Setup Device (GPU/CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.secho(f"Using device: {device}", fg="blue")

    # --- 2. Load Data and Identify Unique Templates ---
    click.echo(f"Loading logs from {input_merged_csv}...")
    df = pd.read_csv(input_merged_csv)

    # We only need to vectorize each unique log template once.
    # We find the first occurrence of each EventId to get its corresponding raw content.
    df_unique_templates = df.drop_duplicates(subset=[event_id_column])
    event_ids = df_unique_templates[event_id_column].tolist()
    logs_to_vectorize = df_unique_templates[log_column].tolist()
    
    click.secho(f"Found {len(logs_to_vectorize)} unique log templates to vectorize.", fg="green")

    # --- 3. Load Pre-trained BERT Model and Tokenizer ---
    click.echo(f"Loading model '{model_name}' from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    vector_map = {}
    
    # --- 4. Process Logs in Batches ---
    click.echo(f"Starting vectorization with batch size {batch_size}...")
    
    # Use tqdm for a nice progress bar
    iterator = range(0, len(logs_to_vectorize), batch_size)
    for i in tqdm(iterator, desc="Vectorizing Batches"):
        batch_event_ids = event_ids[i : i + batch_size]
        batch_logs = logs_to_vectorize[i : i + batch_size]
        
        # Tokenize the batch of log messages
        inputs = tokenizer(
            batch_logs,
            padding=True,      # Pad to the length of the longest sequence in the batch
            truncation=True,   # Truncate to the model's max input size (e.g., 512 for BERT)
            max_length=512,
            return_tensors='pt' # Return PyTorch tensors
        )

        # Move tokenized inputs to the selected device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get embeddings from BERT. We don't need gradients for this.
        with torch.no_grad():
            outputs = model(**inputs)

        # The most common way to get a sentence embedding from BERT is to
        # take the hidden state of the first token, which is the [CLS] token.
        # Shape: (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Store the vectors in our map, moving them to CPU to save GPU memory
        for event_id, vector in zip(batch_event_ids, cls_embeddings):
            vector_map[event_id] = vector.cpu()

    # --- 5. Save the Vector Map ---
    click.echo(f"Saving vector map to {output_path}...")
    torch.save(vector_map, output_path)
    click.secho("Vectorization complete!", fg="green", bold=True)


