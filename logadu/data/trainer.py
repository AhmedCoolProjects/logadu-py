# In logadu/deep_learning/trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm
import os
from pathlib import Path

# Updated import
from logadu.data.dataloader import get_deeplog_dataloader 
from logadu.models.deeplog import DeepLog

def train_deeplog(args):
    """ Main training function for the DeepLog model. """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_name = Path(args.dataset_file).stem
    print(f"Training DeepLog model on dataset: {file_name}")
    
    # -- 1. Create Dataloader and get vocabulary size --
    # The dataloader now handles everything internally.
    dataloader, num_labels = get_deeplog_dataloader(
        sequences_file=args.dataset_file,
        batch_size=args.batch_size
    )
    print(f"Model will be trained with a vocabulary size of: {num_labels}")
    
    # -- 2. Initialize model, loss, and optimizer --
    model = DeepLog(
        num_labels=num_labels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # -- 3. Training loop --
    print("Starting training...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for step, (seq_window, next_event_label) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            seq_window = seq_window.to(device)
            next_event_label = next_event_label.to(device)
            
            output = model(seq_window)
            loss = criterion(output, next_event_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Average Loss: {avg_loss:.4f}")

    # -- 4. Save the model --
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    model_path = os.path.join(args.output_dir, f'deeplog_{file_name}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dataloader.dataset.vocab, # Save the vocabulary with the model!
        'window_size': args.window_size
    }, model_path)
    print(f"Training complete. Model and vocabulary saved to {model_path}")