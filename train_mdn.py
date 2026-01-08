from data.loaders import LatentSeqDataset
from models.mdn_rnn import MDNRNN, mdn_loss_function
from utils.misc import LATENT_SIZE, ACTION_SIZE

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from os.path import join, exists

# length of sequence to train on
SEQ_LEN = 32
BATCH_SIZE = 64
HIDDEN_SIZE = 512

def main():
    parser = argparse.ArgumentParser(description='Train MDN-RNN')
    parser.add_argument('--data-dir', type=str, default='datasets/carracing_latent')
    parser.add_argument('--save-dir', type=str, default='logs/mdn')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() 
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Loading dataset...")
    dataset = LatentSeqDataset(args.data_dir, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = MDNRNN(latent_size=LATENT_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    # standard Adam for first run through: decay might mess things up as it shifts
    # the "memory"
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Training on {len(dataset)} episodes...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(loader):
            z_in = batch['z_input'].to(device)    # (B, Seq, 32)
            a_in = batch['a_input'].to(device)    # (B, Seq, 3)
            z_tgt = batch['z_target'].to(device)  # (B, Seq, 32)

            optimizer.zero_grad()

            # Forward pass
            # Note: We pass hidden=None to initialize with zeros for each new batch
            # This is standard for "Stateless" training (random chunks)
            mus, sigmas, logpi, _ = model(a_in, z_in)
        
            # Calculate Loss
            loss = mdn_loss_function(mus, sigmas, logpi, z_tgt)

            # Backward
            loss.backward()
            
            # Clip gradients (Crucial for RNNs to prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(loader)
        print(f"====> Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        torch.save(model.state_dict(), join(args.save_dir, 'best.tar'))

if __name__ == '__main__':
    main()