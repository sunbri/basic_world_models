from data.loaders import LatentSeqDataset
from models.mdn_rnn import MDNRNN, mdn_loss_function
from utils.misc import LATENT_SIZE, ACTION_SIZE, HIDDEN_SIZE, save_checkpoint

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from os.path import join, exists

# length of sequence to train on
SEQ_LEN = 32
BATCH_SIZE = 64

def main():
    parser = argparse.ArgumentParser(description='Train MDN-RNN')
    parser.add_argument('--data-dir', type=str, default='datasets/carracing_latent')
    parser.add_argument('--save-dir', type=str, default='logs/mdn')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--no-reload', action='store_true',
                        help='Best model is not reloaded if specified')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() 
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    if not exists(args.save_dir):
        os.makedirs(args.save_dir)
    best_filename = join(args.save_dir, 'best.tar')
    filename = join(args.save_dir, 'checkpoint.tar')

    print("Loading dataset...")
    train_dataset = LatentSeqDataset(args.data_dir, seq_len=SEQ_LEN, mode='train', epoch_repeat=50)
    test_daaset = LatentSeqDataset(args.data_dir, seq_len=SEQ_LEN, mode='test', epoch_repeat=10)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_daaset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MDNRNN(latent_size=LATENT_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    # standard Adam for first run through: decay might mess things up as it shifts
    # the "memory"
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    print(f"Training...")

    best_test_loss = None

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
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
            total_train_loss += loss.item()
            
            # Clip gradients (Crucial for RNNs to prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Test
        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                z_in = batch['z_input'].to(device)    # (B, Seq, 32)
                a_in = batch['a_input'].to(device)    # (B, Seq, 3)
                z_tgt = batch['z_target'].to(device)  # (B, Seq, 32)

                mus, sigmas, logpi, _ = model(a_in, z_in)
                loss = mdn_loss_function(mus, sigmas, logpi, z_tgt)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        scheduler.step(avg_test_loss)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Test Loss:  {avg_test_loss:.4f}")

        is_best = not best_test_loss or total_test_loss < best_test_loss
        if is_best:
            best_test_loss = total_test_loss
            print(f"  New Best!")

        save_checkpoint({
            'epoch': epoch,
            'precision': total_test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, filename, best_filename)

if __name__ == '__main__':
    main()