from models.vae import VModel, vae_loss_function

import glob
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, random_split
from torchvision import transforms

class StreamingRolloutDataset(IterableDataset):
    """
    Efficiently loads 'chunks' (entire .npz files) at once to minimize disk seeking.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.file_paths = glob.glob(os.path.join(data_dir, "*.npz"))
        self.transform = transform
        
        if not self.file_paths:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        # Calculate approximate length for progress bars (optional)
        # Assuming ~1000 frames per file
        self.est_length = len(self.file_paths) * 1000 

    def process_data(self, data):
        """Convert uint8 (H, W, C) -> float32 (C, H, W) normalized"""
        # 1. Convert to Float Tensor [0.0, 1.0]
        tensor = torch.from_numpy(data).float() / 255.0
        # 2. Permute dimensions (H,W,C) -> (C,H,W)
        tensor = tensor.permute(2, 0, 1)
        return tensor

    def __iter__(self):
        """
        This is where the magic happens. 
        Each Worker process calls this function independently.
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. Split files among workers
        if worker_info is None:
            # Single-process data loading
            my_files = self.file_paths
        else:
            # Multi-process: Split the file list evenly
            per_worker = int(np.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_paths))
            my_files = self.file_paths[start:end]

        # 2. Shuffle files so workers don't always read in the same order (optional but good)
        np.random.shuffle(my_files)

        # 3. The "Chunk" Loading Loop
        for fpath in my_files:
            try:
                # FAST: Load the entire 1000-frame chunk into RAM at once
                with np.load(fpath) as data:
                    observations = data['observations'] # Shape (T, 64, 64, 3)

                # MIXING: Shuffle indices *within* this chunk.
                # This prevents the batch from having 128 consecutive frames (which are identical).
                indices = np.random.permutation(len(observations))

                # Yield frames one by one from the cached chunk
                for idx in indices:
                    yield self.process_data(observations[idx])

            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue

    def __len__(self):
        # Only used by tqdm for progress bar estimates
        return self.est_length

def train(args):
    device = torch.cuda("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    dataset = StreamingRolloutDataset(args.data_dir)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True
    )

    model = VModel(latent_size=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    print(f"Starting training on approximately {len(dataset)} frames...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, x in enumerate(loader):
            x = x.to(device)

            optimizer.zero_grad()

            # Forward
            recon_x, mu, logvar = model(x)

            # Loss
            loss = vae_loss_function(recon_x, x, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item() / len(x):.4f}")
        
        avg_loss = total_loss / batch_count
        print(f"====> Epoch {epoch} Average Loss: {avg_loss:.4f}")

        save_path = os.path.join(args.save_dir, "vae_latest.pth")
        torch.save(model.state_dict(), save_path)

        scheduler.step(avg_loss)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/carracing', help='Path to .npz files')
    parser.add_argument('--save_dir', type=str, default='weights', help='Path to save model weights')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
