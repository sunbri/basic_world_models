import argparse
import torch
import numpy as np
import os
from os.path import join, exists
from torchvision import transforms
from models.vae import VAE

def main():
    parser = argparse.ArgumentParser(description='Extract Latent Data')
    parser.add_argument('--data-dir', type=str, default='datasets/carracing', help='Input .npz files')
    parser.add_argument('--save-dir', type=str, default='datasets/carracing_latent', help='Output directory')
    parser.add_argument('--vae-path', type=str, default='logs/vae/best.tar', help='Path to trained VAE')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() 
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 1. Load VAE
    print(f"Loading VAE from {args.vae_path}...")
    model = VAE(image_channels=3, latent_size=32).to(device)
    checkpoint = torch.load(args.vae_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    files = [f for f in os.listdir(args.data_dir) if f.endswith('.npz')]
    files.sort()
    print(f"Found {len(files)} episodes.")

    # 2. Transform (MATCHING YOUR TRAINING!)
    # No ToPILImage, No Resize (assuming raw data is good)
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])

    print("Starting extraction...")
    with torch.no_grad():
        for i, filename in enumerate(files):
            filepath = join(args.data_dir, filename)
            
            # Load Data
            data = np.load(filepath)
            obs = data['observations']       # Expected: (1001, 64, 64, 3)
            action = data['actions'] # Expected: (1000, 3)

            # Debug check on first file
            if i == 0:
                print(f"Debug | Obs Shape: {obs.shape}, Action Shape: {action.shape}")

            # Prepare Batch
            obs_tensor = [transform(img) for img in obs]
            obs_tensor = torch.stack(obs_tensor).to(device)

            # VAE Pass (Encoder Only)
            h = model.encoder(obs_tensor)
            h = torch.flatten(h, start_dim=1)
            mu = model.fc_mu(h)
            logvar = model.fc_logvar(h)

            # Save
            save_path = join(args.save_dir, filename)
            np.savez_compressed(
                save_path, 
                mu=mu.cpu().numpy(), 
                logvar=logvar.cpu().numpy(), 
                action=action
            )

            if i % 100 == 0:
                print(f"Processed {i} / {len(files)}...")

    print("Extraction complete.")

if __name__ == "__main__":
    main()