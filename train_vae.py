from models.vae import VAE, vae_loss_function
from data.loaders import BufferedRolloutDataset
from utils.misc import RESIZE_SIZE, LATENT_SIZE, save_checkpoint

import argparse
import glob
from os.path import exists, join
from os import mkdir
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

def main():
    parser = argparse.ArgumentParser(description='VAE Trainer')
    parser.add_argument('--data-dir', type=str, default='datasets/carracing', 
                        help='Path to .npz files')
    parser.add_argument('--save-dir', type=str, default='logs', 
                        help='Path to save model data')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-reload', action='store_true',
                        help='Best model is not reloaded if specified')
    parser.add_argument('--no-samples', action='store_true',
                        help='Does not save samples during training')

    args = parser.parse_args()
    if not exists(args.save_dir):
        mkdir(args.save_dir)

    vae_dir = join(args.save_dir, 'vae')
    if not exists(vae_dir):
        mkdir(vae_dir)
        mkdir(join(vae_dir, 'samples'))

    torch.manual_seed(123)
    mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() 
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # get train and test data
    all_files = sorted(glob.glob(join(args.data_dir, "*.npz")))
    random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    print(f"Total Files: {len(all_files)}")
    print(f"Train Split: {len(train_files)} files")
    print(f"Test Split:  {len(test_files)} files")

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = BufferedRolloutDataset(
        source=train_files,
        mode='ae',
        transform=transform_train
    )
    test_dataset = BufferedRolloutDataset(
        source=test_files,
        mode='ae',
        transform=transform_test
    )

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=4, 
                              shuffle=False)
    test_loader = DataLoader(test_dataset, 
                             batch_size=args.batch_size, 
                             num_workers=2, 
                             shuffle=False)

    model = VAE(latent_size=32).to(device)

    estimated_total_frames = len(train_files) * 1000
    steps_per_epoch = estimated_total_frames // args.batch_size

    # AdamW with milder weight decay for a "sharper" decoder
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        epochs=args.epochs
    )

    epoch = 1

    # reload best so far
    reload_file = join(vae_dir, 'best.tar')
    if not args.no_reload and exists(reload_file):
        state = torch.load(reload_file)
        print(f"Reloading model after epoch {state['epoch']} \
and test error {state['precision']}")
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        # we save at the end
        epoch = state['epoch'] + 1

    curr_best = None

    # start at epoch 1
    while epoch < args.epochs + 1:
        model.train()
        train_loss = 0
        batch_count = 0

        # Training Loop
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            batch_count += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} \
(Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f})")
        
        avg_loss = train_loss / batch_count
        print(f"====> Epoch {epoch} Average Loss: {avg_loss:.4f}")

        # Validation Loop
        model.eval()
        test_total_loss = 0
        test_batch_count = 0
        with torch.no_grad():
            for test_batch_index, x in enumerate(test_loader):
                x = x.to(device)
                recon_x, mu, logvar = model(x)
                test_loss, test_recon_loss, test_kl_loss = vae_loss_function(recon_x, x, mu, logvar)
                test_total_loss += test_loss.item()
                test_batch_count += 1
                if test_batch_index % 100 == 0:
                    print(f"Test | Loss: {test_loss.item():.4f} \
(Recon: {test_recon_loss.item():.4f}, KL: {test_kl_loss.item():.4f})")

            test_total_loss /= test_batch_count
            print(f"====> Test set loss: {test_total_loss:.4f}")

        # Checkpointing
        best_filename = join(vae_dir, 'best.tar')
        filename = join(vae_dir, 'checkpoint.tar')
        is_best = not curr_best or test_loss < curr_best
        if is_best:
            curr_best = test_loss

        save_checkpoint({
            'epoch': epoch,
            'precision': test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, filename, best_filename)

        # sample from VAE
        with torch.no_grad():
            sample = torch.randn(64, LATENT_SIZE).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, RESIZE_SIZE, RESIZE_SIZE),
                        join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))
    
    epoch += 1

if __name__ == "__main__":
    main()