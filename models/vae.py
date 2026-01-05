import torch
import torch.nn as nn
import torch.nn.functional as F

# Variational Autoencoder built as a CNN
class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_size=32):
        super().__init__()
        self.latent_size = latent_size

        # ==========================================================
        # Encoder
        # Conv2d: different filter for each input/output channel
        # Input: (B, 3, 64, 64)
        # ==========================================================
        self.encoder = nn.Sequential(
            # Layer 1: 3 -> 32 filters, kernel 4, stride 2 (Valid padding)
            # Output: (B, 32, 31, 31)
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),

            # Layer 2: 32 -> 64 filters
            # Output: (B, 64, 14, 14)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Layer 3: 64 -> 128 filters
            # Output: (B, 128, 6, 6)
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),

            # Layer 4: 128 -> 256 filters
            # Output: (B, 256, 2, 2)
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU()
        )

        # Flatten size: 256 filters * 2 * 2 = 1024
        self.feature_dim = 256 * 2 * 2
        
        # Latent heads (Mu and LogVar)
        # Dense layers from the flattened features
        self.fc_mu = nn.Linear(self.feature_dim, latent_size)
        # Learn log variance since variance must be positive
        self.fc_logvar = nn.Linear(self.feature_dim, latent_size)

        # ==========================================================
        # Decoder
        # Input: Latent Z -> Output: (B, 3, 64, 64)
        # ==========================================================
        self.decoder = nn.Sequential(
            # Expand Latent to 1024 features
            nn.Linear(latent_size, 1024),

            # Reshape from (B, 1024) -> (B, 1024, 1, 1)
            nn.Unflatten(dim=1, unflattened_size=(1024, 1, 1)),

            # Layer 1: Input (1024, 1, 1) -> Output (128, 5, 5)
            # Math: (1-1)*2 + 5 = 5
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            
            # Layer 2: Input (128, 5, 5) -> Output (64, 13, 13)
            # Math: (5-1)*2 + 5 = 13
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            
            # Layer 3: Input (64, 13, 13) -> Output (32, 30, 30)
            # Math: (13-1)*2 + 6 = 30
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            
            # Layer 4: Input (32, 30, 30) -> Output (3, 64, 64)
            # Math: (30-1)*2 + 6 = 64
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid()
        )

    # Reparameterization "trick" for gradient flow
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            # unit normal with shape of std
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # for inference, just return the mean
            return mu
    
    def forward(self, x):
        # 1. Encode
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1) # Flatten

        # 2. Latent States
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 3. Sample
        z = self.reparameterize(mu, logvar)

        # 4. Decode
        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes VAE ELBO loss 
    """
    # 1. Reconstruction Loss (Sum over pixels, then Mean over batch)
    # reduction='sum' sums ALL pixels in the batch. We divide by batch_size to get the average per-image error.
    # POSSIBLY SWITCH TO BCE!
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    recon_loss = recon_loss.div(x.size(0))

    # 2. KL Divergence (Sum over latent dimensions, then Mean over batch)
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # We sum over the latent dimension (dim=1), resulting in a vector of shape (Batch_Size,)
    kl_elementwise = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    # Then we take the mean of that vector to get the average KL per image
    kl_loss = kl_elementwise.mean()

    # Total Loss
    total_loss = recon_loss + (beta * kl_loss)
    
    return total_loss, recon_loss, kl_loss