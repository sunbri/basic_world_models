import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: For this world model, we are NOT learning rewards, termination, etc. This means
# that the RL agent CANNOT learn in imagination, and so is not a full-fledged "Dreamer" type
# world model

# Overarching goal: given hidden state, action, and compressed image, predict the
# next compressed image. By proxy it should have learned features.
# GRU: ~843k params
# MDN head: ~246k params
# Total: ~1.1 million params, small model
class MDNRNN(nn.Module):
    def __init__(self, latent_size=32, action_size=3, hidden_size=512, num_gaussians=5):
        super().__init__()
        self.latent_size = latent_size
        self.num_gaussians = num_gaussians
        self.hidden_size = hidden_size

        # 1. GRU
        # Input: z_t (32) + a_t (3) = 32
        self.rnn = nn.GRU(latent_size + action_size, hidden_size, batch_first=True)

        # 2. The Mixer (MDN Output)
        # Output mu, sigma, log_pi for each latent dimension
        self.mdn = nn.Linear(hidden_size, latent_size * num_gaussians * 3)

    def forward(self, action, latent, hidden=None):
        """
        action: (Batch, Seq, 3)
        latent: (Batch, Seq, 32)
        hidden: (1, Batch, 512) - The previous state
        """
        # concatenate inputs and run GRU
        inputs = torch.cat([latent, action], dim=-1)
        rnn_out, next_hidden = self.rnn(inputs, hidden)

        out = self.mdn(rnn_out)
        out = out.view(out.size(0), out.size(1), self.latent_size, self.num_gaussians, 3)
        mu = out[:, :, :, :, 0]
        log_sigma = out[:, :, :, :, 1]
        log_pi = out[:, :, :, :, 2]

        # apply constraints for math
        sigma = torch.exp(log_sigma)
        log_pi = F.log_softmax(log_pi, dim=-1)

        return mu, sigma, log_pi, next_hidden

def mdn_loss_function(mu, sigma, log_pi, target):
    """
    Calculates Negative Log Likelihood (NLL) of the target z_{t+1} 
    given the mixture distribution predicted by the RNN.
    
    mu, sigma, log_pi: (Batch, Seq, 32, 5)
    target: (Batch, Seq, 32)
    """
    # 1. Expand target to match the 5 gaussians
    # Target becomes (Batch, Seq, 32, 1) -> (Batch, Seq, 32, 5)
    target = target.unsqueeze(-1).expand_as(mu)

    # 2. Gaussian Probability Formula (Log Space)
    # log N(x) = -0.5 * log(2pi) - log(sigma) - 0.5 * ((x - mu)/sigma)^2
    dist = torch.distributions.Normal(mu, sigma)
    log_probs = dist.log_prob(target)  # (Batch, Seq, 32, 5)

    # 3. Combine with Mixture Weights (Pi)
    # We want: log( sum( pi * N(x) ) )
    # In log space: LogSumExp( log_pi + log_probs )
    weighted_log_probs = log_pi + log_probs
    
    # Sum across the 5 gaussians (dim -1)
    log_prob_sum = torch.logsumexp(weighted_log_probs, dim=-1) # (Batch, Seq, 32)

    # 4. Average over all dimensions and batch AND latent size
    # We negate because we want to Minimize Negative Log Likelihood
    return -log_prob_sum.mean()