import argparse
import torch
import numpy as np
import gymnasium as gym
import os
from torchvision import transforms

# Imports from your project structure
from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller
from utils.wrappers import ResizeObservationWrapper
from utils.misc import LATENT_SIZE, HIDDEN_SIZE, ACTION_SIZE

def play(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print("Loading models...")

    # 1. Load VAE
    vae = VAE(image_channels=3, latent_size=LATENT_SIZE).to(device)
    vae.load_state_dict(torch.load(args.vae_path, map_location=device)['state_dict'])
    vae.eval()

    # 2. Load MDN-RNN
    rnn = MDNRNN(latent_size=LATENT_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    rnn.load_state_dict(torch.load(args.mdn_path, map_location=device)['state_dict'])
    rnn.eval()

    # 3. Load Best Controller
    controller = Controller(z_size=LATENT_SIZE, h_size=HIDDEN_SIZE, a_size=ACTION_SIZE).to(device)
    controller.load_state_dict(torch.load(args.controller_path, map_location=device))
    controller.eval()

    print("Models loaded. Starting simulation...")

    # 4. Setup Environment
    render_mode = "human" if not args.record else "rgb_array"
    # Note: Using v2 as it is the standard stable version, but v3 works if installed
    env = gym.make('CarRacing-v3', render_mode=render_mode)
        
    env = ResizeObservationWrapper(env, shape=(64, 64))

    # Optional: Record Video
    if args.record:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(env, video_folder="videos", name_prefix="best_agent_run")
        print("Recording video to /videos folder...")

    # 5. Run Loop
    # We run for a few episodes to see consistency
    for episode in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        # Initialize Hidden State (Same as training)
        hidden = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
        transform = transforms.Compose([transforms.ToTensor()])

        print(f"--- Episode {episode + 1} Start ---")
        
        with torch.no_grad():
            while not done:
                # A. Vision
                obs_tensor = transform(obs).unsqueeze(0).to(device)
                feat = vae.encoder(obs_tensor)
                feat = feat.reshape(feat.size(0), -1)
                z = vae.fc_mu(feat) # Deterministic check (using mean)

                # B. Controller (Reflex)
                # Input: Latent + Hidden Memory
                action = controller(z, hidden.squeeze(0))
                action = action.cpu().numpy()[0]

                # C. Step
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

                # D. Memory Update (Dream)
                # Shape logic: (Batch=1, Sequence=1, Feature)
                action_tensor = torch.FloatTensor(action).view(1, 1, ACTION_SIZE).to(device)
                z_tensor = z.unsqueeze(1) 
                
                _, _, _, next_hidden = rnn(action_tensor, z_tensor, hidden)
                hidden = next_hidden
                
                # Optional: Stop if it gets stuck (negative reward loop)
                if step > 100 and total_reward < -50:
                    print("Agent is stuck. Killing episode.")
                    break
        
        print(f"Episode {episode + 1} Finished. Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae-path', default='logs/vae/best.tar', help='Path to VAE weights')
    parser.add_argument('--mdn-path', default='logs/mdn/best.tar', help='Path to MDN-RNN weights')
    parser.add_argument('--controller-path', default='logs/controller/best.tar', help='Path to Best Controller')
    parser.add_argument('--episodes', type=int, default=3, help='Number of test runs')
    parser.add_argument('--record', action='store_true', help='Record video instead of showing window')
    args = parser.parse_args()

    play(args)