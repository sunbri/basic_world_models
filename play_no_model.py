import argparse
import torch
import numpy as np
import gymnasium as gym
import os
from torchvision import transforms

# Imports
from models.vae import VAE
from models.controller_no_model import Controller
from utils.wrappers import ResizeObservationWrapper
from utils.misc import LATENT_SIZE, ACTION_SIZE

def play(args):
    device = torch.device("cuda" if torch.cuda.is_available() 
                else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    print(f"Device: {device}")
    print("Loading VAE and Vision-Only Controller...")

    # 1. Load VAE
    vae = VAE(image_channels=3, latent_size=LATENT_SIZE).to(device)
    vae.load_state_dict(torch.load(args.vae_path, map_location=device)['state_dict'])
    vae.eval()

    # 2. Load Controller (Vision Only)
    controller = Controller(z_size=LATENT_SIZE, a_size=ACTION_SIZE).to(device)
    controller.load_state_dict(torch.load(args.controller_path, map_location=device))
    controller.eval()

    print("Models loaded. Starting simulation...")

    # 3. Setup Environment
    render_mode = "human" if not args.record else "rgb_array"
    env = gym.make('CarRacing-v3', render_mode=render_mode)
        
    env = ResizeObservationWrapper(env, shape=(64, 64))

    if args.record:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(env, 
                          video_folder="videos_vision_only", 
                          name_prefix="vision_agent",
                          episode_trigger=lambda x: True)

    # 4. Run Loop
    for episode in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        # Transform
        transform = transforms.Compose([transforms.ToTensor()])
        
        print(f"--- Episode {episode + 1} Start ---")
        
        with torch.no_grad():
            while not done:
                # A. Vision
                obs_tensor = transform(obs).unsqueeze(0).to(device)
                feat = vae.encoder(obs_tensor)
                feat = feat.reshape(feat.size(0), -1)
                z = vae.fc_mu(feat) # Deterministic

                # B. Controller (Vision Only)
                action = controller(z)
                action = action.cpu().numpy()[0]

                # C. Step
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
                
        print(f"Episode {episode + 1} Finished. Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae-path', default='logs/vae/best.tar', help='Path to VAE weights')
    parser.add_argument('--controller-path', default='logs/controller_no_model/best.tar', help='Path to Best Controller')
    parser.add_argument('--episodes', type=int, default=3, help='Number of test runs')
    parser.add_argument('--record', action='store_true', help='Record video')
    args = parser.parse_args()

    play(args)