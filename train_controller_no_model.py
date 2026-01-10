from models.vae import VAE
from models.controller_no_model import Controller
from utils.misc import ACTION_SIZE, LATENT_SIZE
from utils.wrappers import ResizeObservationWrapper

import argparse
import torch
import numpy as np
import cma
import gymnasium as gym
import os
from os.path import join, exists
from torchvision import transforms

# Configuration
POPULATION_SIZE = 16 # candidates per generation
GENERATIONS = 100
TARGET_RETURN = 900 # Goal of algorithm

def get_reward(env: gym.Env, vae, controller, device):
    """
    Returns reward from one episode
    """
    obs, _ = env.reset()
    total_reward = 0
    done = False

    transform = transforms.Compose([transforms.ToTensor()])

    with torch.no_grad():
        while not done:
            # vision
            obs_tensor = transform(obs).unsqueeze(0).to(device) # (1, 3, 64, 64)
            feat = vae.encoder(obs_tensor)
            feat = feat.reshape(feat.size(0), -1)
            z = vae.fc_mu(feat)

            # controller
            action = controller(z)
            action = action.cpu().numpy()[0]

            # step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

    return total_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae-path', default='logs/vae/best.tar')
    parser.add_argument('--save-dir', default='logs/controller_no_model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() 
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load pre-trained
    print("Loading VAE and MDN...")
    vae = VAE(image_channels=3, latent_size=LATENT_SIZE).to(device)
    vae.load_state_dict(torch.load(args.vae_path, map_location=device)['state_dict'])
    vae.eval()

    # set up controller
    controller = Controller(z_size=LATENT_SIZE, a_size=ACTION_SIZE).to(device)
    parameters = controller.get_parameters_flat()
    print(f"Controller has {len(parameters)} parameters to optimize.")

    # initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0=parameters, sigma0=0.1, inopts={'popsize': POPULATION_SIZE})

    # evolution loop
    base_env = gym.make('CarRacing-v3', render_mode=None)
    env = ResizeObservationWrapper(base_env, shape=(64, 64))

    # tracking global best
    best_global_reward = -np.inf

    print("Starting evolution")
    for generation in range(GENERATIONS):
        solutions = es.ask()
        rewards = []

        for i, sol in enumerate(solutions):
            controller.set_parameters_flat(sol)
            reward = get_reward(env, vae, controller, device)
            rewards.append(reward)

        # minimize negative reward = maximize reward
        es.tell(solutions, [-r for r in rewards])

        # stats
        best_reward = max(rewards)
        avg_reward = sum(rewards) / len(rewards)
        es.disp()
        print(f"Generation {generation} | Best: {best_reward:2f} | Avg: {avg_reward:2f}")

        # save best agent
        if best_reward > best_global_reward:
            best_global_reward = best_reward
            best_idx = np.argmax(rewards)
            best_params = solutions[best_idx]

            controller.set_parameters_flat(best_params)
            torch.save(controller.state_dict(), join(args.save_dir, 'best.tar'))
            print(f"--> New record! Saved best.tar with reward: {best_global_reward:.2f}")

        if best_reward > TARGET_RETURN:
            print("Target Sovled!")
            break

if __name__ == '__main__':
    main()