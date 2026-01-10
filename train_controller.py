from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller
from utils.misc import ACTION_SIZE, HIDDEN_SIZE, LATENT_SIZE
from utils.wrappers import ResizeObservationWrapper

import argparse
import torch
import numpy as np
import cma
import gymnasium as gym
import os
from os.path import join, exists
from torchvision import transforms

import cProfile
import pstats
import io

# Configuration
POPULATION_SIZE = 16 # candidates per generation
GENERATIONS = 100
TARGET_RETURN = 900 # Goal of algorithm
ROLLOUTS_PER_AGENT = 3

def get_reward(env: gym.Env, vae, rnn, controller, device, seed=None):
    """
    Returns reward from one episode
    """
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()

    total_reward = 0
    done = False

    hidden = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    # Transform: Converts numpy (H, W, C) -> Tensor (C, H, W) and scales 0-1
    transform = transforms.Compose([transforms.ToTensor()])

    with torch.no_grad():
        while not done:
            # vision
            obs_tensor = transform(obs).unsqueeze(0).to(device) # (1, 3, 64, 64)
            feat = vae.encoder(obs_tensor)
            feat = feat.reshape(feat.size(0), -1)
            z = vae.fc_mu(feat)

            # controller
            action = controller(z, hidden.squeeze(0))
            action = action.cpu().numpy()[0]

            # step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # memory update: pretend we have sequence of length 1
            action_tensor = torch.FloatTensor(action).view(1, 1, ACTION_SIZE).to(device)
            z_tensor = z.unsqueeze(1) # (1, 1, LATENT_SIZE)

            # forward pass MDNRNN
            _, _, _, next_hidden = rnn(action_tensor, z_tensor, hidden)
            hidden = next_hidden
    
    return total_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae-path', default='logs/vae/best.tar')
    parser.add_argument('--mdn-path', default='logs/mdn/best.tar')
    parser.add_argument('--save-dir', default='logs/controller')
    parser.add_argument('--profile', action='store_true', help='Profile get_reward and exit')
    args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() 
    #                     else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device("cpu")
    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load pre-trained
    print("Loading VAE and MDN...")
    vae = VAE(image_channels=3, latent_size=LATENT_SIZE).to(device)
    vae.load_state_dict(torch.load(args.vae_path, map_location=device)['state_dict'])
    vae.eval()

    rnn = MDNRNN(latent_size=LATENT_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    rnn.load_state_dict(torch.load(args.mdn_path, map_location=device)['state_dict'])
    rnn.eval()

    # set up controller
    controller = Controller(z_size=LATENT_SIZE, h_size=HIDDEN_SIZE, a_size=ACTION_SIZE).to(device)
    parameters = controller.get_parameters_flat()
    print(f"Controller has {len(parameters)} parameters to optimize.")

    # initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0=parameters, sigma0=0.1, inopts={'popsize': POPULATION_SIZE})

    # evolution loop
    base_env = gym.make('CarRacing-v3', render_mode=None)
    env = ResizeObservationWrapper(base_env, shape=(64, 64))

    # ------------- Profiling ---------------
    if args.profile:
        print("Starting profile mode for get_reward()...")

        pr = cProfile.Profile()
        pr.enable()
        get_reward(env, vae, rnn, controller, device)
        pr.disable()

        s = io.StringIO()
        sortby = 'cumtime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(20)

        print(s.getvalue())
        print("Profiling complete. Exiting.")
        return
    # ---------------------------------------


    # tracking global best
    best_global_avg = -np.inf

    print("Starting evolution")
    for generation in range(GENERATIONS):
        solutions = es.ask()
        rewards = []

        for sol in solutions:
            controller.set_parameters_flat(sol)
            agent_rewards = []
            for _ in range(ROLLOUTS_PER_AGENT):
                reward = get_reward(env, vae, rnn, controller, device)
                agent_rewards.append(reward)

            avg_reward = sum(agent_rewards) / len(agent_rewards)
            rewards.append(avg_reward)

        # minimize negative reward = maximize reward
        es.tell(solutions, [-r for r in rewards])

        # stats
        curr_best_idx = np.argmax(rewards)
        curr_best_reward = rewards[curr_best_idx]
        avg_reward = np.mean(rewards)

        print(f"Generation {generation} | Best: {curr_best_reward:1f} | Avg: {avg_reward:1f}")

        # Save Best
        if curr_best_reward > best_global_avg:
            best_global_avg = curr_best_reward
            controller.set_parameters_flat(solutions[curr_best_idx])
            torch.save(controller.state_dict(), join(args.save_dir, 'best.tar'))
            print(f"--> New Record! Saved best.tar (Avg: {best_global_avg:.1f})")

        if curr_best_reward > TARGET_RETURN:
            print("Target Solved!")
            break

if __name__ == '__main__':
    main()