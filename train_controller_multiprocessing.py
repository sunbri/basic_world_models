from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller
from utils.misc import ACTION_SIZE, HIDDEN_SIZE, LATENT_SIZE, RESIZE_SIZE
from utils.wrappers import ResizeObservationWrapper

import argparse
import torch
import multiprocessing as mp
import numpy as np
import cma
import gymnasium as gym
import os
from os.path import join, exists
from torchvision import transforms

# Configuration
POPULATION_SIZE = 32 # candidates per generation
GENERATIONS = 100
TARGET_RETURN = 900 # Goal of algorithm
ROLLOUTS_PER_AGENT = 5

# global variables for workers to avoid pickling
# we do not want to send over env and vae and rnn
g_env = None
g_vae = None
g_rnn = None
g_controller = None
g_device = None
g_transform = None

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

def init_worker(vae_path, mdn_path, latent_size, hidden_size, action_size):
    global g_env, g_vae, g_rnn, g_controller, g_device

    g_device = torch.device('cpu')
    base_env = gym.make('CarRacing-v3', render_mode=None)
    g_env = ResizeObservationWrapper(base_env, shape=(RESIZE_SIZE, RESIZE_SIZE))

    g_vae = VAE(image_channels=3, latent_size=LATENT_SIZE).to(g_device)
    g_vae.load_state_dict(torch.load(vae_path, map_location=g_device)['state_dict'])
    g_vae.eval()

    g_rnn = MDNRNN(latent_size=LATENT_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE).to(g_device)
    g_rnn.load_state_dict(torch.load(mdn_path, map_location=g_device)['state_dict'])
    g_rnn.eval()

    g_controller = Controller(z_size=LATENT_SIZE, h_size=HIDDEN_SIZE, a_size=ACTION_SIZE).to(g_device)

def evaluate_solution(solution):
    """
    Execute by pool.map
    """
    global g_controller, g_env, g_vae, g_rnn, g_device
    g_controller.set_parameters_flat(solution)

    agent_rewards = []
    for _ in range(ROLLOUTS_PER_AGENT):
        r = get_reward(g_env, g_vae, g_rnn, g_controller, g_device)
        agent_rewards.append(r)

    return sum(agent_rewards) / len(agent_rewards)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae-path', default='logs/vae/best.tar')
    parser.add_argument('--mdn-path', default='logs/mdn/best.tar')
    parser.add_argument('--save-dir', default='logs/controller')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count(), help='Number of parallel workers')
    args = parser.parse_args()
    if not exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # master controller for saving weights
    main_device = torch.device("cpu")
    controller = Controller(z_size=LATENT_SIZE, h_size=HIDDEN_SIZE, a_size=ACTION_SIZE).to(main_device)
    parameters = controller.get_parameters_flat()
    print(f"Controller has {len(parameters)} parameters to optimize.")

    # initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0=parameters, sigma0=0.1, inopts={'popsize': POPULATION_SIZE})
    best_global_avg = -np.inf

    print(f"Initializing pool with {args.num_workers} workers...")
    init_args = (args.vae_path, args.mdn_path, LATENT_SIZE, HIDDEN_SIZE, ACTION_SIZE)
    pool = mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=init_args)

    try:
        for generation in range(GENERATIONS):
            solutions = es.ask()
            rewards = pool.map(evaluate_solution, solutions)
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
    except KeyboardInterrupt:
        print("Interrupted! Closing pool...")
    finally:
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()