from utils.wrappers import ResizeObservationWrapper
from utils.misc import RESIZE_LENGTH

import gymnasium as gym
import math
import multiprocessing
import numpy as np
import os

def sample_brownian_action(action_space: gym.spaces.Box, seq_len, dt):
    """ 
    Brownian motion policy: a_{t+1} = a_t + sqrt(dt) N(0, 1)
    """
    # initial sample
    actions = [action_space.sample()]
    for _ in range(seq_len):
        brow = np.random.standard_normal(action_space.shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * brow,
                    action_space.low, action_space.high)
        )
    return actions

def collect_single_rollout(args):
    rollout_idx, data_dir, noise_type = args
    # Reseed numpy each time
    np.random.seed()

    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = ResizeObservationWrapper(env, shape=(RESIZE_LENGTH, RESIZE_LENGTH))

    # usually only goes 1000 steps
    seq_len = 1000
    obs, _ = env.reset()

    if noise_type == "white":
        # +1 to match brownian length
        action_rollout = [env.action_space.sample() for _ in range(seq_len + 1)]
    elif noise_type == "brownian":
        action_rollout = sample_brownian_action(env.action_space, seq_len, 1. / 50)

    obs_rollout = [obs]
    reward_rollout = []
    done_rollout = []

    t = 0
    while True:
        action = action_rollout[t]
        t += 1
        
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_rollout.append(obs)
        reward_rollout.append(reward)
        
        is_done = terminated or truncated
        done_rollout.append(is_done)
        if is_done:
            break

    env.close()

    actual_steps = len(reward_rollout)            
    np.savez_compressed(
        os.path.join(data_dir, f"rollout_{rollout_idx}"),
        observations=np.array(obs_rollout),
        actions=np.array(action_rollout[:actual_steps]),
        rewards=np.array(reward_rollout),
        terminals=np.array(done_rollout)
    )

    return f"Saved rollout {rollout_idx}, ({actual_steps} steps)"

def generate_data_parallel(rollouts, data_dir, noise_type, num_workers):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print(f"Starting parallel collection using {num_workers} workers.")

    tasks = [(i, data_dir, noise_type) for i in range(rollouts)]
    with multiprocessing.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(collect_single_rollout, tasks):
            print(res)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Total number of rollouts to collect")
    parser.add_argument('--dir', type=str, help="Directory to save .npz files")
    parser.add_argument('--policy', type=str, choices=['white', 'brownian'],
                        default='brownian', help="white | brownian")
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help="Number of parallel processes (default: all CPUs)")
    args = parser.parse_args()

    generate_data_parallel(args.rollouts, args.dir, args.policy, args.workers)