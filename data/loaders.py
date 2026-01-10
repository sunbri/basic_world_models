import glob
import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

# for training the VAE
class BufferedRolloutDataset(IterableDataset):
    def __init__(self, source, file_buffer_size=20, mode='rl', transform=None):
        """
        Args:
            source (str or list): 
                - If str: Path to directory (loads all .npz files inside).
                - If list: List of specific file paths (good for train/test splits).
            file_buffer_size (int): Number of files to load into RAM at once.
            mode (str): 'ae' (observations only) or 'rl' (obs, next_obs, actions, rewards, terminals).
            transform (callable, optional): Transform to apply to observations (e.g. Normalization).
        """
        super().__init__()
        
        # 1. Flexible Input Handling
        if isinstance(source, str):
            # User passed a directory string
            self.file_paths = sorted(glob.glob(os.path.join(source, "*.npz")))
            if len(self.file_paths) == 0:
                raise ValueError(f"No .npz files found in directory: {source}")
        elif isinstance(source, list):
            # User passed a specific list of files
            self.file_paths = source
        else:
            raise TypeError("Source must be a directory path (str) or list of files.")

        self.file_buffer_size = file_buffer_size
        self.mode = mode
        self.transform = transform
        
        assert mode in ['ae', 'rl'], "Mode must be 'ae' or 'rl'"

    def _load_data_from_files(self, files):
        buffer_data = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'rewards': [],
            'terminals': []
        }

        for file_path in files:
            try:
                with np.load(file_path) as data:
                    raw_obs = data['observations'] # Shape (T+1, ...)
                    
                    if self.mode == 'ae':
                        # AE Mode: Load everything including the extra frame
                        buffer_data['observations'].append(raw_obs)
                        
                    elif self.mode == 'rl':
                        # RL Mode: Align Obs(t) with Act(t)
                        raw_actions = data['actions']
                        T = len(raw_actions)

                        # Obs = 0 to T-1
                        buffer_data['observations'].append(raw_obs[:T])
                        # Next Obs = 1 to T
                        buffer_data['next_observations'].append(raw_obs[1:T+1])
                        
                        buffer_data['actions'].append(raw_actions)
                        buffer_data['rewards'].append(data['rewards'])
                        buffer_data['terminals'].append(data['terminals'])
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        # Concatenate whatever was loaded
        final_buffer = {}
        keys = ['observations'] if self.mode == 'ae' else ['observations', 'next_observations', 'actions', 'rewards', 'terminals']

        for key in keys:
            if buffer_data[key]:
                final_buffer[key] = np.concatenate(buffer_data[key], axis=0)

        return final_buffer

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Worker Split Logic
        if worker_info is None:
            my_files = self.file_paths
        else:
            per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths))
            my_files = self.file_paths[iter_start:iter_end]

        # Global Shuffle (Files)
        np.random.shuffle(my_files)

        # Chunk Loop
        for i in range(0, len(my_files), self.file_buffer_size):
            file_chunk = my_files[i : i + self.file_buffer_size]
            buffer = self._load_data_from_files(file_chunk)
            
            if 'observations' not in buffer:
                continue

            n_samples = len(buffer['observations'])
            indices = np.random.permutation(n_samples) # Local Shuffle (Buffer)

            for idx in indices:
                # Observations start as uint8
                obs = buffer['observations'][idx]
                
                # Apply Transform (uint8 -> float / normalize / augment)
                if self.transform:
                    obs = self.transform(obs)
                else:
                    obs = torch.from_numpy(obs).float()

                if self.mode == 'ae':
                    yield obs
                else:
                    # Apply transform to next_obs as well
                    next_obs = buffer['next_observations'][idx]
                    if self.transform:
                        next_obs = self.transform(next_obs)
                    else:
                        next_obs = torch.from_numpy(next_obs).float()

                    action = torch.from_numpy(buffer['actions'][idx]).float()
                    reward = torch.tensor(buffer['rewards'][idx]).float()
                    terminal = torch.tensor(buffer['terminals'][idx]).float()
                    
                    yield {
                        'observation': obs,
                        'next_observation': next_obs,
                        'action': action, 
                        'reward': reward, 
                        'terminal': terminal
                    }

# for training MDN-RNN, the latent vectors are small
# load it all into RAM
class LatentSeqDataset(Dataset):
    def __init__(self, data_dir, seq_len=32, mode='train', split_ratio=0.9, epoch_repeat=50):
        """
        :param epoch_repeat: how many times to sample from each file
        """
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.epoch_repeat = epoch_repeat

        all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        all_files.sort()

        split_idx = int(len(all_files) * split_ratio)
        if mode == 'train':
            self.files = all_files[:split_idx]
        elif mode == 'test':
            self.files = all_files[split_idx:]
            self.epoch_repeat = 10 # scan fewer times for test

        # Load all data into RAM
        self.data_cache = []
        print(f"Pre-loading {mode} dataset into RAM...")

        for f in tqdm(self.files):
            path = os.path.join(data_dir, f)
            with np.load(path) as data:
                self.data_cache.append({
                    'mu': np.copy(data['mu']),
                    'action': np.copy(data['action'])
                })

    # this function determines the size of an epoch
    def __len__(self):
        return len(self.files) * self.epoch_repeat

    def __getitem__(self, index):
        file_idx = index % len(self.files)
        data = self.data_cache[file_idx]
        mu = data['mu']
        action = data['action']
        max_start = len(action) - self.seq_len - 1

        if max_start < 0:
            raise ValueError(f"Episode at index {file_idx} is too short for seq_len {self.seq_len}")
        
        start_idx = np.random.randint(0, max_start + 1)
        end_idx = start_idx + self.seq_len

        z_input = mu[start_idx : end_idx]
        a_input = action[start_idx : end_idx]
        z_target = mu[start_idx+1 : end_idx+1]

        return {
            'z_input': torch.FloatTensor(z_input),
            'a_input': torch.FloatTensor(a_input),
            'z_target': torch.FloatTensor(z_target)
        }

# class LatentSeqDataset(Dataset):
#     def __init__(self, data_dir, seq_len=32):
#         super().__init__()
#         self.data_dir = data_dir
#         self.seq_len = seq_len
#         self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
#         self.files.sort()
# 
#     def __len__(self):
#         return len(self.files)
# 
#     def __getitem__(self, index):
#         # Load one episode
#         filepath = os.path.join(self.data_dir, self.files[index])
#         data = np.load(filepath)
# 
#         mu = data['mu'] # (1001, 32)
#         action = data['action'] # (1000, 3)
# 
#         start_idx = np.random.randint(0, len(action) - self.seq_len)
#         end_idx = start_idx + self.seq_len
# 
#         z_input = mu[start_idx : end_idx]
#         a_input = action[start_idx : end_idx]
#         z_target = mu[start_idx+1 : end_idx+1]
# 
#         return {
#             'z_input': torch.FloatTensor(z_input),
#             'a_input': torch.FloatTensor(a_input),
#             'z_target': torch.FloatTensor(z_target)
#         }