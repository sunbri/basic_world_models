import logging
import torch
import os
import sys

RESIZE_SIZE = 64
ACTION_SIZE = 3
LATENT_SIZE = 32
HIDDEN_SIZE = 256

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def setup_logger(save_dir, name="Training_Log"):
    """
    Sets up a logger that writes to text file and console
    """
    log_file = os.path.join(save_dir, 'log.txt')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger