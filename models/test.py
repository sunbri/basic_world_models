import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np
import pygame

# MAPPING: Key -> Action [Steering, Gas, Brake]
# Steering: -1 (Left) to 1 (Right)
# Gas: 0 to 1
# Brake: 0 to 1

mapping = {
    # SINGLE KEYS
    (pygame.K_w,): np.array([0, 1, 0], dtype=np.float32),     # Gas
    (pygame.K_s,): np.array([0, 0, 1], dtype=np.float32),     # Brake
    (pygame.K_a,): np.array([-1, 0, 0], dtype=np.float32),    # Left
    (pygame.K_d,): np.array([1, 0, 0], dtype=np.float32),     # Right

    # COMBINATIONS (Crucial for driving properly!)
    (pygame.K_w, pygame.K_a): np.array([-1, 1, 0], dtype=np.float32),  # Gas + Left
    (pygame.K_w, pygame.K_d): np.array([1, 1, 0], dtype=np.float32),   # Gas + Right
    (pygame.K_s, pygame.K_a): np.array([-1, 0, 1], dtype=np.float32),  # Brake + Left
    (pygame.K_s, pygame.K_d): np.array([1, 0, 1], dtype=np.float32),   # Brake + Right
}

# We use "rgb_array" because the 'play' function renders the window itself
env = gym.make("CarRacing-v3", render_mode="rgb_array")

print("Controls:")
print("W: Gas")
print("S: Brake")
print("A: Left")
print("D: Right")
print("Press ESC to quit.")

# zoom=3 makes the window bigger so you can actually see
play(env, keys_to_action=mapping, zoom=3, fps=30, noop=np.array([0, 0, 0],
                                                                dtype=np.float32))
