import cv2
import gymnasium as gym
import numpy as np

class ResizeObservationWrapper(gym.ObservationWrapper):
    """
    Removes bottom dashboard and downsamples to 64x64 for CarRacing-v3
    """
    def __init__(self, env, shape=(64, 64)):
        super().__init__(env)
        self.shape = shape

        self.observation_space = gym.spaces.Box(
            # save ints as less space on disk
            # dataloader will convert to float
            low=0, high=255,
            shape=(shape[0], shape[1], 3),
            dtype=np.uint8
        )

    def observation(self, observation):
        obs_cropped = observation[:84, :, :]
        return cv2.resize(obs_cropped, self.shape, interpolation=cv2.INTER_AREA)