import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

import cv2
import einops
import gymnasium as gym
import gc


class GymDataGenerator(IterableDataset):
    def __init__(self, state_size=32, environment_name="Acrobot-v1", training_examples=10_000, autoencoder_time_compression=4, return_anyways=True):
        self.state_size = state_size
        self.environment_name = environment_name
        self.evolution_time = 10
        self.terminate_size = 800  # Acrobot episodes can be quite long
        self.training_examples = training_examples
        self.autoencoder_time_compression = autoencoder_time_compression
        self.frame_collection_interval = 2
        self.return_anyways = return_anyways

        assert state_size % autoencoder_time_compression == 0

    def is_acrobot_in_frame(self, state):
        """Check if the acrobot is in a good configuration to capture.
        Acrobot state contains cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2"""
        # For Acrobot, we always return True as it's always in the frame
        # The rendering keeps it centered in the viewport
        return True

    def __iter__(self):
        env = gym.make(self.environment_name, render_mode="rgb_array")
        terminated = True
        n_data_yielded = 0

        while n_data_yielded < self.training_examples:
            if not terminated and step_count > 0 and step_count % (self.state_size * self.frame_collection_interval) == 0:
                # For Acrobot, it's always in frame so we just check if we want to return anyway
                if self.return_anyways or all(self.is_acrobot_in_frame(s) for s in state_history):
                    frames, actions = np.stack(frame_history), np.stack(action_history)
                    yield frames, actions, reward
                    n_data_yielded += 1
                # Reset histories whether we yield or not to maintain sequence alignment
                frame_history, state_history, action_history = [], [], []

            if terminated:
                state, _ = env.reset()
                terminated = False
                reward = 0
                action = 0
                frame_history, state_history, action_history = [], [], []
                step_count = -self.evolution_time
            else:
                if step_count % (self.autoencoder_time_compression * self.frame_collection_interval) == 0:
                    # Acrobot has 3 discrete actions (0, 1, 2)
                    action = env.action_space.sample()  # Random action
                    if step_count >= 0:
                        action_history.append(action)
                
                # Capture the state along with reward and termination
                state, reward, terminated, truncated, _ = env.step(action)
                
                if truncated:
                    terminated = True
            
            if step_count >= 0 and step_count % self.frame_collection_interval == 0:
                frame = env.render()
                frame = resize_image(frame)
                frame_history.append(np.array(frame))
                state_history.append(state)  # Store the state for this frame
            
            if step_count > self.terminate_size:
                terminated = True
                
            step_count += 1


def resize_image(image_array):
    # Acrobot's default rendering is 500x500
    # Resize to 256x256 for consistency with other environments
    resized_image = cv2.resize(image_array, (256, 256), interpolation=cv2.INTER_AREA)
    return resized_image


def gym_collate_function(batch):
    frame_histories, action_histories, rewards = zip(*batch)
    padded_frames = np.stack(frame_histories)
    padded_actions = np.stack(action_histories)
    return padded_frames, padded_actions, rewards