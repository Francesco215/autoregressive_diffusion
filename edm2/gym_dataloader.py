import torch
from torch import Tensor
from torch.utils.data import IterableDataset

import cv2
import einops
import gymnasium as gym


class GymDataGenerator(IterableDataset):
    def __init__(self, state_size=6, environment_name="CartPole-v1", training_examples=10_000):
        self.state_size = state_size

        self.env = gym.make(environment_name,render_mode="rgb_array")

        self.evolution_time = 5
        self.training_examples = training_examples

    def __iter__(self):
        terminated = True

        for _ in range(self.training_examples):
            if terminated:
                observation, _ = self.env.reset()
                terminated = False
                reward = 0
                action = 0
                frame_history = []
                # action_history = []
                step_count = -self.evolution_time
            else:
                action = self.env.action_space.sample()  # Random action
                _ , reward, terminated, _, _ = self.env.step(action)
            
            # action_history.append(action)
            # action_history = action_history[-self.state_size:]

            if step_count >= 0: # This if can be removed, but having it avoids rendering useless frames
                frame = self.env.render()
                frame = resize_image(frame)
                # frame_image = Image.fromarray(frame)
                frame_history.append(torch.tensor(frame))
                frame_history = frame_history[-self.state_size:]
            
            if step_count > 0 and step_count%self.state_size==0:  # Skip the first step as we don't have a previous state
                assert len(frame_history)>=self.state_size
                frames = torch.stack(frame_history[-self.state_size:])

                actions = torch.tensor(action).clone()
                # if step_count%self.evolution_time==self.evolution_time-1:

                yield frames, actions, torch.tensor(reward).clone()
                frame_history = []
                
            step_count += 1


def resize_image(image_array):
    # Check if the input array has the correct shape
    if image_array.shape != (400, 600, 3):
        raise ValueError("Input array must have shape (400, 600, 3)")
    
    # Resize the image using OpenCV
    resized_image = cv2.resize(image_array, (256, 256), interpolation=cv2.INTER_AREA)
    
    return resized_image 
def gym_collate_function(batch):
    frame_histories, action_histories, rewards = zip(*batch)
    padded_frames = torch.stack(frame_histories)
    padded_actions = torch.stack(action_histories)
    return padded_frames, padded_actions, torch.Tensor(rewards)

    
@torch.no_grad()
def frames_to_latents(autoencoder, frames)->Tensor:
    """
    frames.shape: (batch_size, time, height, width, rgb)
    latents.shape: (batch_size, time, latent_channels, latent_height, latent_width)
    """
    batch_size = frames.shape[0]

    frames = frames / 127.5 - 1  # Normalize from (0,255) to (-1,1)
    frames = einops.rearrange(frames, 'b t h w c -> (b t) c h w')

    #split the conversion to not overload the GPU RAM
    split_size = 64
    for i in range (0, frames.shape[0], split_size):
        l = autoencoder.encode(frames[i:i+split_size]).latent_dist.sample()
        if i == 0:
            latents = l
        else:
            latents = torch.cat((latents, l), dim=0)

    # Apply scaling factor
    latents = latents * autoencoder.config.scaling_factor

    latents = einops.rearrange(latents, '(b t) c h w -> b t c h w', b=batch_size)
    return latents


@torch.no_grad()        
def latents_to_frames(autoencoder,latents):
    """
        Converts latent representations to frames.
        Args:
            latents (torch.Tensor): A tensor of shape (batch_size, time, latent_channels, latent_height, latent_width) 
                                    representing the latent representations.
        Returns:
            numpy.ndarray: A numpy array of shape (batch_size, height, width * time, rgb) representing the decoded frames.
        Note:
            - The method uses an autoencoder to decode the latent representations.
            - The frames are rearranged and clipped to the range [0, 255] before being converted to a numpy array.
    """
    batch_size = latents.shape[0]
    latents = einops.rearrange(latents, 'b t c h w -> (b t) c h w')
    # Apply inverse scaling factor
    latents = latents / autoencoder.config.scaling_factor

    #split the conversion to not overload the GPU RAM
    split_size = 64
    for i in range (0, latents.shape[0], split_size):
        l = autoencoder.decode(latents[i:i+split_size]).sample
        if i == 0:
            frames = l
        else:
            frames = torch.cat((frames, l), dim=0)

    frames = einops.rearrange(frames, '(b t) c h w -> b t h w c', b=batch_size) 
    frames = torch.clip((frames + 1) * 127.5, 0, 255).cpu().detach().numpy().astype(int)
    return frames

        