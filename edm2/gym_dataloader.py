import torch
from torch import Tensor
from torch.utils.data import IterableDataset

import cv2
import einops
import gymnasium as gym



class GymDataGenerator(IterableDataset):
    def __init__(self, state_size=6, environment_name="LunarLander-v3", training_examples=10_000, autoencoder_time_compression=4, return_anyways=True):
        self.state_size = state_size
        self.environment_name = environment_name
        self.evolution_time = 10
        self.terminate_size = 512
        self.training_examples = training_examples
        self.autoencoder_time_compression = autoencoder_time_compression
        self.frame_collection_interval = 2
        self.return_anyways=return_anyways

    def is_lander_in_frame(self, state):
        """Check if the lander is within the visible frame based on its state."""
        x, y = state[0], state[1]
        return abs(y) < 1 and abs(x) < .95

    @torch.no_grad()
    def __iter__(self):
        env = gym.make(self.environment_name, render_mode="rgb_array")
        terminated = True
        n_data_yielded = 0

        while n_data_yielded < self.training_examples:
            if terminated:
                env.reset()
                terminated = False
                reward = 0
                action = 0
                frame_history = []
                state_history = []  # Added to track states
                action_history = []
                step_count = -self.evolution_time
            else:
                if step_count % (self.autoencoder_time_compression * self.frame_collection_interval) == 0:
                    action = env.action_space.sample()  # Random action
                    action_history.append(action)
                # Capture the state along with reward and termination
                state, reward, terminated, _, _ = env.step(action)
            
            if step_count >= 0 and step_count % self.frame_collection_interval == 0:
                frame = env.render()
                frame = resize_image(frame)
                frame_history.append(torch.tensor(frame))
                state_history.append(state)  # Store the state for this frame
            
            if step_count > 0 and step_count % (self.state_size * self.frame_collection_interval)== 0:
                # Check if the lander is in the frame for all states in the sequence
                if self.return_anyways or all(self.is_lander_in_frame(s) for s in state_history[-self.state_size:]):
                    frames = torch.stack(frame_history[-self.state_size:])
                    actions = torch.tensor(action_history[-self.state_size // self.autoencoder_time_compression:])
                    yield frames, actions, torch.tensor(reward).clone()
                    n_data_yielded += 1
                # Reset histories whether we yield or not to maintain sequence alignment
                frame_history, state_history, action_history = [], [], []
            
            if step_count > self.terminate_size:
                terminated = True
                
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

# saved_mean = torch.load('mean_LunarLander-v3_latent.pt').mean(dim=(1,2))
# std_latent = 12446.0
@torch.no_grad()
def frames_to_latents(autoencoder, frames)->Tensor:
    """
    frames.shape: (batch_size, time, height, width, rgb)
    latents.shape: (batch_size, time, latent_channels, latent_height, latent_width)
    """
    batch_size = frames.shape[0]

    frames = frames / 127.5 - 1  # Normalize from (0,255) to (-1,1)
    frames = einops.rearrange(frames, 'b t h w c -> b c t h w')

    #split the conversion to not overload the GPU RAM
    split_size = 64
    for i in range (0, frames.shape[0], split_size):
        # l = autoencoder.encode(frames[i:i+split_size]).latent_dist.sample()
        _, l, _, _ = autoencoder.encode(frames[i:i+split_size].to(autoencoder.device))
        if i == 0:
            latents = l
        else:
            latents = torch.cat((latents, l), dim=0)

    # Apply scaling factor
    # latents = latents * autoencoder.config.scaling_factor
    # mean,std = torch.tensor(autoencoder.config.latents_mean)[:,None, None, None].to(latents), torch.tensor(autoencoder.config.latents_std)[:, None, None, None].to(latents)
    # mean = mean + saved_mean.view(mean.shape).to(latents) 
    # latents = (latents - mean)/std

    latents = einops.rearrange(latents, 'b c t h w -> b t c h w', b=batch_size)
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
    latents = einops.rearrange(latents, 'b t c h w -> b c t h w')

    latents = latents * 1.2

    #split the conversion to not overload the GPU RAM
    split_size = 16
    for i in range (0, latents.shape[0], split_size):
        l, _ = autoencoder.decode(latents[i:i+split_size])
        if i == 0:
            frames = l
        else:
            frames = torch.cat((frames, l), dim=0)

    frames = einops.rearrange(frames, 'b c t h w -> b t h w c', b=batch_size) 
    frames = torch.clip((frames + 1) * 127.5, 0, 255).cpu().detach().numpy().astype(int)
    return frames

        