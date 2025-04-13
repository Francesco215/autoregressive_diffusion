import pygame
import torch
import numpy as np
import einops
import copy
import time
import gymnasium as gym
import cv2
from edm2.gym_dataloader import frames_to_latents, latents_to_frames, resize_image
from edm2.sampler import edm_sampler_with_mse
from flask import Flask, Response
import threading
import cv2
import numpy as np
import os


class GymInteractiveDemo:
    def __init__(self, autoencoder, precond, env_name="LunarLander-v3", device='cuda', min_frames=8):
        self.autoencoder = autoencoder
        self.precond = precond
        self.device = device
        self.env_name = env_name
        self.min_frames = min_frames  # Minimum frames needed for VAE
        
        # Initialize action tracking
        self.last_action = 0  # Set default action to 0 (NoOp) before using it
        
        # Initialize models
        self.autoencoder.eval()
        self.precond.eval()
        
        # Initialize the environment
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.state = self.env.reset()[0]
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((768, 512))
        pygame.display.set_caption(f"Interactive {env_name} with Diffusion")
        self.clock = pygame.time.Clock()
        
        # Mapping from pygame keys to gym actions
        self.action_map = {
            pygame.K_LEFT: 1,   # Left engine
            pygame.K_RIGHT: 2,  # Right engine
            pygame.K_UP: 3,     # Main engine
            pygame.K_SPACE: 0   # Do nothing
        }
        
        # Initialize frame buffers
        self.real_frames = []
        self.generated_frames = []
        
        # Collect initial frames
        print("Collecting initial frames...")
        for _ in range(self.min_frames):
            self.step_environment(0)  # No-op to collect frames
        
        # Prepare initial context and generate first frame
        self.prepare_initial_context()
        
        self.running = True
        self.auto_play = False
        
    def capture_real_frame(self):
        """Capture a real frame from the environment"""
        frame = self.env.render()
        frame = resize_image(frame)
        self.real_frames.append(frame)
        if len(self.real_frames) > self.min_frames + 8:  # Keep only recent frames
            self.real_frames.pop(0)
    
    def prepare_initial_context(self):
        """Convert initial real frames to latent space for model input"""
        if len(self.real_frames) >= self.min_frames:
            # Convert frames to tensor format
            frames = np.stack(self.real_frames[-self.min_frames:])
            frames = np.expand_dims(frames, 0)  # Add batch dimension
            frames_tensor = torch.tensor(frames, device=self.device)
            
            # Print shape for debugging
            print(f"Initial frames shape: {frames_tensor.shape}")
            
            try:
                # Convert to latents
                self.latents = frames_to_latents(self.autoencoder, frames_tensor)
                print(f"Initial latents shape: {self.latents.shape}")
                
                # Initialize cache with sigma (noise level)
                sigma = torch.ones(self.latents.shape[:2], device=self.device) * 0.05
                _, self.cache = self.precond(self.latents, sigma)
                
                # Generate the first frame
                self.generate_next_frame(0)  # Use 0 (NoOp) for first generation
            except Exception as e:
                print(f"Error during initialization: {e}")
                import traceback
                traceback.print_exc()
    
    def generate_next_frame(self, action_idx):
        """Generate the next frame using the diffusion model"""
        try:
            # Convert action to tensor format
            action = torch.tensor([[action_idx]], device=self.device)
            
            # Make a deep copy of the cache
            cache_copy = copy.deepcopy(self.cache)
            
            # Print shapes for debugging
            print(f"Input latents shape: {self.latents.shape}")
            print(f"Action shape: {action.shape}")
            
            # Run diffusion sampling
            with torch.no_grad():
                start_time = time.time()
                x, _, _, new_cache = edm_sampler_with_mse(
                    self.precond, 
                    cache=cache_copy, 
                    conditioning=action,
                    sigma_max=80, 
                    num_steps=32,
                    rho=7, 
                    guidance=1, 
                    S_churn=20
                )
                print(f"Sampling took {time.time() - start_time:.2f} seconds")
                print(f"Generated frame shape: {x.shape}")
            
            # Update cache and latents
            self.cache = new_cache
            self.latents = torch.cat((self.latents, x), dim=1)
            
            # If latents get too long, remove oldest frames while keeping at least min_frames
            if self.latents.shape[1] > self.min_frames + 8:
                self.latents = self.latents[:, -(self.min_frames + 8):]
            
            # Convert generated latents to frames - use at least 4 frames for decoding
            # Take multiple frames to satisfy the VAE's temporal convolution requirements
            frames_to_decode = self.latents[:, -4:] if self.latents.shape[1] >= 4 else self.latents
            frames = latents_to_frames(self.autoencoder, frames_to_decode)

            # Only take the last frame (the newly generated one)
            last_frame = frames[0, -1]
            print(f"Decoded frame shape: {last_frame.shape}, type: {type(last_frame)}")
            # Ensure the frame is a numpy array
            if isinstance(last_frame, torch.Tensor):
                last_frame = last_frame.cpu().numpy()
            self.generated_frames.append(last_frame)
            
            # Limit number of saved generated frames
            if len(self.generated_frames) > 8:
                self.generated_frames.pop(0)
                
            return True
        
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step_environment(self, action):
        """Take a step in the real environment"""
        self.state, reward, terminated, truncated, _ = self.env.step(action)
        self.capture_real_frame()
        
        if terminated or truncated:
            self.state = self.env.reset()[0]
            self.capture_real_frame()
    
    def render(self):
        """Render the current state to the pygame window"""
        self.screen.fill((0, 0, 0))
        
        # Render real frames on left side
        if len(self.real_frames) > 0:
            for i in range(min(8, len(self.real_frames))):
                frame = self.real_frames[-(i+1)]  # Display most recent frames at the top
                y_pos = i * 64
                # Ensure frame is a valid numpy array for resizing
                if isinstance(frame, np.ndarray) and frame.size > 0:
                    # Resize frame to fit display
                    resized = cv2.resize(frame, (256, 64))
                    # Convert to pygame surface
                    surface = pygame.surfarray.make_surface(resized.transpose(1, 0, 2))
                    self.screen.blit(surface, (0, y_pos))
        
        # Render generated frames on right side
        if len(self.generated_frames) > 0:
            for i in range(min(8, len(self.generated_frames))):
                frame = self.generated_frames[-(i+1)]  # Display most recent frames at the top
                y_pos = i * 64
                # Ensure frame is a valid numpy array for resizing
                if isinstance(frame, np.ndarray) and frame.size > 0:
                    # Debug info
                    print(f"Generated frame shape: {frame.shape}, dtype: {frame.dtype}")
                    # Ensure frame is a valid format for OpenCV
                    frame = frame.astype(np.uint8)
                    # Resize frame to fit display
                    resized = cv2.resize(frame, (256, 64))
                    # Convert to pygame surface
                    surface = pygame.surfarray.make_surface(resized.transpose(1, 0, 2))
                    self.screen.blit(surface, (512, y_pos))
        
        # Display instructions and info
        font = pygame.font.SysFont('Arial', 20)
        text = font.render('Left/Right/Up/Space: Control lander', True, (255, 255, 255))
        self.screen.blit(text, (200, 512 - 60))
        
        text = font.render('A: Toggle auto-play, ESC: Quit', True, (255, 255, 255))
        self.screen.blit(text, (200, 512 - 30))
        
        text = font.render('Real Frames', True, (255, 255, 255))
        self.screen.blit(text, (0, 0))
        
        text = font.render('Generated Frames', True, (255, 255, 255))
        self.screen.blit(text, (512, 0))
        
        action_names = ["NoOp", "Left", "Right", "Main"]
        text = font.render(f'Last Action: {action_names[self.last_action]}', True, (255, 255, 255))
        self.screen.blit(text, (300, 512 - 90))
    
    def run(self):
        while self.running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.action_map:
                        self.last_action = self.action_map[event.key]
                        self.step_environment(self.last_action)
                        self.generate_next_frame(self.last_action)
                    elif event.key == pygame.K_a:
                        self.auto_play = not self.auto_play
                        print(f"Auto-play: {'ON' if self.auto_play else 'OFF'}")
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Auto-play mode: randomly select actions
            if self.auto_play:
                if np.random.random() < 0.3:  # 30% chance to change action
                    self.last_action = np.random.randint(0, 4)
                
                self.step_environment(self.last_action)
                self.generate_next_frame(self.last_action)
            
            # Render and update display
            self.render()
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        pygame.quit()
        self.env.close()

# Usage example:
from edm2.vae import VAE
from edm2.networks_edm2 import UNet, Precond
from edm2.sampler import edm_sampler_with_mse
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = VAE.from_pretrained("saved_models/vae_10000.pt").to(device)
unet = UNet.from_pretrained('saved_models/unet_14.0M_topk1.pt').to(device)
precond = Precond(unet, use_fp16=True, sigma_data=1.0).to(device)

demo = GymInteractiveDemo(autoencoder, precond, device=device, min_frames=8)
demo.run()