from flask import Flask, Response
import threading
import cv2
import numpy as np
import torch
import time
import copy
import gymnasium as gym
from edm2.vae import VAE
from edm2.networks_edm2 import UNet, Precond
from edm2.sampler import edm_sampler_with_mse
from edm2.gym_dataloader import frames_to_latents, latents_to_frames, resize_image

# Initialize Flask app
app = Flask(__name__)
demo = None  # Will be set in main()

# Define the model class
class GymInteractiveDemo:
    def __init__(self, autoencoder, precond, env_name="LunarLander-v3", device='cuda', min_frames=8, initial_frames=20):
        self.autoencoder = autoencoder
        self.precond = precond
        self.device = device
        self.env_name = env_name
        self.min_frames = min_frames  # Minimum frames needed for VAE
        
        # Initialize action tracking
        self.last_action = 0
        
        # Initialize models
        self.autoencoder.eval()
        self.precond.eval()
        
        # Initialize the environment
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.state = self.env.reset()[0]
        
        # Initialize frame buffers
        self.real_frames = []
        self.generated_frames = []
        
        # Collect initial frames - use initial_frames instead of min_frames
        print(f"Collecting {initial_frames} initial frames...")
        for _ in range(initial_frames):
            self.step_environment(0)  # No-op to collect frames
        
        # Prepare initial context and generate first frame
        self.prepare_initial_context()
        
        # Close the real environment to free up resources
        print("Closing real environment...")
        self.env.close()
        self.env = None
        
        self.auto_play = True
        
    def capture_real_frame(self):
        """Capture a real frame from the environment"""
        frame = self.env.render()
        frame = resize_image(frame)
        self.real_frames.append(frame)
        if len(self.real_frames) > self.min_frames + 8:
            self.real_frames.pop(0)
    
    def prepare_initial_context(self):
        """Convert initial real frames to latent space for model input"""
        if len(self.real_frames) >= self.min_frames:
            # Convert frames to tensor format
            frames = np.stack(self.real_frames[-self.min_frames:])
            frames = np.expand_dims(frames, 0)  # Add batch dimension
            frames_tensor = torch.tensor(frames, device=self.device)
            
            print(f"Initial frames shape: {frames_tensor.shape}")
            
            try:
                # Convert to latents
                self.latents = frames_to_latents(self.autoencoder, frames_tensor)
                print(f"Initial latents shape: {self.latents.shape}")
                
                # Initialize cache with sigma (noise level)
                sigma = torch.ones(self.latents.shape[:2], device=self.device) * 0.05
                _, self.cache = self.precond(self.latents, sigma)
                
                # Generate the first frame
                self.generate_next_frame(0)
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
            
            print(f"Input latents shape: {self.latents.shape}")
            
            # Run diffusion sampling
            with torch.no_grad():
                start_time = time.time()
                x, _, _, new_cache = edm_sampler_with_mse(
                    self.precond, 
                    cache=cache_copy, 
                    conditioning=action,
                    sigma_max=80, 
                    num_steps=10,
                    rho=7, 
                    guidance=1, 
                    S_churn=20
                )
                print(f"Sampling took {time.time() - start_time:.2f} seconds")
            
            # Update cache and latents
            self.cache = new_cache
            self.latents = torch.cat((self.latents, x), dim=1)
            
            # If latents get too long, remove oldest frames
            if self.latents.shape[1] > self.min_frames + 8:
                self.latents = self.latents[:, -(self.min_frames + 8):]
            
            # Convert generated latents to frames - use at least 4 frames for decoding
            frames_to_decode = self.latents[:, -4:] if self.latents.shape[1] >= 4 else self.latents
            frames = latents_to_frames(self.autoencoder, frames_to_decode)
            
            # Only take the last frame
            last_frame = frames[0, -1].astype(np.uint8)
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

# Flask routes
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Model Stream</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }
            h1 { color: #333; }
            .container { display: flex; justify-content: space-around; margin-top: 20px; }
            .frame-container { border: 1px solid #ddd; padding: 10px; }
            .controls { margin-top: 20px; }
            button { padding: 10px 20px; margin: 0 10px; cursor: pointer; }
        </style>
        <script>
            function setAction(action) {
                fetch('/set_action/' + action)
                    .then(response => response.text())
                    .then(data => console.log(data));
            }
            
            function toggleAutoplay() {
                fetch('/toggle_autoplay')
                    .then(response => response.text())
                    .then(data => {
                        document.getElementById('autoplay-status').innerText = data;
                    });
            }
            
            // Monitor the stream and reload if it fails
            function monitorStream() {
                const img = document.getElementById('stream');
                img.onerror = function() {
                    console.log('Stream error detected, reloading...');
                    img.src = '/video_feed?t=' + new Date().getTime();
                };
            }
            
            window.onload = function() {
                monitorStream();
            };
        </script>
    </head>
    <body>
        <h1>Interactive LunarLander with Video Diffusion</h1>
        <div class="container">
            <img id="stream" src="/video_feed" width="800" />
        </div>
        <div class="controls">
            <button onclick="setAction(0)">No-op</button>
            <button onclick="setAction(1)">Left Engine</button>
            <button onclick="setAction(2)">Right Engine</button>
            <button onclick="setAction(3)">Main Engine</button>
            <button onclick="toggleAutoplay()">Toggle Autoplay</button>
            <p>Autoplay: <span id="autoplay-status">ON</span></p>
        </div>
    </body>
    </html>
    """
# Modify the generate_frames function to only show generated frames
def generate_frames():
    last_timestamp = 0
    
    while True:
        current_time = time.time()
        
        if demo and len(demo.generated_frames) > 0:
            try:
                # Just use the generated frame at full size
                gen_frame = cv2.resize(demo.generated_frames[-1], (768, 512))
                
                # Add minimal labels with smaller font
                action_names = ["NoOp", "Left", "Right", "Main"]
                cv2.putText(gen_frame, f"Action: {action_names[demo.last_action]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Convert to jpeg
                _, buffer = cv2.imencode('.jpg', gen_frame)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                last_timestamp = current_time
            except Exception as e:
                print(f"Error in generate_frames: {e}")
        else:
            # If no frames yet, send a blank frame
            if current_time - last_timestamp > 1.0:
                blank_frame = np.zeros((512, 768, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Initializing diffusion model...", (250, 256), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                
                _, buffer = cv2.imencode('.jpg', blank_frame)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                last_timestamp = current_time
        
        # Don't send frames too quickly
        time.sleep(0.1)

# Also modify the run_auto_play function
def run_auto_play():
    """Runs the model in auto-play mode in a separate thread"""
    while True:
        if demo and demo.auto_play:
            try:
                if np.random.random() < 0.3:  # 30% chance to change action
                    demo.last_action = np.random.randint(0, 4)
                
                # Don't step the real environment, just update the diffusion model
                demo.generate_next_frame(demo.last_action)
            except Exception as e:
                print(f"Error in auto_play: {e}")
        
        time.sleep(0.5)  # Run auto-play at 2 FPS

# And modify the set_action route
@app.route('/set_action/<int:action>')
def set_action(action):
    if demo:
        demo.last_action = action
        # Don't step the real environment
        demo.generate_next_frame(action)
        return f"Action set to {action}"
    return "Demo not initialized"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/toggle_autoplay')
def toggle_autoplay():
    if demo:
        demo.auto_play = not demo.auto_play
        return "ON" if demo.auto_play else "OFF"
    return "Demo not initialized"


def main():
    global demo
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    autoencoder = VAE.from_pretrained("saved_models/vae_10000.pt").to(device)
    unet = UNet.from_pretrained('saved_models/unet_14.0M_topk1.pt').to(device)
    precond = Precond(unet, use_fp16=True, sigma_data=1.0).to(device)
    
    # Create demo instance
    demo = GymInteractiveDemo(autoencoder, precond, device=device, min_frames=8)
    
    # Start auto-play thread
    auto_play_thread = threading.Thread(target=run_auto_play)
    auto_play_thread.daemon = True
    auto_play_thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == "__main__":
    main()