#%%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Box2D.b2 import vec2 # Box2D vector type for position

# Try importing the specific environment class if needed for type hints/checks
# try:
#     from your_module import LunarLander # Adjust 'your_module' as needed
# except ImportError:
#     LunarLander = type(gym.make("LunarLander-v3").unwrapped) # Fallback

# Assume constants are defined if needed outside the env instance
# Make sure these are accessible or defined if running standalone.
# VIEWPORT_W = 600
# VIEWPORT_H = 400
# SCALE = 30.0
# LEG_DOWN = 18 # Needed for state[1] calculation

def show_lander_at_position(env, pos_x: float, pos_y: float):
    """
    Manually sets the Lunar Lander's position, renders the environment,
    calculates the corresponding state[0] and state[1], and displays the
    image with the state values in the title.

    Args:
        env: An initialized LunarLander-v3 environment instance, created with
             render_mode="rgb_array". Must have called env.reset() at least once.
        pos_x: The desired X coordinate (Box2D world units, e.g., 0 to 20).
        pos_y: The desired Y coordinate (Box2D world units, e.g., 0 to 13.33).

    Warning:
        Directly manipulating the physics state can lead to unexpected results
        if the simulation is continued via env.step() afterwards. Intended for snapshots.
    """
    print(f"Attempting to show lander at world pos ({pos_x:.2f}, {pos_y:.2f})...")

    # --- Basic Input Validation ---
    # (Keep validation as before)
    if not hasattr(env, 'unwrapped'):
         print("Error: Invalid environment.")
         return
    if env.render_mode != "rgb_array":
        print("Error: Environment must be created with render_mode='rgb_array'.")
        return
    if not hasattr(env.unwrapped, 'lander') or env.unwrapped.lander is None:
        print("Error: Lander body not found. Has env.reset() been called?")
        return

    try:
        lander = env.unwrapped.lander
        new_pos = vec2(float(pos_x), float(pos_y))

        # --- Manually Set State ---
        print("  Setting position and resetting dynamics...")
        lander.position = new_pos
        lander.linearVelocity = vec2(0, 0)
        lander.angularVelocity = 0.0
        lander.angle = 0.0
        lander.awake = True

        # --- Calculate Corresponding State Values ---
        state_0, state_1 = None, None # Initialize
        try:
            # Ensure constants are accessible
            if 'VIEWPORT_W' not in globals() or 'VIEWPORT_H' not in globals() or \
               'SCALE' not in globals() or 'LEG_DOWN' not in globals():
                 # Attempt to get from env spec if possible (less reliable)
                 # Or raise error/use defaults
                 raise NameError("Required constants (VIEWPORT_W/H, SCALE, LEG_DOWN) not found.")

            world_w = VIEWPORT_W / SCALE
            world_h = VIEWPORT_H / SCALE
            # helipad_y calculation depends on how env was reset, but usually H/4
            # Safer to get it from env if possible, otherwise assume default.
            helipad_y = getattr(env.unwrapped, 'helipad_y', world_h / 4.0)

            # Calculate state[0] and state[1] using the logic from env.step()
            state_0 = (pos_x - world_w / 2) / (world_w / 2)
            # state[1] = (pos_y - (helipad_y + LEG_DOWN / SCALE)) / (world_h / 2) # Corrected y normalization
            # Use VIEWPORT_H instead of world_h in denominator as per original code state[1] calculation:
            state_1 = (pos_y - (helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)


            print(f"  Calculated State Approx: state[0]={state_0:.3f}, state[1]={state_1:.3f}")

        except NameError as e:
            print(f"Warning: Could not calculate state values due to missing constants: {e}")
        except Exception as e:
            print(f"Warning: Error calculating state values: {e}")

        # --- Render ---
        print("  Rendering...")
        image_array = env.render()

        if image_array is None or not isinstance(image_array, np.ndarray):
            print(f"Error: env.render() did not return a valid NumPy array.")
            return

        # --- Display Image ---
        print("  Displaying image...")
        plt.figure(figsize=(8, 6))
        plt.imshow(image_array)

        # *** Use calculated state values in the title ***
        if state_0 is not None and state_1 is not None:
            plt.title(f"Lander at state[0] ≈ {state_0:.2f}, state[1] ≈ {state_1:.2f}")
        else:
            # Fallback title if state calculation failed
            plt.title(f"Lander set to ({pos_x:.2f}, {pos_y:.2f}) [World Coords]")

        plt.xlabel("X Pixels")
        plt.ylabel("Y Pixels")
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    print("Running example usage...")
    try:
        # Define constants if not globally available for the example
        if 'VIEWPORT_W' not in globals():
            VIEWPORT_W = 600
            VIEWPORT_H = 400
            SCALE = 30.0
            LEG_DOWN = 18 # *** Added LEG_DOWN ***
            # from lunar_lander_code import LunarLander # Example import

        # Create the environment with the necessary render mode
        env_display = gym.make("LunarLander-v3", render_mode="rgb_array")
        print("Environment created.")

        # CRITICAL: Reset the environment once to create the lander body
        print("Resetting environment...")
        env_display.reset(seed=123)
        print("Environment reset.")

        world_w = VIEWPORT_W / SCALE
        world_h = VIEWPORT_H / SCALE
        print(f"World Dimensions (approx): W={world_w:.2f}, H={world_h:.2f}")

        # Show lander near the center-bottom
        show_lander_at_position(env_display, world_w / 2, world_h * 0.2)

        # Show lander near the top-left
        show_lander_at_position(env_display, world_w * 0.1, world_h * 0.9)

        # Show lander outside the typical bounds (top right)
        # Note: State calculation might be less meaningful far outside bounds
        show_lander_at_position(env_display, world_w * 1.1, world_h * 1.1)

        # Show lander at origin (0,0) - likely below ground visually
        show_lander_at_position(env_display, 0, 0)

        print("Closing environment...")
        env_display.close()
        print("Example usage finished.")

    except ImportError as e:
         print(f"\nError: Missing dependencies. Please install requirements.")
         print(f" pip install gymnasium[box2d] matplotlib")
         print(f" Details: {e}")
    except NameError as e:
         print(f"\nError: Required name not found (maybe LunarLander class or constants like LEG_DOWN?).")
         print(f" Details: {e}")
    except Exception as e:
         print(f"\nAn unexpected error occurred during example: {e}")
# %%
