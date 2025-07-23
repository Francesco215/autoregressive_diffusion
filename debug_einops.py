# debug.py
import torch
import einops
import traceback
import functools
from torch import nn
torch._dynamo.config.recompile_limit = 64
# Global tracking variables
problematic_calls = []
call_stack_info = []

def trace_einops_calls():
    """Monkey patch einops to track all calls and identify problematic ones"""
    original_rearrange = einops.rearrange
    original_reduce = einops.reduce
    original_repeat = einops.repeat
    
    def traced_rearrange(*args, **kwargs):
        import inspect
        frame = inspect.currentframe()
        call_info = {
            'function': 'rearrange',
            'args': args,
            'kwargs': kwargs,
            'caller': frame.f_back.f_code.co_filename + ':' + str(frame.f_back.f_lineno),
            'caller_function': frame.f_back.f_code.co_name
        }
        call_stack_info.append(call_info)
        
        try:
            result = original_rearrange(*args, **kwargs)
            print(f"✓ einops.rearrange at {call_info['caller']} in {call_info['caller_function']}")
            return result
        except Exception as e:
            call_info['error'] = str(e)
            problematic_calls.append(call_info)
            print(f"✗ einops.rearrange FAILED at {call_info['caller']}: {e}")
            raise
    
    def traced_reduce(*args, **kwargs):
        import inspect
        frame = inspect.currentframe()
        call_info = {
            'function': 'reduce',
            'args': args,
            'kwargs': kwargs,
            'caller': frame.f_back.f_code.co_filename + ':' + str(frame.f_back.f_lineno),
            'caller_function': frame.f_back.f_code.co_name
        }
        call_stack_info.append(call_info)
        
        try:
            result = original_reduce(*args, **kwargs)
            print(f"✓ einops.reduce at {call_info['caller']} in {call_info['caller_function']}")
            return result
        except Exception as e:
            call_info['error'] = str(e)
            problematic_calls.append(call_info)
            print(f"✗ einops.reduce FAILED at {call_info['caller']}: {e}")
            raise
    
    def traced_repeat(*args, **kwargs):
        import inspect
        frame = inspect.currentframe()
        call_info = {
            'function': 'repeat',
            'args': args,
            'kwargs': kwargs,
            'caller': frame.f_back.f_code.co_filename + ':' + str(frame.f_back.f_lineno),
            'caller_function': frame.f_back.f_code.co_name
        }
        call_stack_info.append(call_info)
        
        try:
            result = original_repeat(*args, **kwargs)
            print(f"✓ einops.repeat at {call_info['caller']} in {call_info['caller_function']}")
            return result
        except Exception as e:
            call_info['error'] = str(e)
            problematic_calls.append(call_info)
            print(f"✗ einops.repeat FAILED at {call_info['caller']}: {e}")
            raise
    
    # Monkey patch einops
    einops.rearrange = traced_rearrange
    einops.reduce = traced_reduce  
    einops.repeat = traced_repeat
    
    return original_rearrange, original_reduce, original_repeat

def test_compilation_with_tracing():
    """Test your actual code with einops tracing to find compilation issues"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing compilation with tracing on device: {device}")
    
    # Start tracing
    print("=== Starting einops call tracing ===")
    original_funcs = trace_einops_calls()
    
    try:
        # Import your modules (this might trigger some einops calls)
        print("Importing your modules...")
        from edm2.vae import VAE, MixedDiscriminator
        print("✓ Modules imported")
        
        # Create your models
        print("\n=== Creating models ===")
        channels = [3, 16, 64, 256, 8]
        n_res_blocks = 3
        
        vae = VAE(
            channels=channels, 
            n_res_blocks=n_res_blocks, 
            spatial_compressions=[1,2,2,2], 
            time_compressions=[1,2,2,1]
        ).to(device)
        print("✓ VAE created")
        
        discriminator = MixedDiscriminator().to(device)
        print("✓ Discriminator created")
        
        # Test forward pass without compilation first
        print("\n=== Testing uncompiled forward pass ===")
        test_frames = torch.randn(1, 3, 32, 64, 64).to(device)
        
        with torch.no_grad():
            r_mean, r_logvar, mean, logvar = vae(test_frames)
            print(f"✓ Uncompiled VAE forward pass successful")
            print(f"  r_mean shape: {r_mean.shape}")
            print(f"  r_logvar shape: {r_logvar.shape}")
        
        # Now test compilation - this is where the issues likely occur
        print("\n=== Testing compilation ===")
        print("Compiling VAE (watch for einops calls that cause issues)...")
        
        # Clear previous calls
        global call_stack_info, problematic_calls
        call_stack_info.clear()
        problematic_calls.clear()
        
        try:
            compiled_vae = torch.compile(vae)
            print("✓ VAE compilation successful")
        except Exception as e:
            print(f"✗ VAE compilation failed: {e}")
            traceback.print_exc()
        
        print("\nCompiling Discriminator...")
        try:
            compiled_discriminator = torch.compile(discriminator)
            print("✓ Discriminator compilation successful")
        except Exception as e:
            print(f"✗ Discriminator compilation failed: {e}")
            traceback.print_exc()
        
        # Test compiled forward pass - this triggers the actual compilation
        print("\n=== Testing compiled forward pass (this triggers actual compilation) ===")
        
        try:
            with torch.no_grad():
                r_mean_compiled, r_logvar_compiled, mean_compiled, logvar_compiled = compiled_vae(test_frames)
            print("✓ Compiled VAE forward pass successful")
        except Exception as e:
            print(f"✗ Compiled VAE forward pass failed: {e}")
            traceback.print_exc()
        
        # Test the training loop einops operations
        print("\n=== Testing training loop einops operations ===")
        
        # These are from your training code
        frames = torch.randn(1, 32, 64, 64, 3).to(device)  # Your original format
        frames = frames.float() / 127.5 - 1
        
        # This is the first einops call in your training code
        print("Testing: einops.rearrange(frames, 'b t h w c-> b c t h w')")
        frames_rearranged = einops.rearrange(frames, 'b t h w c-> b c t h w')
        
        # Test with compiled VAE
        with torch.no_grad():
            r_mean, r_logvar, mean, _ = compiled_vae(frames_rearranged)
        
        # These are the problematic einops calls from your training loop
        print("Testing: einops.rearrange(frames, 'b c t h w -> (b t) c h w')")
        frames_flat = torch.clip(einops.rearrange(frames_rearranged, 'b c t h w -> (b t) c h w'), -1, 1)
        
        print("Testing: einops.rearrange(r_mean, 'b c t h w -> (b t) c h w')")
        r_mean_flat = torch.clip(einops.rearrange(r_mean, 'b c t h w -> (b t) c h w'), -1, 1)
        
        print("✓ All training loop einops operations successful")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        traceback.print_exc()
    
    finally:
        # Restore original functions
        einops.rearrange, einops.reduce, einops.repeat = original_funcs
    
    # Report findings
    print("\n" + "="*60)
    print("=== EINOPS CALL ANALYSIS ===")
    print(f"Total einops calls traced: {len(call_stack_info)}")
    print(f"Problematic calls: {len(problematic_calls)}")
    
    if call_stack_info:
        print("\nAll einops calls made:")
        for i, call in enumerate(call_stack_info):
            print(f"{i+1}. {call['function']} at {call['caller']} in {call['caller_function']}")
            if 'error' in call:
                print(f"   ERROR: {call['error']}")
    
    if problematic_calls:
        print("\n⚠️  PROBLEMATIC CALLS:")
        for call in problematic_calls:
            print(f"- {call['function']} at {call['caller']}")
            print(f"  Error: {call['error']}")
    else:
        print("\n✓ No einops calls failed directly")
        print("The compilation warnings are likely from dynamic shape issues")
        print("in einops internal operations, not from your direct calls.")

def find_exact_problematic_operations():
    """Try to trigger the exact compilation warnings you're seeing"""
    print("\n" + "="*60)
    print("=== FINDING EXACT PROBLEMATIC OPERATIONS ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test specific patterns that are known to cause torch.compile issues
    test_cases = [
        {
            'name': 'Dynamic batch*time flattening',
            'operation': lambda x: einops.rearrange(x, 'b c t h w -> (b t) c h w'),
            'input_shape': (1, 8, 32, 64, 64)
        },
        {
            'name': 'Dynamic batch*time reconstruction', 
            'operation': lambda x: einops.rearrange(x, '(b t) c h w -> b c t h w', b=1),
            'input_shape': (32, 8, 64, 64)
        },
        {
            'name': 'Time-channel mixing',
            'operation': lambda x: einops.rearrange(x, 'b c t h w -> b (c t) h w'),
            'input_shape': (1, 8, 32, 64, 64)
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        @torch.compile
        def compiled_operation(x):
            return test_case['operation'](x)
        
        try:
            test_input = torch.randn(*test_case['input_shape']).to(device)
            result = compiled_operation(test_input)
            print(f"✓ {test_case['name']} compiled successfully")
        except Exception as e:
            print(f"✗ {test_case['name']} compilation failed: {e}")

if __name__ == "__main__":
    print("=== DEBUGGING YOUR SPECIFIC EINOPS COMPILATION ISSUES ===")
    test_compilation_with_tracing()
    find_exact_problematic_operations()
    
    print("\n" + "="*60)
    print("=== RECOMMENDATIONS ===")
    print("1. Look at the einops calls traced above")
    print("2. The ones inside your VAE/MixedDiscriminator are likely the culprits")
    print("3. Replace problematic einops with native torch operations")
    print("4. Or add @torch._dynamo.disable to specific methods")