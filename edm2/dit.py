import einops
import torch
from torch import nn
from torch.nn import functional as F

from .utils import BetterModule, normalize, resample, mp_silu, mp_sum, mp_cat, MPFourier, bmult
from .conv import  MPConv, MPCausal3DGatedConv, Gating
from .attention import FrameAttention, VideoAttention

import inspect

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        mlp_hidden_size = int(hidden_size * mlp_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # TODO: change the attention 
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(approximate="tanh"), 
            nn.Linear(mlp_hidden_size, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.initialize_adaLN_zero()

    def initialize_adaLN_zero(self):
        """
        Initializes the weights and biases of the adaLN_modulation layer
        to ensure that the DiT block starts as an identity function.
        The gates (`gate_msa` and `gate_mlp`) are initialized to zero.
        """
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


    def forward(self, x, c):
        modulation_params = self.adaLN_modulation(c)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            torch.chunk(modulation_params, chunks=6, dim=1)

        x_norm1 = self.norm1(x)
        x_modulated1 = modulate(x_norm1, shift_msa, scale_msa)
        attn_output, _ = self.attn(x_modulated1, x_modulated1, x_modulated1)

        x = x + gate_msa.unsqueeze(1) * attn_output

        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_modulated2)

        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x



class DiffusionTransformer(BetterModule):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels         = 192,       # Base multiplier for the number of channels.
        patch_size             = 4,         # Size of the image patches
        num_blocks             = 3,         # Number of residual blocks per resolution.
        label_balance          = 0.5,       # Balance between noise embedding (0) and class embedding (1).
        concat_balance         = 0.5,       # Balance between skip connections (0) and main path (1).
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.patch_size = patch_size
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_res = Gating()

        # Embedding.
        self.emb_fourier_sigma = MPFourier(model_channels)
        self.emb_sigma = MPConv(model_channels, model_channels, kernel=[]) 
        self.emb_fourier_time = MPFourier(model_channels)
        self.emb_time = MPConv(model_channels, model_channels, kernel=[]) 
        self.emb_label = MPConv(label_dim, model_channels, kernel=[]) if label_dim != 0 else None       

        # Blocks
        self.conv_in = MPConv(img_channels*patch_size**2, model_channels, kernel=[])
        self.blocks = nn.ModuleList([DiTBlock(model_channels, num_heads = 4) for _ in range(num_blocks)])
        self.conv_out = MPConv(model_channels, img_channels*patch_size**2, kernel=[])

        # Saves the kwargs
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self.kwargs = {arg: values[arg] for arg in args if arg != "self"}

    def forward(self, x, c_noise, conditioning = None, cache:dict=None, update_cache=False):
        if cache is None: cache = {}
        batch_size, time_dimention = x.shape[:2]
        n_context_frames = cache.get('n_context_frames', 0)

        res = x.clone()
        out_res, updated_n_context_frames = self.out_res(c_noise, n_context_frames)
        if update_cache: cache['n_context_frames']=updated_n_context_frames

        # Reshaping
        x = einops.rearrange(x, 'b t c (h ph) (w pw) -> (b t) (h w) (ph pw c)', ph=self.patch_size, pw=self.patch_size)
        c_noise = einops.rearrange(c_noise, 'b t -> (b t)')

        # Time embedding
        frame_labels = torch.arange(time_dimention, device=x.device).repeat(batch_size) + n_context_frames
        frame_labels = frame_labels.log1p().to(c_noise.dtype) / 4 
        frame_embeddings = self.emb_time(self.emb_fourier_time(frame_labels))

        emb = self.emb_sigma(self.emb_fourier_sigma(c_noise))
        emb = mp_sum(emb, frame_embeddings, t=0.5)
        if self.emb_label is not None and conditioning is not None:
            conditioning = einops.rearrange(conditioning, 'b t -> (b t)')
            conditioning = F.one_hot(conditioning, num_classes=self.label_dim).to(c_noise.dtype)*self.label_dim**(0.5)
            conditioning = self.emb_label(conditioning)
            emb = mp_sum(emb, conditioning, t=1/3)
        emb = mp_silu(emb)

        c_noise = einops.rearrange(c_noise, '(b t) -> b t', b=batch_size)
        
        x = self.conv_in(x)
        for i, block in enumerate(self.blocks):
            x = block(x, emb)
        x = self.conv_out(x)

        x = einops.rearrange(x, '(b t) (h w) (ph pw c) -> b t c h w', b=batch_size, ph=self.patch_size, pw=self.patch_size)
        x = mp_sum(x, res, out_res)
        return x, cache