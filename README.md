# Sample efficient autoregressive diffusion

This repository shows how diffusion models should be trained for video generation and world-modelling

## Inference

Diffusion models work by estimating the score function

$$ s(x,\sigma) = -\nabla \log p(x,\sigma)$$

The most used architectures are mainly Unets and image transformers.

Language models work by estimating the probablilty distribuition of the last token given all of the previous ones

$$F(x_i,\dots,x_0)=-\log p(x_{i+1}|x_i,\dots,x_0)$$

The transformer architecture allows to train such a model in a sample efficient way. This is very important because it multiplies the effictive batch size by the sequence lenght.

To generate a video where each frame if generated autoregressively we need to unite the two paradigms by estimating the score given all of the previous frames.


$$s(x_{i+1},\sigma,x_i,\dots,x_0)=-\nabla_x \log p(x_{i+1},\sigma|x_i,\dots,x_0) $$

Where $x_i,\dots,x_0$ are the noise-free context frames, $x_{i+1}$ is the noisy (to be denoised) frame, and $\sigma$ is the noise level

## Training


## Comparison

Most SOTA video diffusion models denoise the entire sequence of pixels all at once.
The main difference is that

- They tend to have a fixed video lenght (~8 seconds)
- They can't be used for world modelling



