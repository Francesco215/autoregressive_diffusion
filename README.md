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
Here is how you make the training in a way that is sample efficient.

Let $(x_1,\dots,x_n)$ be a sequence of frames from the training set.
To train the model in a way that is sample efficient we create two copies of the input sequence:
- The first part is not noised $x_c=(x_1,\dots,x_n)$
- The second part is noised $x_n=(\tilde x_1,\dots,\tilde x_n)$ where each frame is noised as $\tilde x_i = x_i +\sigma_i\epsilon$.

The input sequence $x$ is formed by concatenating the two sequences
$$
    x = x_c \oplus x_n = (x_1,\dots,x_n,\tilde x_1,\dots,\tilde x_n)
$$

To maintain causality we have to make sure that the $i$-th output of the network $N(x)_i$ is only dependent from the noised frame $\tilde x_i$ and all of the previous frames $\{x_j\}_{j<i}$
$$
N(x)_i=f(\tilde x_i,  \{x_j\}_{j<i},\sigma_i)
$$

In this model there are two modules that can transfer information between frames
- `VideoAttention`
- `3DCausalConvolution`

### Video Attention module
Here is an illustrative image that shows how the information moves


This can be archieved by doing block-sparse masking using [FlexAttention](https://pytorch.org/blog/flexattention/). Thanks to it no computation is wasted.
![](readme_images/masking.png)

### 3D Causal Convolution
Wierdly enough, the convolution layer is the hardest to explain because it relies on a couple of tricks to make sure that the code runs as fast as possible during training.

I'll write later how it works exactly. For now you can read the code




## Comparison

Most SOTA video diffusion models denoise the entire sequence of pixels all at once.
The main difference is that

- They tend to have a fixed video lenght (~8 seconds)
- They can't be used for world modelling



TODO: Cambia l'immagine che questa è imbarazzante
