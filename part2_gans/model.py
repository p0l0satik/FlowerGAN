from tkinter.tix import Tree
from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm



class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)

        self.gamma_layer = nn.Linear(num_features, embed_features, bias=False)
        self.bias_layer = nn.Linear(num_features, embed_features, bias=False)


    def forward(self, inputs, embeds):

        gamma = self.gamma_layer(embeds) # TODO 
        bias = self.bias_layer(embeds) # TODO

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = super().forward(inputs) # TODO: apply batchnorm

        return outputs * gamma[..., None, None] + bias[..., None, None]


class NormReluConv(nn.Module):
  def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False):
    super().__init__()
    if batchnorm:
        self.bn = AdaptiveBatchNorm(in_channels, embed_channels)
    else:
        self.bn = nn.BatchNorm2d(in_channels)

    self.relu = nn.ReLU()
    self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3))

  def forward(self, x, embeds=None):
    if embeds:
        x = self.bn(x, embeds)
    else:
        x = self.bn(x)

    x = self.relu(x)
    return self.conv(x)


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        self.apply_conv = False
        self.up = False
        self.down = False
        self.module_list = nn.ModuleList()
        if upsample:
            self.up = True
            self.upsample = nn.UpsamplingNearest2d(scale_factor = 2)
        self.block1 = NormReluConv(in_channels, out_channels, embed_channels, batchnorm)
        self.block2 = NormReluConv(out_channels, out_channels, embed_channels, batchnorm)
        if downsample:
            self.down = True
            self.downsample = nn.AvgPool2d(2)

        if in_channels != out_channels:
            self.apply_conv = True
            self.residual_conv = nn.Conv2d(in_channels, in_channels, 1)


        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        res_inputs = inputs.copy()

        if self.apply_conv:
            res_inputs = self.residual_conv(res_inputs)
        if self.up:
            inputs = self.upsample(inputs)
        
        inputs = self.block1(inputs, embeds)
        inputs = self.block2(inputs, embeds)

        if self.down:
            inputs = self.downsample(inputs)

        return res_inputs + inputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks
        # TODO

    def forward(self, noise, labels):
        # TODO

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()
        # TODO

    def forward(self, inputs, labels):
        pass # TODO

        scores = ... # TODO

        assert scores.shape = (inputs.shape[0],)
        return scores