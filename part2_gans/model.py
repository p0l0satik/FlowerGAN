from tkinter.tix import Tree
from turtle import forward
from numpy import pad
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
    # print("conv:", in_channels, out_channels)
    if batchnorm:
        self.bn = AdaptiveBatchNorm(in_channels, embed_channels)
    else:
        self.bn = nn.BatchNorm2d(in_channels)

    self.relu = nn.ReLU()
    self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))

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
        # print("res block", in_channels, out_channels)
        self.apply_conv = False
        self.up = upsample
        self.down = False
        self.module_list = nn.ModuleList()
        if upsample:
            self.upsample = nn.UpsamplingNearest2d(scale_factor = 2)
        self.block1 = NormReluConv(in_channels, out_channels, embed_channels, batchnorm)
        self.block2 = NormReluConv(out_channels, out_channels, embed_channels, batchnorm)
        if downsample:
            self.down = True
            self.downsample = nn.AvgPool2d(2)

        if in_channels != out_channels:

            self.apply_conv = True
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)


        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        if self.up:
            inputs = self.upsample(inputs)

        res_inputs = inputs
        if self.apply_conv:
            res_inputs = self.residual_conv(res_inputs)
        inputs = self.block1(inputs, embeds)

        inputs = self.block2(inputs, embeds)


        if self.down:
            inputs = self.downsample(inputs)
            res_inputs = self.downsample(res_inputs)

        # print("dow", inputs.shape)
        
        # print("res_block", res_inputs.shape, inputs.shape)
        return res_inputs + inputs

class CalcChannels():
    def __init__(self, min_ch, max_ch):
        self.min_ch = min_ch
        self.max_ch = max_ch

    def up(self, step):
        curr_ch = self.min_ch * (2**step)
        return int(self.max_ch) if curr_ch > self.max_ch else int(curr_ch)

    def down(self, step):
        curr_ch = self.max_ch / (2**step)
        return int(self.min_ch) if curr_ch < self.min_ch else int(curr_ch)


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
        super().__init__()
        self.max_ch = max_channels
        self.min_ch = min_channels
        calc = CalcChannels(min_channels, max_channels)
        self.class_cond = use_class_condition
        self.output_size = 4 * 2**num_blocks
        self.embeddings = nn.Embedding(num_classes, noise_channels)
        if self.class_cond:
            self.embed_lin = nn.utils.spectral_norm(nn.Linear(noise_channels*2, max_channels*16))
        else:
            self.embed_lin = nn.utils.spectral_norm(nn.Linear(noise_channels, max_channels*16))
        self.conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.conv_blocks.append(PreActResBlock( calc.down(i), 
                                                    calc.down(i+1), 
                                                    noise_channels, 
                                                    use_class_condition, 
                                                    True, 
                                                    False))

        self.conv_blocks.append(NormReluConv(calc.down(num_blocks), 3))
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise, labels):

        # print("forw", noise.shape)
        if self.class_cond:
            class_embeddings = self.embeddings(labels)
            input_embeddings = torch.cat((class_embeddings, noise), dim = -1)
        else:
            input_embeddings = noise
        
        input = self.embed_lin(input_embeddings).reshape((-1, self.max_ch, 4, 4))
        # print("inp", input.shape)
        for layer in self.conv_blocks:
            input = layer(input)
        
        outputs = self.sigmoid(input)

        # print("outputs", outputs.shape, noise.shape[0], 3, self.output_size, self.output_size)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        # print("GEN OK")
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
        super().__init__()
        self.head = use_projection_head
        calc = CalcChannels(min_channels, max_channels)
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(NormReluConv(3, calc.up(0)))
        for i in range(num_blocks):
            self.conv_blocks.append(PreActResBlock(calc.up(i), calc.up(i+1), None, False, False, True))
        self.conv_blocks.append(nn.ReLU())
        self.conv_blocks.append(torch.nn.AvgPool2d(4, stride=2, divisor_override=1))
        if self.head:
            self.embeddings = nn.utils.spectral_norm(nn.Embedding(num_classes, max_channels))
        print(num_classes, max_channels)
        self.lin = nn.utils.spectral_norm(nn.Linear(max_channels, 1, True))
        
    def forward(self, inputs, labels):
        for layer in self.conv_blocks:
            inputs = layer(inputs) 
            print("shape", inputs.shape)

        # inputs = torch.sum(inputs,  axis=(2, 3))
        print(inputs.shape)
        batch_size = inputs.shape[0]
        scores = self.lin(inputs.squeeze())
        print(scores.shape)

        if self.head:
            y = self.embeddings(labels)
            scores += y
        scores = scores.squeeze()
        print(scores.shape)

        assert scores.shape == (inputs.shape[0],)

        return scores