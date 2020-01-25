import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models import ResidualBlock

import os
import cv2
import numpy as np


# Constants
NGF             = 32
NDF             = 64
IMG_DIM         = 256
IMG_DEPTH       = 3
BATCH_SIZE      = 1
POOL_SIZE       = 50


# Residual blocks used between the encoder
# and decoder layers in the Generators.
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # Residual function, F(x):
        #   input: x
        #   --> Convolutional layer + Batch Normalization
        #   --> ReLU
        #   --> Convolutional layer + Batch Normalization
        self.F = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(dim)
        )
    
    def forward(self, x):
        # M(x) = F(x) + x
        # Where:
        #   M(x) is the desired mapping from input to output.
        #   F(x) = M(x) - x
        return self.F(x) + x


# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()