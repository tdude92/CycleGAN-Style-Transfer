import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
NGF             = 32    # Number of filters in the first Generator layer.
NDF             = 64    # Number of filters in the first Discriminator layer.
IMG_DIM         = 256   # Image input/output dimensions.
IMG_DEPTH       = 3     # Colour channels (RGB)


# Residual blocks used between the encoder
# and decoder layers in the Generators.
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # Residual function, F(x):
        #   input: x
        #   --> Convolutional layer + Instance Normalization
        #   --> ReLU
        #   --> Convolutional layer + Instance Normalization
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
        return F.relu(self.F(x) + x, inplace = True)


# CycleGAN Generator
class Generator(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.encode = nn.Sequential(
            # Dimensions: (256, 256, IMG_DEPTH)

            nn.Conv2d(IMG_DEPTH, NGF, kernel_size = 7, stride = 1, padding = 3),
            nn.LeakyReLU(0.2, inplace = True),
            # Dimensions: (256, 256, NGF)

            nn.Conv2d(NGF, NGF * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.InstanceNorm2d(NGF * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # Dimensions: (128, 128, NGF * 2)

            nn.Conv2d(NGF * 2, NGF * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.InstanceNorm2d(NGF * 4),
            nn.LeakyReLU(0.2, inplace = True)
            # Dimensions: (64, 64, NGF * 4)
        )

        self.transform = nn.Sequential(
            ResidualBlock(NGF * 4),
            ResidualBlock(NGF * 4),
            ResidualBlock(NGF * 4),
            ResidualBlock(NGF * 4),
            ResidualBlock(NGF * 4),
            ResidualBlock(NGF * 4)
            # Dimensions: (64, 64, NGF * 4)
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(NGF * 4, NGF * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.InstanceNorm2d(NGF * 2),
            nn.ReLU(True),
            # Dimensions: (128, 128, NGF * 2)

            nn.ConvTranspose2d(NGF * 2, NGF, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.InstanceNorm2d(NGF),
            nn.ReLU(True),
            # Dimensions: (256, 256, NGF)

            nn.Conv2d(NGF, IMG_DEPTH, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.Tanh()
            # Dimensions: (256, 256, IMG_DEPTH)
        )
    
    def forward(self, x):
        x = self.encode(x)
        x = self.transform(x)
        x = self.decode(x)
        return x


# PatchGAN discriminator
class Discriminator(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.pipeline = nn.Sequential(
            # Dimensions: (256, 256, IMG_DEPTH)

            nn.Conv2d(IMG_DEPTH, NDF, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace = True),
            # Dimensions: (128, 128, NDF)

            nn.Conv2d(NDF, NDF * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.InstanceNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # Dimensions: (64, 64, NDF * 2)

            nn.Conv2d(NDF * 2, NDF * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.InstanceNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # Dimensions: (32, 32, NDF * 4)

            nn.Conv2d(NDF * 4, NDF * 8, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # Dimensions: (32, 32, NDF * 8)

            nn.Conv2d(NDF * 8, 1, kernel_size = 3, stride = 1, padding = 1)
            # Dimensions: (32, 32, 1)
        )
    
    def forward(self, x):
        # Return the average probability.
        return self.pipeline(x).mean()
