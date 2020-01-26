import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models import Generator, Discriminator
from utilities import weight_init, LambdaLR

import os
import cv2
import itertools
import numpy as np


# Constants
MODEL_PATH      = "models/0"
START_EPOCH     = 1
END_EPOCH       = 200
DECAY_START     = 100

BATCH_SIZE      = 1
POOL_SIZE       = 50
LOSS_G_LAMBDA   = 10
ADAM_BETA_1     = 0.5                       # Adam optimizer beta1 (beta2 is always 0.999).
G_LR            = 0.0002                    # Generator learning rate.
D_LR            = 0.00005                   # Discriminator learning rate.
ON_CUDA         = torch.cuda.is_available() # Boolean for CUDA availability.

if ON_CUDA:
    device = "cuda:0"
else:
    device = "cpu"

# Function used to update pools.
def update_pool(pool, images, max_size = POOL_SIZE, device = "cpu"):
    selected = []
    for image in images:
        if len(pool) < max_size:
            # Add to the pool
            pool.append(image)
            selected.append(image)
        elif torch.randn(1).item() < 0.5:
            # Use image but don't add to pool.
            selected.append(image)
        else:
            # Replace an image in the pool with the new image.
            idx = torch.randint(len(pool), (1, 1)).item()
            selected.append(pool[idx])
            pool[idx] = image
    output = torch.stack(selected).to(device)
    return output


# Define transformations applied to input images.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
data_A = datasets.ImageFolder("data/monet2photo/A", transform = transform)
data_B = datasets.ImageFolder("data/monet2photo/B", transform = transform)
loader_A = torch.utils.data.DataLoader(data_A, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
loader_B = torch.utils.data.DataLoader(data_B, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

# Instantiate generator and discriminator models.
gen_A = Generator("generator_B2A")
gen_B = Generator("generator_A2B")
disc_A = Discriminator("discriminator_A")
disc_B = Discriminator("discriminator_B")

# Load models if available.
if MODEL_PATH[-1] != "/":
    MODEL_PATH += "/"
try:
    # Attempt to load models.
    gen_A.load_state_dict(torch.load(MODEL_PATH + "gen_A.pth"))
    gen_B.load_state_dict(torch.load(MODEL_PATH + "gen_B.pth"))
    disc_A.load_state_dict(torch.load(MODEL_PATH + "disc_A.pth"))
    disc_B.load_state_dict(torch.load(MODEL_PATH + "disc_B.pth"))
except FileNotFoundError:
    # Create folder to store models.
    os.makedirs(MODEL_PATH)

    # Initialize weights.
    gen_A.apply(weight_init)
    gen_B.apply(weight_init)
    disc_A.apply(weight_init)
    disc_B.apply(weight_init)

    # Save newly created models.
    torch.save(gen_A.state_dict(), MODEL_PATH + "gen_A.pth")
    torch.save(gen_B.state_dict(), MODEL_PATH + "gen_B.pth")
    torch.save(disc_A.state_dict(), MODEL_PATH + "disc_A.pth")
    torch.save(disc_B.state_dict(), MODEL_PATH + "disc_B.pth")

if ON_CUDA:
    gen_A.cuda()
    gen_B.cuda()
    disc_A.cuda()
    disc_B.cuda()

# Loss functions.
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Optimizers and LR schedulers.
optimizer_G = torch.optim.Adam(itertools.chain(gen_A.parameters(), gen_B.parameters()), lr = G_LR, betas = (ADAM_BETA_1, 0.999))
optimizer_D_A = torch.optim.Adam(disc_A.parameters(), lr = D_LR, betas = (ADAM_BETA_1, 0.999))
optimizer_D_B = torch.optim.Adam(disc_B.parameters(), lr = D_LR, betas = (ADAM_BETA_1, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda = LambdaLR(END_EPOCH, START_EPOCH - 1, DECAY_START).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda = LambdaLR(END_EPOCH, START_EPOCH - 1, DECAY_START).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda = LambdaLR(END_EPOCH, START_EPOCH - 1, DECAY_START).step)

# Training loop
pool_A = []
pool_B = []
for epoch in range(START_EPOCH, END_EPOCH + 1):
    for real_A,  real_B in zip(loader_A, loader_B):
        # Each element of zip(loader_A, loader_B) is in the form:
        # ((image_A, label_A), (image_B, label_B))
        # label_A, label_B are useless, so we discard them.
        real_A = real_A[0]
        real_B = real_B[0]
        fake_A = gen_A.forward(real_B)
        fake_B = gen_B.forward(real_A)

        # Update pools.
        fake_A = update_pool(pool_A, fake_A, device = device)
        fake_B = update_pool(pool_B, fake_B, device = device)

        # Generate labels.
        real_label = torch.Tensor([1]).view(-1)
        fake_label = torch.Tensor([0]).view(-1)

        #########################
        # Train Discriminators. #
        #########################

        # Real batch
        output_DA = disc_A(real_A)
        output_DB = disc_B(real_B)
        DA_A = output_DA.item()
        DB_B = output_DB.item()

        real_loss_DA = criterion_GAN(output_DA, real_label)
        real_loss_DB = criterion_GAN(output_DB, real_label)

        real_loss_DA.backward()
        real_loss_DB.backward()

        # Fake batch
        output_DA = disc_A(fake_A.detach())
        output_DB = disc_B(fake_B.detach())
        DA_GA_B = output_DA.item()
        DB_GB_A = output_DB.item()

        fake_loss_DA = criterion_GAN(output_DA, fake_label)
        fake_loss_DB = criterion_GAN(output_DB, fake_label)

        loss_DA = (real_loss_DA + fake_loss_DA) / 2
        loss_DB = (real_loss_DB + fake_loss_DB) / 2

        fake_loss_DA.backward()
        fake_loss_DB.backward()

        optimizer_D_A.step()
        optimizer_D_B.step()

        ###################
        # Train Generator #
        ###################
        output_DA = disc_A(fake_A)
        output_DB = disc_B(fake_B)

        # Adversarial loss
        loss_G_adv = torch.mean(criterion_GAN(output_DA, real_label) + criterion_GAN(output_DB, real_label))

        # Cyclic loss
        cyc_A = gen_A(fake_B)
        cyc_B = gen_B(fake_A)

        loss_G_cyc = torch.mean(criterion_cycle(cyc_A, real_A) + criterion_cycle(cyc_B, real_B))

        # Total Generator loss
        loss_G = loss_G_adv + LOSS_G_LAMBDA*loss_G_cyc

        loss_G.backward()
        optimizer_G.step()

        # TODO: Add identity mapping loss.

    # Output training stats.
    print("#######################################")
    print("Epoch", epoch)
    print("---------------------------------------")
    print("Discriminator_A Loss:", loss_DA.item())
    print("Discriminator_B Loss:", loss_DB.item())
    print("Generator Loss:", loss_G.item())
    print()
    print("D_A(A) =", DA_A)
    print("D_A(G_A(B)) =", DA_GA_B)
    print()
    print("D_B(B) =", DB_B)
    print("D_B(G_B(A)) =", DB_GB_A)
    print("#######################################")
    print()

    # TODO: Save a sample for visual reference.
