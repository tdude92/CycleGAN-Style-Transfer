import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models import Generator, Discriminator
from utilities import LambdaLR, weight_init, update_pool

import os
import cv2
import itertools
import numpy as np

# Constants
DATA_PATH       = "data/monet2photo"
MODEL_PATH      = "models/0"
START_EPOCH     = 1
END_EPOCH       = 200
DECAY_START     = 100

BATCH_SIZE      = 1
ADAM_BETA_1     = 0.5                       # Adam optimizer beta1 (beta2 is always 0.999).
G_LR            = 0.0002                    # Generator learning rate.
D_LR            = 0.00003                   # Discriminator learning rate.
ON_CUDA         = torch.cuda.is_available() # Boolean for CUDA availability.

if ON_CUDA:
    print("GPU available. Training with CUDA.")
    device = "cuda:0"
else:
    print("GPU not available. Training with CPU.")
    device = "cpu"

if MODEL_PATH[-1] != "/":
    MODEL_PATH += "/"
if DATA_PATH[-1] != "/":
    DATA_PATH += "/"

# Define transformations applied to input images.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
data_A = datasets.ImageFolder(DATA_PATH + "A", transform = transform)
data_B = datasets.ImageFolder(DATA_PATH + "B", transform = transform)
loader_A = torch.utils.data.DataLoader(data_A, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
loader_B = torch.utils.data.DataLoader(data_B, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

# Instantiate generator and discriminator models.
gen_A = Generator("generator_B2A")
gen_B = Generator("generator_A2B")
disc_A = Discriminator("discriminator_A")
disc_B = Discriminator("discriminator_B")

# Load models if available.
try:
    # Attempt to load models.
    gen_A.load_state_dict(torch.load(MODEL_PATH + "gen_A.pth"))
    gen_B.load_state_dict(torch.load(MODEL_PATH + "gen_B.pth"))
    disc_A.load_state_dict(torch.load(MODEL_PATH + "disc_A.pth"))
    disc_B.load_state_dict(torch.load(MODEL_PATH + "disc_B.pth"))
    print("Loaded models from " + MODEL_PATH)
except FileNotFoundError:
    print("Saving models to " + MODEL_PATH)

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
    for real_A, real_B in zip(loader_A, loader_B):
        # Each element of zip(loader_A, loader_B) is in the form:
        # ((image_A, label_A), (image_B, label_B))
        # label_A, label_B are useless, so we discard them.
        real_A = real_A[0]
        real_B = real_B[0]

        # Move tensors to GPU if available.
        if ON_CUDA:
            real_A = real_A.cuda()
            real_B = real_B.cuda()

        fake_A = gen_A.forward(real_B)
        fake_B = gen_B.forward(real_A)

        # Update pools.
        fake_A = update_pool(pool_A, fake_A, device = device)
        fake_B = update_pool(pool_B, fake_B, device = device)

        # Generate labels.
        real_label = torch.Tensor([1])
        fake_label = torch.Tensor([0])

        # Move tensors to GPU if available.
        if ON_CUDA:
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()

        ####################
        # Train Generators #
        ####################
        gen_A.zero_grad()
        gen_B.zero_grad()

        output_DA = disc_A(fake_A)
        output_DB = disc_B(fake_B)

        DA_GA_B = output_DA.item()
        DA_GB_A = output_DB.item()

        # Adversarial loss
        loss_GA_adv = torch.mean(criterion_GAN(output_DA, real_label))
        loss_GB_adv = torch.mean(criterion_GAN(output_DB, real_label))

        # Cyclic consistency loss
        cyc_A = gen_A(fake_B)
        cyc_B = gen_B(fake_A)

        loss_GA_cyc = torch.mean(criterion_cycle(cyc_A, real_A))
        loss_GB_cyc = torch.mean(criterion_cycle(cyc_B, real_B))

        # Identity mapping loss.
        id_A = gen_A(real_A)
        id_B = gen_B(real_B)

        loss_GA_id = criterion_identity(id_A, real_A)
        loss_GB_id = criterion_identity(id_B, real_B)

        # Total Generator losses
        loss_GA = loss_GA_adv + 10*loss_GA_cyc + 5*loss_GA_id
        loss_GB = loss_GB_adv + 10*loss_GB_cyc + 5*loss_GB_id

        loss_G = loss_GA + loss_GB

        loss_G.backward(retain_graph = True)
        optimizer_G.step()

        ########################
        # Train Discriminators #
        ########################
        disc_A.zero_grad()
        disc_B.zero_grad()

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

        fake_loss_DA.backward()
        fake_loss_DB.backward()

        loss_DA = (real_loss_DA + fake_loss_DA) / 2
        loss_DB = (real_loss_DB + fake_loss_DB) / 2

        optimizer_D_A.step()
        optimizer_D_B.step()

    # Step learning rate schedulers.
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Output training stats.
    print("#######################################")
    print("Epoch", epoch)
    print("---------------------------------------")
    print("Discriminator_A Loss:", loss_DA.item())
    print("Discriminator_B Loss:", loss_DB.item())
    print("Generator_B2A Loss:", loss_GA.item())
    print("Generator_A2B Loss:", loss_GB.item())
    print()
    print("D_A(A) =", DA_A)
    print("D_A(G_A(B)) =", DA_GA_B)
    print()
    print("D_B(B) =", DB_B)
    print("D_B(G_B(A)) =", DB_GB_A)
    print("#######################################")
    print()

    # Save a sample for visual reference.
    os.makedirs("out/Epoch" + str(epoch), exist_ok = True)

    real_A = cv2.cvtColor(((real_A.cpu().detach().numpy().squeeze().transpose(1, 2, 0) + np.ones((256, 256, 3))) * 127.5).astype(np.float32), cv2.COLOR_RGB2BGR)
    real_B = cv2.cvtColor(((real_B.cpu().detach().numpy().squeeze().transpose(1, 2, 0) + np.ones((256, 256, 3))) * 127.5).astype(np.float32), cv2.COLOR_RGB2BGR)
    fake_A = cv2.cvtColor(((fake_A.cpu().detach().numpy().squeeze().transpose(1, 2, 0) + np.ones((256, 256, 3))) * 127.5).astype(np.float32), cv2.COLOR_RGB2BGR)
    fake_B = cv2.cvtColor(((fake_B.cpu().detach().numpy().squeeze().transpose(1, 2, 0) + np.ones((256, 256, 3))) * 127.5).astype(np.float32), cv2.COLOR_RGB2BGR)
    cyc_A  = cv2.cvtColor(((cyc_A.cpu().detach().numpy().squeeze().transpose(1, 2, 0) + np.ones((256, 256, 3))) * 127.5).astype(np.float32), cv2.COLOR_RGB2BGR)
    cyc_B  = cv2.cvtColor(((cyc_B.cpu().detach().numpy().squeeze().transpose(1, 2, 0) + np.ones((256, 256, 3))) * 127.5).astype(np.float32), cv2.COLOR_RGB2BGR)

    cv2.imwrite("out/Epoch" + str(epoch) + "/" + "A_real.jpg", real_A)
    cv2.imwrite("out/Epoch" + str(epoch) + "/" + "A_fake.jpg", fake_A)
    cv2.imwrite("out/Epoch" + str(epoch) + "/" + "A_cyclic.jpg", cyc_A)
    cv2.imwrite("out/Epoch" + str(epoch) + "/" + "B_real.jpg", real_B)
    cv2.imwrite("out/Epoch" + str(epoch) + "/" + "B_fake.jpg", fake_B)
    cv2.imwrite("out/Epoch" + str(epoch) + "/" + "B_cyclic.jpg", cyc_B)
