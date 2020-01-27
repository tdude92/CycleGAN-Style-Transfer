import torch

# Learning rate decay for Adam optimizers.
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


# Weight initialization function.
# Initialize with a standard deviation of 0.02
def weight_init(module):
    class_name = module.__class__.__name__
    if class_name.find("Conv") != -1:
        module.weight.data.normal_(0.0, 0.02)


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
    return torch.stack(selected).to(device)
