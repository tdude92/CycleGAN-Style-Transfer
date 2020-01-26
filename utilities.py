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