from torch import nn
from utils import Tester
from network import FeedbackNet
from torch.utils.data import DataLoader
from data import cifar100_test_data

# Set Test parameters
params = Tester.TestParams()
params.use_gpu = True
params.ckpt = './models/ckpt_epoch_baseline.pth'

# models
model = FeedbackNet(100)

# val dataloader
val_dataloader = DataLoader(cifar100_test_data, batch_size=64, shuffle=False, num_workers=4)
print('val dataset len: {}'.format(len(val_dataloader.dataset)))

# Test
tester = Tester(model, params, val_dataloader)
tester.test()
