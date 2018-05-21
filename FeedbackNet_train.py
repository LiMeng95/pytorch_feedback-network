import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import cifar100_train_data, cifar100_test_data
from utils import Trainer
from network import FeedbackNet

# Hyper-params
data_root = './data/'
save_dir = './models/'
batch_size = 64  # batch_size in per GPU, if use GPU mode
num_workers = 4

init_lr = 0.6  # 0.6
lr_decay = 0.5
momentum = 0.9
weight_decay = 1.e-4
nesterov = True

# Set Training parameters
params = Trainer.TrainParams()
params.max_epoch = 300
params.criterion = nn.CrossEntropyLoss()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU mode
params.save_dir = save_dir
params.ckpt = None
params.save_freq_epoch = 10

# load data
print("Loading dataset...")
train_data = cifar100_train_data
val_data = cifar100_test_data

batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('train dataset len: {}'.format(len(train_dataloader.dataset)))

val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('val dataset len: {}'.format(len(val_dataloader.dataset)))

# models
model = FeedbackNet(100)

# optimizer
# trainable_vars = [param for param in model.parameters() if param.requires_grad]
print("Training with sgd")
params.optimizer = torch.optim.SGD(model.parameters(), lr=init_lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay,
                                   nesterov=nesterov)

# Train
params.lr_scheduler = ReduceLROnPlateau(params.optimizer, mode='min', factor=lr_decay,
                                        patience=5, cooldown=5, verbose=True, min_lr=5.e-4)
trainer = Trainer(model, params, train_dataloader, val_dataloader)
trainer.train()
