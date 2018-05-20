import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from .feedbackmodule import FeedbackConvLSTM
from .feedbackmodule import FeedbackModule
from .convlstmstack import ConvLSTMStack

class FeedbackNet(nn.Module):  # 12 physical depth, 4 iterations, n_classes output classes
    def __init__(self, n_classes):
        super(FeedbackNet, self).__init__()
        self.n_classes = n_classes
        self.num_iterations = 4
        self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        self.batchnorm = nn.BatchNorm2d(16)
        stack = [
            ConvLSTMStack(16, 16, 3, 3, 1, 3),  # Iterate(fi,fo,kernel,kernel,stride,conv_num_per_stack)
            ConvLSTMStack(16, 32, 3, 3, 2, 3),
            ConvLSTMStack(32, 64, 3, 3, 2, 3),
            ConvLSTMStack(64, 64, 3, 3, 1, 3),
        ]
        self.feedback = FeedbackModule(stack, self.num_iterations)
        self.avg_pool = nn.AvgPool2d(8)#4)
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        # self.output = nn.Linear(256, self.n_classes)#n_output_classes)
        self.output = nn.Linear(64, self.n_classes)
        print('Initializing FeedbackNet48!')
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x_all = self.feedback(x)
        x_finished = []
        for x_i in x_all:
            x_i = F.relu(x_i)
            x_i = self.avg_pool(x_i)
            # x_i = x_i.view(-1, 256)
            # x_i = self.dropout1(x_i)
            # x_i = self.linear(x_i)
            # x_i = F.relu(x_i)
            # x_i = self.dropout2(x_i)
            # x_i = self.output(x_i)
            x_i = x_i.view(-1, 64)
            x_i = self.output(x_i)
            x_finished.append(x_i)
        return x_finished
