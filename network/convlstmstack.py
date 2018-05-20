import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from .convlstmcell import ConvLSTMCell

class ConvLSTMStack(nn.Module):
    
    def __init__(self, input_size, output_size, x_kernel_size, h_kernel_size, stride, n):        
        super(ConvLSTMStack, self).__init__()
        operations = []
        self.cells = nn.ModuleList()
        first_cell = ConvLSTMCell(input_size, output_size, x_kernel_size, h_kernel_size, stride, n)
        self.cells.append(first_cell)
        operations.append(first_cell)
        # for _ in range(n-1):
        #     operations.append(nn.ReLU())
        #     next_cell = ConvLSTMCell(output_size, output_size, x_kernel_size, h_kernel_size, 1)
        #     self.cells.append(next_cell)
        #     operations.append(next_cell)
        self.stack = nn.Sequential(*operations)

        
    def reset_state(self):
        for cell in self.cells:
            cell.reset_state()
    
    def forward(self, x):
        return self.stack(x)#.forward(x)

