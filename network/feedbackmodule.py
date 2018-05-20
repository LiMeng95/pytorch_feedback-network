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

class FeedbackConvLSTM(nn.Module):
    def __init__(self, input_size, output_sizes, strides, num_iterations, x_kernel_size, h_kernel_size):
        super(FeedbackConvLSTM, self).__init__()
        
        assert len(output_sizes) == len(strides)
        
        self.physical_depth = len(output_sizes)
        self.num_iterations = num_iterations
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.strides = strides
        self.x_kernel_size = x_kernel_size
        self.h_kernel_size = h_kernel_size
        
        self.convlstm_cells = nn.ModuleList()
        for it in range(self.physical_depth):
            if it == 0:
                inp_size = input_size
            else:
                inp_size = output_sizes[it-1]
            outp_size = output_sizes[it]
            stride = strides[it]
            
            self.convlstm_cells.append(
                ConvLSTMCell(inp_size, outp_size, x_kernel_size, h_kernel_size, stride)
            )
        
    def forward(self, x):
        end_xts = []
        for t in range(self.num_iterations):
            for d in range(self.physical_depth):
                if d == 0:
                    x_t = x # x_t^{d-1}
                x_t = self.convlstm_cells[d].forward(x_t)
            end_xts.append(x_t)

        # prevent memory leak
        for cell in self.convlstm_cells:
            cell.reset_state()

        return end_xts

class FeedbackModule(nn.Module):
    # physical layers = list of modules, must implement reset_state() method
    # num_iterations, repeat n times, for physical_depth*num_iterations -> virtual_depth
    def __init__(self, physical_layers, num_iterations):
        super(FeedbackModule, self).__init__()
        self.cells = nn.ModuleList(physical_layers)
        self.num_iterations = num_iterations

    def forward(self, x):
        end_xts = []
        for t in range(self.num_iterations):
            x_t = x
            for cell in self.cells:
                x_t = cell(x_t)#.forward(x_t)
            end_xts.append(x_t)

        for cell in self.cells:
            cell.reset_state()

        return end_xts

