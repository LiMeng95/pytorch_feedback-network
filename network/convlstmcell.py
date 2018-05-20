import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math

class ConvLSTMCell(nn.Module):
    
    def __init__(self, input_size, output_size, x_kernel_size, h_kernel_size, stride=1, num_cell=3):
        super(ConvLSTMCell, self).__init__()
        pad_x = int(math.floor(x_kernel_size/2))
        pad_h = int(math.floor(h_kernel_size/2))
        self.output_size = output_size
        self.stride = stride
        
        # input gate
        conv_i_x = nn.Conv2d(input_size, output_size, x_kernel_size, stride=stride, padding=pad_x)
        batchnorm_i_x = nn.BatchNorm2d(output_size)
        self.i_x = nn.Sequential()
        self.i_x.add_module('conv_1_x', conv_i_x)
        self.i_x.add_module('bn_1_x', batchnorm_i_x)
        for i in range(num_cell-1):
            relu_name = 'relu' + str(i+1)
            conv_name = 'conv_{}_x'.format(str(i + 2))
            bn_name = 'bn_{}_x'.format(str(i + 2))
            self.i_x.add_module(relu_name, nn.ReLU())
            self.i_x.add_module(conv_name, nn.Conv2d(output_size, output_size, x_kernel_size, stride=1, padding=pad_x))
            self.i_x.add_module(bn_name, nn.BatchNorm2d(output_size))

        conv_i_h = nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h)
        batchnorm_i_h = nn.BatchNorm2d(output_size)
        self.i_h = nn.Sequential()
        self.i_h.add_module('conv_1_h', conv_i_h)
        self.i_h.add_module('bn_1_h', batchnorm_i_h)
        for i in range(num_cell-1):
            relu_name = 'relu' + str(i+1)
            conv_name = 'conv_{}_h'.format(str(i + 2))
            bn_name = 'bn_{}_h'.format(str(i + 2))
            self.i_h.add_module(relu_name, nn.ReLU())
            self.i_h.add_module(conv_name, nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h))
            self.i_h.add_module(bn_name, nn.BatchNorm2d(output_size))
        
        # forget gate
        # self.conv_f_x = nn.Conv2d(input_size, output_size, x_kernel_size, stride=stride, padding=pad_x)
        # self.batchnorm_f_x = nn.BatchNorm2d(output_size)
        # self.conv_f_h = nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h)
        # self.batchnorm_f_h = nn.BatchNorm2d(output_size)
        # initialize bias to 1 for x forget input
        # self.conv_f_x.bias.data.fill_(1)

        conv_f_x = nn.Conv2d(input_size, output_size, x_kernel_size, stride=stride, padding=pad_x)
        batchnorm_f_x = nn.BatchNorm2d(output_size)
        self.f_x = nn.Sequential()
        self.f_x.add_module('conv_1_x', conv_f_x)
        self.f_x.add_module('bn_1_x', batchnorm_f_x)
        for i in range(num_cell-1):
            relu_name = 'relu' + str(i+1)
            conv_name = 'conv_{}_x'.format(str(i + 2))
            bn_name = 'bn_{}_x'.format(str(i + 2))
            self.f_x.add_module(relu_name, nn.ReLU())
            self.f_x.add_module(conv_name, nn.Conv2d(output_size, output_size, x_kernel_size, stride=1, padding=pad_x))
            self.f_x.add_module(bn_name, nn.BatchNorm2d(output_size))

        conv_f_h = nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h)
        batchnorm_f_h = nn.BatchNorm2d(output_size)
        self.f_h = nn.Sequential()
        self.f_h.add_module('conv_1_h', conv_f_h)
        self.f_h.add_module('bn_1_h', batchnorm_f_h)
        for i in range(num_cell-1):
            relu_name = 'relu' + str(i+1)
            conv_name = 'conv_{}_h'.format(str(i + 2))
            bn_name = 'bn_{}_h'.format(str(i + 2))
            self.f_h.add_module(relu_name, nn.ReLU())
            self.f_h.add_module(conv_name, nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h))
            self.f_h.add_module(bn_name, nn.BatchNorm2d(output_size))
        
        # cell gate
        # self.conv_c_x = nn.Conv2d(input_size, output_size, x_kernel_size, stride=stride, padding=pad_x)
        # self.batchnorm_c_x = nn.BatchNorm2d(output_size)
        # self.conv_c_h = nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h)
        # self.batchnorm_c_h = nn.BatchNorm2d(output_size)

        conv_c_x = nn.Conv2d(input_size, output_size, x_kernel_size, stride=stride, padding=pad_x)
        batchnorm_c_x = nn.BatchNorm2d(output_size)
        self.c_x = nn.Sequential()
        self.c_x.add_module('conv_1_x', conv_c_x)
        self.c_x.add_module('bn_1_x', batchnorm_c_x)
        for i in range(num_cell-1):
            relu_name = 'relu' + str(i+1)
            conv_name = 'conv_{}_x'.format(str(i + 2))
            bn_name = 'bn_{}_x'.format(str(i + 2))
            self.c_x.add_module(relu_name, nn.ReLU())
            self.c_x.add_module(conv_name, nn.Conv2d(output_size, output_size, x_kernel_size, stride=1, padding=pad_x))
            self.c_x.add_module(bn_name, nn.BatchNorm2d(output_size))

        conv_c_h = nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h)
        batchnorm_c_h = nn.BatchNorm2d(output_size)
        self.c_h = nn.Sequential()
        self.c_h.add_module('conv_1_h', conv_c_h)
        self.c_h.add_module('bn_1_h', batchnorm_c_h)
        for i in range(num_cell-1):
            relu_name = 'relu' + str(i+1)
            conv_name = 'conv_{}_h'.format(str(i + 2))
            bn_name = 'bn_{}_h'.format(str(i + 2))
            self.c_h.add_module(relu_name, nn.ReLU())
            self.c_h.add_module(conv_name, nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h))
            self.c_h.add_module(bn_name, nn.BatchNorm2d(output_size))

        # output gate
        # self.conv_o_x = nn.Conv2d(input_size, output_size, x_kernel_size, stride=stride, padding=pad_x)
        # self.batchnorm_o_x = nn.BatchNorm2d(output_size)
        # self.conv_o_h = nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h)
        # self.batchnorm_o_h = nn.BatchNorm2d(output_size)

        conv_o_x = nn.Conv2d(input_size, output_size, x_kernel_size, stride=stride, padding=pad_x)
        batchnorm_o_x = nn.BatchNorm2d(output_size)
        self.o_x = nn.Sequential()
        self.o_x.add_module('conv_1_x', conv_o_x)
        self.o_x.add_module('bn_1_x', batchnorm_o_x)
        for i in range(num_cell-1):
            relu_name = 'relu' + str(i+1)
            conv_name = 'conv_{}_x'.format(str(i + 2))
            bn_name = 'bn_{}_x'.format(str(i + 2))
            self.o_x.add_module(relu_name, nn.ReLU())
            self.o_x.add_module(conv_name, nn.Conv2d(output_size, output_size, x_kernel_size, stride=1, padding=pad_x))
            self.o_x.add_module(bn_name, nn.BatchNorm2d(output_size))

        conv_o_h = nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h)
        batchnorm_o_h = nn.BatchNorm2d(output_size)
        self.o_h = nn.Sequential()
        self.o_h.add_module('conv_1_h', conv_o_h)
        self.o_h.add_module('bn_1_h', batchnorm_o_h)
        for i in range(num_cell-1):
            relu_name = 'relu' + str(i+1)
            conv_name = 'conv_{}_h'.format(str(i + 2))
            bn_name = 'bn_{}_h'.format(str(i + 2))
            self.o_h.add_module(relu_name, nn.ReLU())
            self.o_h.add_module(conv_name, nn.Conv2d(output_size, output_size, h_kernel_size, stride=1, padding=pad_h))
            self.o_h.add_module(bn_name, nn.BatchNorm2d(output_size))

        # init bias
        for i in range(num_cell):
            self.i_x[i * 3].bias.data.fill_(0)
            self.f_x[i * 3].bias.data.fill_(0)
            self.c_x[i * 3].bias.data.fill_(0)
            self.o_x[i * 3].bias.data.fill_(0)

        self.last_cell = None
        self.last_h = None
        
    def reset_state(self):
        self.last_cell = None
        self.last_h = None
    
    def forward(self, x):
        if self.last_cell is None:
            self.last_cell = Variable(torch.zeros(
                (x.size(0), self.output_size, int(x.size(2)/self.stride), 
                 int(x.size(3)/self.stride))
            ))
            if x.is_cuda:
                self.last_cell = self.last_cell.cuda()
        if self.last_h is None:
            self.last_h = Variable(torch.zeros(
                (x.size(0), self.output_size, int(x.size(2)/self.stride), 
                 int(x.size(3)/self.stride))
            ))
            if x.is_cuda:
                self.last_h = self.last_h.cuda()
        h = self.last_h
        c = self.last_cell
        
        # input gate
        # input_x = self.batchnorm_i_x(self.conv_i_x(x))
        # input_h = self.batchnorm_i_h(self.conv_i_h(h))
        input_x = self.i_x(x)
        input_h = self.i_h(h)
        input_gate = F.sigmoid(input_x + input_h)
        
        # forget gate
        # forget_x = self.batchnorm_f_x(self.conv_f_x(x))
        # forget_h = self.batchnorm_f_h(self.conv_f_h(h))
        forget_x = self.f_x(x)
        forget_h = self.f_h(h)
        forget_gate = F.sigmoid(forget_x + forget_h)
        
        # forget gate
        # cell_x = self.batchnorm_c_x(self.conv_c_x(x))
        # cell_h = self.batchnorm_c_h(self.conv_c_h(h))
        cell_x = self.c_x(x)
        cell_h = self.c_h(h)
        cell_intermediate = F.tanh(cell_x + cell_h) # g
        cell_gate = (forget_gate * c) + (input_gate * cell_intermediate)
        
        # output gate
        # output_x = self.batchnorm_o_x(self.conv_o_x(x))
        # output_h = self.batchnorm_o_h(self.conv_o_h(h))
        output_x = self.o_x(x)
        output_h = self.o_h(h)
        output_gate = F.sigmoid(output_x + output_h)
        
        next_h = output_gate * F.tanh(cell_gate)
        self.last_cell = cell_gate
        self.last_h = next_h
        
        return next_h
