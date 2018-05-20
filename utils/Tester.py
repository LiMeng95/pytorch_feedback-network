from __future__ import print_function

import os
from PIL import Image
from .log import logger
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F


class TestParams(object):
    # params based on your local env
    use_gpu = False             # default do not use gpu

    # loading existing checkpoint
    ckpt = './models/ckpt_epoch_baseline.pth'     # path to the ckpt file


class Tester(object):

    TestParams = TestParams

    def __init__(self, model, test_params, val_data):
        assert isinstance(test_params, TestParams)
        self.params = test_params

        # load model
        self.model = model
        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES
        if self.params.use_gpu:
            logger.info('Set CUDA_VISIBLE_DEVICES to 0...')
            self.model = self.model.cuda()

        # DataLoader
        self.val_data = val_data

        self.model.eval()

    def test(self):
        logger.info('Val on validation set...')
        correct_tp1 = np.zeros(self.model.num_iterations)
        correct_tp5 = np.zeros(self.model.num_iterations)
        total = 0

        for step, (data, label) in enumerate(self.val_data):
            # val model
            inputs = Variable(data, volatile=True)
            if self.params.use_gpu:
                inputs = inputs.cuda()

            outputs = self.model(inputs)
            total += label.size(0)
            for it in range(self.model.num_iterations):
                for p_index, p in enumerate(outputs[it].data):
                    p = p.view(1, 100)
                    if label[p_index] in p.topk(1)[1].squeeze().tolist():
                        correct_tp1[it] += 1
                        correct_tp5[it] += 1
                    elif label[p_index] in p.topk(5)[1].squeeze().tolist():
                        correct_tp5[it] += 1

        for it in range(self.model.num_iterations):
            print('Test accuracy(tp1) for iteration %i: %f %%' % (it, 100 * correct_tp1[it] / total))

        for it in range(self.model.num_iterations):
            print('Test accuracy(tp5) for iteration %i: %f %%' % (it, 100 * correct_tp5[it] / total))


    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(torch.load(ckpt))
