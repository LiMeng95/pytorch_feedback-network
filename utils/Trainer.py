from __future__ import print_function

import os
import numpy as np

import torch as t
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchnet import meter

from .log import logger


def get_learning_rates(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    lrs = np.asarray(lrs, dtype=np.float)
    return lrs


class TrainParams(object):
    # required params
    max_epoch = 30

    # optimizer and criterion and learning rate scheduler
    optimizer = None
    criterion = None
    lr_scheduler = None         # should be an instance of ReduceLROnPlateau or _LRScheduler

    # params based on your local env
    gpus = []                   # default to use CPU mode
    save_dir = './models/'            # default `save_dir`

    # loading existing checkpoint
    ckpt = None                 # path to the ckpt file

    # saving checkpoints
    save_freq_epoch = 1         # save one ckpt per `save_freq_epoch` epochs


class Trainer(object):

    TrainParams = TrainParams

    def __init__(self, model, train_params, train_data, val_data=None):
        assert isinstance(train_params, TrainParams)
        self.params = train_params

        # Data loaders
        self.train_data = train_data
        self.val_data = val_data

        # criterion and Optimizer and learning rate
        self.last_epoch = 0
        self.criterion = self.params.criterion
        self.optimizer = self.params.optimizer
        self.lr_scheduler = self.params.lr_scheduler
        logger.info('Set criterion to {}'.format(type(self.criterion)))
        logger.info('Set optimizer to {}'.format(type(self.optimizer)))
        logger.info('Set lr_scheduler to {}'.format(type(self.lr_scheduler)))

        # load model
        self.model = model
        logger.info('Set output dir to {}'.format(self.params.save_dir))
        if os.path.isdir(self.params.save_dir):
            pass
        else:
            os.makedirs(self.params.save_dir)

        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # meters
        self.num_iter = self.model.num_iterations
        self.loss_meter = meter.AverageValueMeter()
        # self.confusion_matrix = [meter.ConfusionMeter(100), meter.ConfusionMeter(100),
        #                          meter.ConfusionMeter(100), meter.ConfusionMeter(100)]

        # set CUDA_VISIBLE_DEVICES
        if len(self.params.gpus) > 0:
            gpus = ','.join([str(x) for x in self.params.gpus])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            self.params.gpus = tuple(range(len(self.params.gpus)))
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpus))
            self.model = nn.DataParallel(self.model, device_ids=self.params.gpus)
            self.model = self.model.cuda()

        self.model.train()

    def train(self):
        # vis = Visualizer()
        best_loss = np.inf
        for epoch in range(self.last_epoch, self.params.max_epoch):

            self.loss_meter.reset()
            # for i in range(self.num_iter):
            #     self.confusion_matrix[i].reset()

            self.last_epoch += 1
            logger.info('Start training epoch {}'.format(self.last_epoch))

            self._train_one_epoch()

            # save model
            if (self.last_epoch % self.params.save_freq_epoch == 0) or (self.last_epoch == self.params.max_epoch - 1):
                save_name = self.params.save_dir + 'ckpt_epoch_{}.pth'.format(self.last_epoch)
                t.save(self.model.state_dict(), save_name)

            self._val_one_epoch()

            if self.loss_meter.value()[0] < best_loss:
                logger.info('Found a better ckpt ({:.3f} -> {:.3f}), '.format(best_loss, self.loss_meter.value()[0]))
                best_loss = self.loss_meter.value()[0]

            # print info
            logger.info("The end of epoch:{epoch}, lr:{lr}, train loss:{loss}:".format(
                epoch=epoch, loss=self.loss_meter.value()[0], lr=get_learning_rates(self.optimizer)
            ))

            # adjust the lr
            # if epoch == 81 or epoch == 122:
            #     for i, param_group in enumerate(self.optimizer.param_groups):
            #         old_lr = float(param_group['lr'])
            #         new_lr = old_lr * 0.1
            #         param_group['lr'] = new_lr
            #         print('Epoch {:5d}: reducing learning rate'
            #               ' of group {} from {:.4e} to {:.4e}.'.format(epoch, i, old_lr, new_lr))
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.loss_meter.value()[0], self.last_epoch)

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(t.load(ckpt))

    def _train_one_epoch(self):
        self.model.train()
        train_correct = np.zeros(self.num_iter)
        train_total = 0
        gamma = 1

        for step, (data, label) in enumerate(self.train_data):
            # train model
            inputs = Variable(data)
            target = Variable(label)
            loss = Variable(t.from_numpy(np.zeros(1))).float()
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.cuda()
                loss = loss.cuda()

            # forward
            outputs = self.model(inputs)
            losses = [self.criterion(out, target) for out in outputs]
            for it in range(len(losses)):
                loss += (gamma ** it) * losses[it]

            # backward
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step(None)

            # meters update
            self.loss_meter.add(loss.data[0])
            # for i in range(self.num_iter):
            #     self.confusion_matrix[i].add(outputs[i].data, target.data)

            # calculate correct number
            train_total += target.size(0)
            for it in range(self.num_iter):
                _, predicted = t.max(outputs[it].data, 1)
                train_correct[it] += (predicted == target.data).sum()

        for it in range(self.num_iter):
            print('Train accuracy for iteration %i: %f %%' % (it, 100 * train_correct[it] / train_total))

    def _val_one_epoch(self):
        self.model.eval()
        logger.info('Val on validation set...')
        correct_tp1 = np.zeros(self.num_iter)
        correct_tp5 = np.zeros(self.num_iter)
        total = 0

        for step, (data, label) in enumerate(self.val_data):
            # val model
            inputs = Variable(data, volatile=True)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()

            outputs = self.model(inputs)
            total += label.size(0)
            for it in range(self.num_iter):
                for p_index, p in enumerate(outputs[it].data):
                    p = p.view(1, 100)
                    if label[p_index] in p.topk(1)[1].squeeze().tolist():
                        correct_tp1[it] += 1
                        correct_tp5[it] += 1
                    elif label[p_index] in p.topk(5)[1].squeeze().tolist():
                        correct_tp5[it] += 1

        for it in range(self.num_iter):
            print('Test accuracy(tp1) for iteration %i: %f %%' % (it, 100 * correct_tp1[it] / total))

        for it in range(self.num_iter):
            print('Test accuracy(tp5) for iteration %i: %f %%' % (it, 100 * correct_tp5[it] / total))
