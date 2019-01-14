#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common PyTorch deep learning structure.

   Author: Meng Cao
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from .general_utils import Progbar
from .transformer import EncoderDecoder


class FullModel:
    """This class implements all model training and evluation methods.
    """
    def __init__(self, config, write_summary=True, device=None):
        """Initialize the model.
        """
        self.config = config
        self.logger = config.logger

        # initialize model
        self.model = None
        
        # create optimizer and criterion
        self.optimizer = None
        self.criterion = None

        # find device
        if device is not None:
            self.model.to(device)

        # create summary for tensorboard visualization
        if write_summary:
            self.writer = SummaryWriter(self.config.path_summary)


    def _initialize_model(self):
        """model initialization.
        """
        raise NotImplementedError


    def _get_optimizer(self):
        """Create Optimizer for training.
        """
        raise NotImplementedError


    def _get_criterion(self):
        """Implement loss function.
        """
        raise NotImplementedError


    def load_weights(self, path):
        """Load pre-trained weights.
        """
        self.model.load_state_dict(torch.load(path))


    def loss_batch(self, model, loss_func, inputs, labels, optimizer=None):
        """Compute loss and update model weights on a batch of data.
        """
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        if optimizer is not None:
            with torch.set_grad_enabled(True):
                loss.backward() # compute gradients
                optimizer.step() # update weights
                optimizer.zero_grad()

        return loss.item(), outputs.detach()


    def train_epoch(self, model, dataset, criterion, optimizer, epoch):
        """Train the model for one single epoch.
        """
        model.train() # set the model to train mode
        # define progress bar for visualization
        prog = Progbar(target=len(dataset))
        
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(dataset):
            # compute loss and update on a batch of data
            batch_loss, _ = self.loss_batch(model, criterion, inputs, labels, optimizer=optimizer)

            train_loss += batch_loss
            prog.update(i + 1, [("train loss", batch_loss)])    

            if self.writer: # write summary to tensorboard
                self.writer.add_scalar('batch_loss', batch_loss, epoch*len(dataset) + i + 1)

        # compute the average loss
        epoch_loss = train_loss / len(dataset)

        return epoch_loss


    def evaluate(self, model, dataset, criterion):
        """Evaluate the model, return average loss and accuracy.
        """
        model.eval()
        with torch.no_grad():
            eval_loss, eval_corrects = 0.0, 0.0
            for i, (inputs, labels) in enumerate(dataset):
                # compute loss and update on a batch of data
                batch_loss, outputs = self.loss_batch(model, criterion, 
                        inputs, labels, None)
                _, preds = torch.max(outputs, 1) # preds: [batch_size]
                eval_loss += batch_loss
                eval_corrects += torch.sum(preds == labels).double()

            avg_loss = eval_loss / len(dataset)
            avg_acc  = eval_corrects / len(dataset.dataset)

        return avg_loss, avg_acc


    def fit(self, train_set, development_set):
        """Model training.
        """
        num_epochs = self.config.num_epochs
        best_acc = 0.

        for epoch in range(num_epochs):
            self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # print('-' * 10)
            # train
            train_loss = self.train_epoch(self.model, train_set, self.criterion, self.optimizer, epoch)
            self.logger.info("Traing Loss: {}".format(train_loss))

            # eval
            eval_loss, eval_acc = self.evaluate(self.model, development_set, self.criterion)
            self.logger.info("Evaluation:")
            self.logger.info("- loss: {}".format(eval_loss))
            self.logger.info("- acc: {}".format(eval_acc))

            if self.writer:
                # monitor loss and accuracy
                self.writer.add_scalar('epoch_loss', train_loss, epoch)
                self.writer.add_scalar('eval_loss', eval_loss, epoch)
                self.writer.add_scalar('eval_acc', eval_acc, epoch)

            if eval_acc >= best_acc:
                best_acc = eval_acc
                # save the model
                self.logger.info("New best score!")
                torch.save(self.model.state_dict(), self.config.dir_model + "model.pickle")
                self.logger.info("model is saved at: {}".format(self.config.dir_model))


    def predict(self, inputs):
        """Prediction.
        """
        # evaluation mode
        self.model.eval()
        with torch.no_grad():
            # [batch_size, num_classes]
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1) # [batch_size]
        return outputs, preds
