#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:55:20 2018

@author: jimmy
"""
import peak_loss_pytorch
import torch
from torch.autograd import Variable

def peak_loss(model, training_set, test_set):
    train_data = training_set.data.numpy()
    test_data = test_set.data.numpy()
    c_input_training = Variable(training_set).cuda()
    c_input_testing = Variable(test_set).cuda()
    c_encoded, c_decoded = model(c_input_training)
    result = c_decoded.cpu().data.numpy()
    training_loss2 = '%.5f'%(peak_loss_pytorch.cal_loss(result, train_data))
    c_encoded, c_decoded = model(c_input_testing)
    result = c_decoded.cpu().data.numpy()
    testing_loss2 = '%.5f'%(peak_loss_pytorch.cal_loss(result, test_data))
    return training_loss2, testing_loss2
