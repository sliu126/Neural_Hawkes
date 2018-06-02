# -*- coding: utf-8 -*-
"""

wrappers

@author: hongyuan
"""

import time
import numpy
import os

#from nhrl.utils import numeric
#from nhrl.agents import models, optimizers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

def Variable(data, use_gpu=False):
    return torch.autograd.Variable(data.cuda() if use_gpu else data)

def Arange(end, start=0, device=torch):
    res = device.LongTensor()
    return torch.arange(start, end, out=res)
