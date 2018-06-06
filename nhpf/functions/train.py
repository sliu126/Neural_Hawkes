# -*- coding: utf-8 -*-
# !/usr/bin/python
"""

Train models

@author: hongyuan
"""

import pickle
import time
import numpy
import os
import sys
import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable

from nhpf.functions import train_nh

#
import argparse
__author__ = 'Hongyuan Mei'

def main():

    parser = argparse.ArgumentParser(description='Trainning model ...')
    parser.add_argument(
        '-m', '--Model', default='nh',
        choices=['nh'],
        help='what model to use? For now only nh is available'
    )
    parser.add_argument(
        '-pd', '--PathData', required=True,
        help='Path of data? e.g. ../../data/pilothawkes/'
    )
    parser.add_argument(
        '-d', '--DimLSTM', default=16, type=int,
        help='Dimension of LSTM?'
    )
    parser.add_argument(
        '-sb', '--SizeBatch', default=50, type=int,
        help='Size of mini-batch'
    )
    parser.add_argument(
        '-tp', '--TrackPeriod', default=1000, type=int,
        help='How many sequences before every checkpoint?'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', default=20, type=int,
        help='Max epoch number of training'
    )
    parser.add_argument(
        '-lr', '--LearnRate', default=1e-3, type=float,
        help='What is the (starting) learning rate?'
    )
    parser.add_argument(
        '-gpu', '--UseGPU', action='store_true', #default=0, type=int, choices=[0,1],
        help='Use GPU?'
    )
    parser.add_argument(
        '-np', '--NumParticle', default=1, type=int,
        help='Num of particles?'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int,
        help='Random seed. e.g. 12345'
    )

    args = parser.parse_args()
    dict_args = vars(args)
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    r"""
    make tag_model with arguments
    """
    use_in_foldername = [
        'Model', 'DimLSTM', 'SizeBatch','TrackPeriod', 'MaxEpoch', 'LearnRate', 'UseGPU', 'Seed'
    ]
    # think about modiying this
    # we definitely do not want to keep all of these
    tag_model = '_PID={}'.format(id_process)
    for k in dict_args:
        v = dict_args[k]
        if k in use_in_foldername:
            tag_model += '_{}={}'.format(k, str(v))

    dict_args['PathData'] = os.path.abspath(dict_args['PathData'])
    dict_args['Version'] = torch.__version__
    dict_args['ID'] = id_process
    dict_args['TIME'] = time_current

    r"""
    we used PID + TIME to label each tracking folder
    we did this in case sometimes we wantted to have different models for same specs
    so we want to distinguish different runs
    but I do changes as:
    1. remove TIME because it is usually too long and not reader friendly
    2. keep PID to distinguish different runs
    3. besides PID, use arguments as parts of folder name, to be reader friendly
    """

    tag_data = '_Data='+os.path.basename(
        os.path.normpath(args.PathData))

    path_log = '../../Logs/log'+tag_data+'/log'+tag_model+'/'
    file_log = os.path.join(os.path.abspath(path_log), 'log.txt')
    file_model = os.path.join(os.path.abspath(path_log), 'model')
    os.system(
        'mkdir -p '+os.path.abspath(path_log))

    dict_args['PathLog'] = file_log
    dict_args['PathSave'] = file_model

    if dict_args['Model'] == 'nh':
        trainer = train_nh
    else:
        raise Exception(
            "Model {} not implemented".format(dict_args['Model']))

    run_func = trainer.run_complete

    run_func(dict_args)

if __name__ == "__main__": main()

