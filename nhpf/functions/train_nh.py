# -*- coding: utf-8 -*-
# !/usr/bin/python
"""

Train neural Hawkes process

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
#import torchvision.transforms as transforms

from nhpf.models import nhp
from nhpf.io import processors
#from nhpf.eval import distance

#
import argparse
__author__ = 'Hongyuan Mei'

def run_complete(args):

    assert args['Model'] == 'nh', "not neural Hawkes?"
    assert args['NumParticle'] == 1, "only one particle (== raw seq) needed"

    numpy.random.seed(args['Seed'])
    torch.manual_seed(args['Seed'])

    with open(os.path.join(args['PathData'], 'train.pkl'), 'rb') as f:
        pkl_train = pickle.load(f, encoding='latin1')
    with open(os.path.join(args['PathData'], 'dev.pkl'), 'rb') as f:
        pkl_dev = pickle.load(f, encoding='latin1')

    learning_rate = args['LearnRate']

    r"""
    train : train -- complete seqs
    dev/test : dev/test -- complete seqs

    obs_num : \# of observed types
    unobs_num : \# of unobserved types

    but this may be changed, if we enable stochastic missingness
    """

    data = pkl_train['train']
    data_dev = pkl_dev['dev']

    obs_num, unobs_num = int(pkl_train['dim_process']), 0
    hidden_dim = args['DimLSTM']

    agent = nhp.NeuralHawkes(
        obs_num, unobs_num, hidden_dim,
        use_gpu=args['UseGPU'] )

    if args['UseGPU']:
        agent.cuda()

    sampling = 1
    total_event_num = obs_num + unobs_num
    proc = processors.DataProcessorNeuralHawkes(
        idx_BOS=total_event_num,
        idx_EOS=total_event_num+1,
        idx_PAD=total_event_num+2,
        sampling=sampling, use_gpu=args['UseGPU'],
        missing_types=range(obs_num, total_event_num)
    )
    r"""
    |FOR FUTURE USE| comments about missing types
    """

    logger = processors.LogWriter(
        args['PathLog'], args)

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    print("Start training ... ")
    total_logP_best = -1e6
    time0 = time.time()

    episodes = []
    total_rewards = []

    max_episode = args['MaxEpoch'] * len(data)
    report_gap = args['TrackPeriod']

    time_sample = 0.0
    time_train_only = 0.0
    input = []

    for episode in range(max_episode):

        idx_seq = episode % len(data)
        one_seq = data[ idx_seq ]

        time_sample_0 = time.time()
        input.append(
            proc.processSeq( one_seq ) )
        time_sample += (time.time() - time_sample_0)

        if len(input) >= args['SizeBatch']:

            batchdata_seqs = proc.processBatchSeqsWithParticles( input )

            agent.train()
            time_train_only_0 = time.time()
            objective, _ = agent(
                batchdata_seqs, mode=1 )

            objective.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            #torch.nn.utils.clip_grad_norm( agent.parameters(), 0.25 )
            #print("backward done")
            optimizer.step()
            optimizer.zero_grad()
            time_train_only += (time.time() - time_train_only_0)
            #agent.checkWeights()
            #agent.constrainWeights()

            input = []

            if episode % report_gap == report_gap - 1:

                time1 = time.time()
                time_train = time1 - time0
                time0 = time1

                print("Validating at episode {}".format(episode))
                total_logP = 0.0
                total_num_act = 0.0

                input_dev = []
                agent.eval()
                #print("scale now is : {}".format(agent.scale.data))

                index = 0
                for one_seq_dev in data_dev:
                    input_dev.append(
                        proc.processSeq(
                            one_seq_dev,
                            one_seq_dev[-1]['time_since_start']
                        )
                    )

                    if len(input_dev) >= args['SizeBatch']:
                        batchdata_seqs_dev = proc.processBatchSeqsWithParticles(
                            input_dev )
                        objective_dev, num_events_dev = agent(
                            batchdata_seqs_dev, mode=1 )
                        total_logP -= float(objective_dev.data.sum() )
                        total_num_act += float(
                            num_events_dev.data.sum() / (args['NumParticle'] * 1.0 ) )

                        input_dev = []
                    index += 1

                total_logP /= total_num_act
                updated = None
                if total_logP > total_logP_best:
                    total_logP_best = total_logP
                    updated = True
                else:
                    updated = False

                message = "Episode {}, loglik is {} and current best is {}".format(episode, round(total_logP, 4), round(total_logP_best, 4))
                if updated:
                    message += ", best updated at this episode"
                    torch.save(
                        agent.state_dict(), args['PathSave'])
                logger.checkpoint(message)
                print(message)
                episodes.append(episode)
                total_rewards.append(round(total_logP, 4))
                time1 = time.time()
                time_dev = time1 - time0
                time0 = time1
                message = "time for train {} episdoes is {} and time for dev is {}".format(
                    report_gap, round(time_train, 2), round(time_dev, 2))

                time_sample, time_train_only = 0.0, 0.0
                logger.checkpoint(message)
                print(message)
                #print("For debug, the log lambda target and neg integral is {} and {}, total num is {}".format( round(total_log_lambda_target, 4), round(total_neg_integral, 4), total_num_act) )
                #print("for complete data, the neg integral should be smaller")
    logger.checkpoint("training finished")
    print("training finished")
    return episodes, total_rewards




def main():

	# default arguments
    args = {
        'Model': 'nh',
        'Version': torch.__version__,
        'PathData': '../../data/pilothawkes/',
        'TrainComplete': True,
        'DimLSTM': 8,
        'SizeBatch': 50,
        'TrackPeriod': 50 * 20,
        'MaxEpoch': 3,
        'LearnRate': 0.001,
        'UseGPU': False,
        'NumParticle': 1,
        'Seed': 12345,
        'ID': str(1),
        'TIME': 'fake',
        'PathLog': os.path.abspath('./log.txt'),
        'PathSave': os.path.abspath('./model')
    }

    episodes, rewards = run_complete(args)


if __name__ == "__main__": main()
