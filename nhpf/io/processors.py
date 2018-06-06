# -*- coding: utf-8 -*-
"""

Processers

@author: hongyuan
"""

import time
import numpy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable


def sampleForIntegral(input, sampling=1, device=torch):
    r"""
    sampling dtimes in each interval given other tensors
    this function only deals with particles of the same seq
    so their duration is the same, but lens may be different
    """

    r"""
    for particles of the same seq, we should use
    same seq of randomly sampled times to compute integral
    because, otherwise, the same particle may get different weights
    so we always bias to the higher-weighted one
    in this way, we are not maximizing the log-likelihood
    instead, we would be maximizing the random upper reach of the log-likelihood
    and there is no guarantee where the mean would go
    to get rid of this bias, for all particles of each seq,
    we should use the EXACTLY same seq of sampled times

    If in the future we want to compare particles while we sample them
    we can precompute this seq of times and compute the integral along the way
    i.e. interval after interval

    given this problem and solution, post(erior) which is computed along sampling
    is completely useless since now
    """

    event, time, post, duration, lens = input

    num_particles, T_plus_2 = event.size()
    assert lens.max() + 2 == T_plus_2, "max len should match"
    #print("sampling for integral ")
    max_sampling = max( int( lens.max() * sampling ), 1 )

    #print("max_sampling = {}".format(max_sampling))
    #print("duration = {}".format(float(duration[0])))
    sampled_times = device.FloatTensor(max_sampling).uniform_(
        0.0, float(duration[0]) ).sort()[0].unsqueeze(0).expand(
            num_particles, max_sampling )

    #sampled_times = device.arange(max_sampling) * duration[0] / max_sampling
    #sampled_times = sampled_times.unsqueeze(0).expand(
    #    num_particles, max_sampling )
    #sampled_times.fill_(0.5)

    dtime_sampling = device.FloatTensor(
        num_particles, max_sampling).fill_(0.0)

    index_of_hidden_sampling = device.LongTensor(
        num_particles, max_sampling).fill_(0)

    dtime_backward_sampling = device.FloatTensor(
        num_particles, max_sampling).fill_(0.0)

    index_of_hidden_backward_sampling = device.LongTensor(
        num_particles, max_sampling).fill_(0)

    cum_time = time.cumsum(dim=1)
    indices_mat = (T_plus_2-1) * device.LongTensor(
        range(num_particles)).unsqueeze(1).expand(num_particles, max_sampling)
    # + device.LongTensor(range(T_plus_2-1)).unsqueeze(0)
    current_step = device.LongTensor(num_particles, max_sampling).fill_(0)

    for j in range( lens.max() + 1 ):

        bench_cum_time = cum_time[:, j].unsqueeze(1).expand(
            num_particles, max_sampling)
        indices_to_edit = sampled_times > bench_cum_time

        dtime_sampling[indices_to_edit] = \
        (sampled_times - bench_cum_time)[indices_to_edit]

        current_step.fill_(j)
        index_of_hidden_sampling[indices_to_edit] = \
        (indices_mat + current_step)[indices_to_edit]

    assert dtime_sampling.min() >= 0.0, "Time >= 0"

    return event, time, post, duration, dtime_sampling, index_of_hidden_sampling
    # idx of output :
    # event 0, time 1, post 2, duration 3,
    # dtime_sampling 4, index_of_hidden_sampling 5


def orgSeq(one_seq, idx_BOS, idx_EOS, idx_PAD, duration):
    r"""
    augment BOS and EOS to the seq
    BOS and EOS are treated as observations
    """
    one_seq_new = []
    duration = one_seq[-1]['time_since_start'] if duration is None else duration
    one_seq_new.append(
        {
            'type_event': idx_BOS, 'time_since_last_event': 0.0,
            'time_since_start': 0.0
        }
    )
    for item in one_seq:
        one_seq_new.append(item)

    one_seq_new.append(
        {
            'type_event': idx_EOS, 'time_since_last_event': 0.0,
            'time_since_start': duration
        }
    )
    return one_seq_new


def processSeq(
    input, duration, idx_BOS, idx_EOS, idx_PAD, device=torch):
    r"""
    make tensors for one seq, adding BOS and EOS events
    """
    one_seq = orgSeq(
        input, idx_BOS, idx_EOS, idx_PAD, duration )
    len_seq = len(one_seq)

    duration = device.FloatTensor(1).fill_(
        float(one_seq[-1]['time_since_start']) )
    post = device.FloatTensor(1).fill_(0.0)
    lens = device.LongTensor(1).fill_(len_seq - 2)

    event = device.LongTensor( 1, len_seq ).fill_(idx_PAD)
    dtime = device.FloatTensor( 1, len_seq ).fill_(0.0)

    for i_item, item in enumerate(one_seq):

        event[:, i_item] = int(item['type_event'])
        dtime[:, i_item] = float(item['time_since_last_event'])

    return event, dtime, post, duration, lens


#@profile
def processBatchParticles(
    batch_of_seqs, idx_BOS, idx_EOS, idx_PAD, device=torch):

    r"""
    in this project, num_particles is always 1
    so much functionality of this method is not used 
    """

    batch_size = len(batch_of_seqs)
    num_particles = batch_of_seqs[0][2].size(0)

    max_len = -1
    max_len_sampling = -1
    max_len_obs = -1

    for i_batch, seq_with_particles in enumerate(batch_of_seqs):
        seq_len = seq_with_particles[0].size(1)
        seq_len_sampling = seq_with_particles[4].size(1)
        max_len = seq_len if seq_len > max_len else max_len
        max_len_sampling = seq_len_sampling if seq_len_sampling > max_len_sampling else max_len_sampling

    post = device.FloatTensor(batch_size, num_particles).fill_(0.0)
    duration = device.FloatTensor(batch_size, num_particles).fill_(0.0)

    r"""
    modify all the vocab size to the right idx : idx_BOS, idx_EOS, idx_EOS
    """
    event = device.LongTensor(
        batch_size, num_particles, max_len ).fill_(idx_PAD)
    time = device.FloatTensor(
        batch_size, num_particles, max_len ).fill_(0.0)

    dtime_sampling = device.FloatTensor(
        batch_size, num_particles, max_len_sampling ).fill_(0.0)
    index_of_hidden_sampling = device.LongTensor(
        batch_size, num_particles, max_len_sampling ).fill_(0)
    mask_sampling = device.FloatTensor(
        batch_size, num_particles, max_len_sampling ).fill_(0.0)
    # note we use batch_size as 0-dim here
    # because we need to flatten num_particles and max_len_sampling
    # in forward method of nhp


    for i_batch, seq_with_particles in enumerate(batch_of_seqs):
        seq_len = seq_with_particles[0].size(1)
        seq_len_sampling = seq_with_particles[4].size(1)

        event[i_batch, :, :seq_len] = seq_with_particles[0].clone()
        time[i_batch, :, :seq_len] = seq_with_particles[1].clone()

        post[i_batch, :] = seq_with_particles[2].clone()
        duration[i_batch, :] = seq_with_particles[3].clone()

        dtime_sampling[i_batch, :, :seq_len_sampling] = seq_with_particles[4].clone()
        mask_sampling[i_batch, :, :seq_len_sampling] = 1.0

        r"""
        since we now have an extra dimension i.e. batch_size
        we need to revise the index_of_hidden_sampling, that is,
        it should not be i_particle * (T+1) + j anymore
        what it should be ?
        consider when we flat the states, we make them to
        ( batch_size * num_particles * T+1 ) * hidden_dim
        and when we flatten the index_of_hidden_sampling, it is
        batch_size * num_particles * max_len_sampling
        so each entry should be :
        i_seq * ( num_particles * (T+1) ) + i_particle * (T+1) + j , that is
        for whatever value of element in this current matrix
        we should add it with i_seq * ( num_particles * (T+1) )
        this part is tricky so I should design sanity check
        """
        remainder = seq_with_particles[5] % ( seq_len - 1 )
        multiple = seq_with_particles[5] / ( seq_len - 1 )
        index_of_hidden_sampling[i_batch, :, :seq_len_sampling] = \
        i_batch * num_particles * (max_len - 1) + multiple * (max_len - 1) + remainder

    return Variable(event), Variable(time), \
    Variable(post), Variable(duration), \
    Variable(dtime_sampling), Variable(index_of_hidden_sampling), Variable(mask_sampling)


class DataProcessorBase(object):
    def __init__(self, mode, idx_BOS, idx_EOS, idx_PAD,
        sampling=1, use_gpu=False):
        self.mode = mode
        self.idx_BOS = idx_BOS
        self.idx_EOS = idx_EOS
        self.idx_PAD = idx_PAD
        self.sampling = sampling
        self.use_gpu = use_gpu
        self.device = torch.cuda if use_gpu else torch
        if self.mode == 'NeuralHawkes':
            self.funcBatch = processBatchParticles
            self.sampleForIntegral = sampleForIntegral
        else:
            raise Exception('Unknown mode: {}'.format(mode))

    def orgSeq(self, input, duration=None):
        return orgSeq(input,
            idx_BOS=self.idx_BOS, idx_EOS=self.idx_EOS, idx_PAD=self.idx_PAD,
            duration=duration)

    def processSeq(self, input, duration=None):
        return processSeq(input, duration,
            idx_BOS=self.idx_BOS, idx_EOS=self.idx_EOS, idx_PAD=self.idx_PAD,
            device=self.device)

    #@profile
    def processBatchParticles(self, input):
        return self.funcBatch(input,
            idx_BOS=self.idx_BOS, idx_EOS=self.idx_EOS, idx_PAD=self.idx_PAD,
            device=self.device)

    #@profile
    def processBatchSeqsWithParticles(self, input):
        r"""
        batch of seqs, where each seq is many particles (as torch tensors)
        """
        batch_of_seqs = []
        for seq in input:
            batch_of_seqs.append(self.sampleForIntegral(
                seq, sampling=self.sampling, device=self.device) )
        return self.processBatchParticles(batch_of_seqs)


class DataProcessorNeuralHawkes(DataProcessorBase):
    def __init__(self, *args, **kwargs):
        super(DataProcessorNeuralHawkes, self).__init__('NeuralHawkes', *args, **kwargs)


class LogWriter(object):

    def __init__(self, path, args):
        self.path = path
        self.args = args
        with open(self.path, 'w') as f:
            f.write("Training Log\n")
            f.write("Hyperparameters\n")
            for argname in self.args:
                f.write("{} : {}\n".format(argname, self.args[argname]))
            f.write("Checkpoints:\n")

    def checkpoint(self, to_write):
        with open(self.path, 'a') as f:
            f.write(to_write+'\n')

class LogReader(object):

    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            self.doc = f.read()

    def isfloat(self, str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    def casttype(self, str):
        res = None
        if str.isdigit():
            res = int(str)
        elif self.isfloat(str):
            res = float(str)
        elif str == 'True' or str == 'False':
            res = True if str == 'True' else False
        else:
            res = str
        return res

    def getArgs(self):
        block_args = self.doc.split('Hyperparameters\n')[-1]
        block_args = block_args.split('Checkpoints:\n')[0]
        lines_args = block_args.split('\n')
        res = {}
        for line in lines_args:
            items = line.split(' : ')
            res[items[0]] = self.casttype(items[-1])
        return res
