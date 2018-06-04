# -*- coding: utf-8 -*-
"""

Processers

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
from torch.autograd import Variable
#import torchvision.transforms as transforms

#from nhpf.utils.wrapper import Variable

#@profile
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

    r"""
    (at least) two more inputs : event_obs, dtime_obs
    we need to make indices to access the right backward hidden states
    and also compute the right dtime for each unobserved events
    note that: in the end, we just need
    p(s, u) and q(u | s)
    p(s) can be obtained by summing up p(s, u) for all different u
    """

    event, time, post, duration, lens, \
    event_obs, dtime_obs, dtime_backward, index_of_hidden_backward = input

    r"""
    comments about unobserved types
    """

    num_particles, T_plus_2 = event.size()
    _, T_obs_plus_2 = event_obs.size()
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

    cum_time_obs = dtime_obs.cumsum(dim=1)
    indices_mat_obs = device.LongTensor(num_particles, max_sampling).fill_(0)
    current_step_obs = device.LongTensor(num_particles, max_sampling).fill_(0)

    for j in range( T_obs_plus_2 - 1 ):

        bench_cum_time = cum_time_obs[:, j].unsqueeze(1).expand(
            num_particles, max_sampling)
        ceiling_cum_time = cum_time_obs[:, j+1].unsqueeze(1).expand(
            num_particles, max_sampling)
        indices_to_edit = (sampled_times > bench_cum_time) & (sampled_times <= ceiling_cum_time)

        dtime_backward_sampling[indices_to_edit] = \
        (ceiling_cum_time - sampled_times)[indices_to_edit]

        current_step_obs.fill_(j)
        index_of_hidden_backward_sampling[indices_to_edit] = \
        (indices_mat_obs + current_step_obs)[indices_to_edit]

    return event, time, post, duration, dtime_sampling, index_of_hidden_sampling, \
    event_obs, dtime_obs, dtime_backward, index_of_hidden_backward, \
    dtime_backward_sampling, index_of_hidden_backward_sampling
    # idx of output :
    # event 0, time 1, post 2, duration 3,
    # dtime_sampling 4, index_of_hidden_sampling 5,
    # event_obs 6, dtime_obs 7,
    # dtime_backward 8, index_of_hidden_backward 9, \
    # dtime_backward_sampling 10, index_of_hidden_backward_sampling 11


def getSeq(input, weights, idx_BOS, idx_EOS, idx_PAD):
    r"""
    use the output of agent.sample_particles and weights
    to get the infered seq with highest weight
    """
    event, dtime, post, duration, lens, _, _, _, _ = input
    _, id_high = torch.max(weights, 0)
    len_seq = lens[id_high]
    dtime_cum = dtime.cumsum(dim=1)

    seq_in_torch = event[id_high, 1: len_seq + 1]
    if len_seq > 0: assert (seq_in_torch < idx_BOS).all(), "event type id should < BOS, but these are {} and corresponding tensor is {} and len is {}".format(seq_in_torch, event[id_high, :], len_seq)
    assert event[id_high, 0] == idx_BOS, "starting with BOS"
    assert event[id_high, len_seq + 1] == idx_EOS, "ending with EOS"

    out = []
    for i in range(len_seq):
        out.append(
            {
                'type_event': int(event[id_high, i + 1]),
                'time_since_last_event': float(dtime[id_high, i + 1]),
                'time_since_start': float(dtime_cum[id_high, i + 1])
            }
        )

    return out


def orgSeq(one_seq, idx_BOS, idx_EOS, idx_PAD, missing_types, duration):
    r"""
    augment BOS and EOS to the seq
    BOS and EOS are treated as observations
    and then create a seq of observed events
    caution to dtime of observed
    """
    one_seq_new, one_seq_obs = [], []
    duration = one_seq[-1]['time_since_start'] if duration is None else duration
    one_seq_new.append(
        {
            'type_event': idx_BOS, 'time_since_last_event': 0.0,
            'time_since_start': 0.0
        }
    )
    one_seq_obs.append(
        {
            'type_event': idx_BOS, 'time_since_last_event': 0.0,
            'time_since_start': 0.0
        }
    )
    for item in one_seq:
        one_seq_new.append(item)
        if item['type_event'] not in missing_types:
            time_temp = one_seq_obs[-1]['time_since_start']
            one_seq_obs.append(
                {
                    'type_event': item['type_event'],
                    'time_since_start': item['time_since_start'],
                    'time_since_last_event': item['time_since_start'] - time_temp
                }
            )
    one_seq_new.append(
        {
            'type_event': idx_EOS, 'time_since_last_event': 0.0,
            'time_since_start': duration
        }
    )
    time_gap = duration - one_seq_obs[-1]['time_since_start']
    one_seq_obs.append(
        {
            'type_event': idx_EOS, 'time_since_last_event':time_gap,
            'time_since_start': duration
        }
    )
    return one_seq_new, one_seq_obs


def processSeq(
    input, duration, idx_BOS, idx_EOS, idx_PAD, missing_types=[], device=torch):
    r"""
    make tensors for one seq, given which types will be missing in eval
    """
    one_seq, one_seq_obs = orgSeq(
        input, idx_BOS, idx_EOS, idx_PAD, missing_types, duration )
    len_seq = len(one_seq)
    len_seq_obs = len(one_seq_obs)

    #assert one_seq[-1]['time_since_start'] == one_seq_obs[-1]['time_since_start']
    #print("print one seq")
    #print(one_seq)
    #print("print one seq obs")
    #print(one_seq_obs)

    duration = device.FloatTensor(1).fill_(
        float(one_seq[-1]['time_since_start']) )
    post = device.FloatTensor(1).fill_(0.0)
    lens = device.LongTensor(1).fill_(len_seq - 2)

    event = device.LongTensor( 1, len_seq ).fill_(idx_PAD)
    dtime = device.FloatTensor( 1, len_seq ).fill_(0.0)

    event_obs = device.LongTensor( 1, len_seq_obs ).fill_(idx_PAD)
    dtime_obs = device.FloatTensor( 1, len_seq_obs ).fill_(0.0)

    dtime_backward = device.FloatTensor( 1, len_seq ).fill_(0.0)
    index_of_hidden_backward = device.LongTensor( 1, len_seq ).fill_(0)

    i_item_obs = 0

    for i_item, item in enumerate(one_seq):

        event[:, i_item] = int(item['type_event'])
        dtime[:, i_item] = float(item['time_since_last_event'])

        if item['type_event'] in missing_types:
            # missing types NOT include BOS EOS PAD
            # if this event ought to be missing in eval
            # track its associated soonest observed events
            #print("i item obs {} and len is {}".format(i_item_obs, len(one_seq_obs)))
            #print("i item is {} and len is {}".format(i_item, len(one_seq)))
            #print("time_since_start is {}".format(item['time_since_start']))
            while one_seq_obs[i_item_obs]['time_since_start'] < item['time_since_start']:
            #    print(one_seq_obs[i_item_obs]['time_since_start'])
                i_item_obs += 1
            #    print(i_item_obs)
            dtime_backward[:, i_item] = float(
                one_seq_obs[i_item_obs]['time_since_start'] - item['time_since_start'])
            index_of_hidden_backward[:, i_item] = int(i_item_obs - 1)
            # -1 cuz when we go through seq obs in reverse order
            # we stop at 1-st event, so 1-st obs event has index 0

    # consider adding assertion : index_of_hidden_backward >= 0
    for i_item_obs, item in enumerate(one_seq_obs):

        event_obs[:, i_item_obs] = int(item['type_event'])
        dtime_obs[:, i_item_obs] = float(item['time_since_last_event'])

    return event, dtime, post, duration, lens, \
    event_obs, dtime_obs, dtime_backward, index_of_hidden_backward


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
        seq_len_obs = seq_with_particles[6].size(1)
        max_len = seq_len if seq_len > max_len else max_len
        max_len_sampling = seq_len_sampling if seq_len_sampling > max_len_sampling else max_len_sampling
        max_len_obs = seq_len_obs if seq_len_obs > max_len_obs else max_len_obs

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

    event_obs = device.LongTensor(
        batch_size, 1, max_len_obs ).fill_(idx_PAD)
    time_obs = device.FloatTensor(
        batch_size, 1, max_len_obs ).fill_(0.0)

    dtime_backward = device.FloatTensor(
        batch_size, num_particles, max_len ).fill_(0.0)
    index_of_hidden_backward = device.LongTensor(
        batch_size, num_particles, max_len ).fill_(0)

    dtime_backward_sampling = device.FloatTensor(
        batch_size, num_particles, max_len_sampling ).fill_(0.0)
    index_of_hidden_backward_sampling = device.LongTensor(
        batch_size, num_particles, max_len_sampling ).fill_(0)

    for i_batch, seq_with_particles in enumerate(batch_of_seqs):
        seq_len = seq_with_particles[0].size(1)
        seq_len_sampling = seq_with_particles[4].size(1)
        seq_len_obs = seq_with_particles[6].size(1)

        event[i_batch, :, :seq_len] = seq_with_particles[0].clone()
        time[i_batch, :, :seq_len] = seq_with_particles[1].clone()

        post[i_batch, :] = seq_with_particles[2].clone()
        duration[i_batch, :] = seq_with_particles[3].clone()

        dtime_sampling[i_batch, :, :seq_len_sampling] = seq_with_particles[4].clone()
        mask_sampling[i_batch, :, :seq_len_sampling] = 1.0

        event_obs[i_batch, :, :seq_len_obs] = seq_with_particles[6].clone()
        time_obs[i_batch, :, :seq_len_obs] = seq_with_particles[7].clone()

        dtime_backward[i_batch, :, :seq_len] = seq_with_particles[8].clone()
        dtime_backward_sampling[i_batch, :, :seq_len_sampling] = \
        seq_with_particles[10].clone()

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

        remainder = seq_with_particles[9] % ( seq_len_obs - 1 )
        multiple = seq_with_particles[9] / ( seq_len_obs - 1 )
        index_of_hidden_backward[i_batch, :, :seq_len] = \
        i_batch * 1 * (max_len_obs - 1) + multiple * (max_len_obs - 1) + remainder

        remainder = seq_with_particles[11] % ( seq_len_obs - 1 )
        multiple = seq_with_particles[11] / ( seq_len_obs - 1 )
        index_of_hidden_backward_sampling[i_batch, :, :seq_len_sampling] = \
        i_batch * 1 * (max_len_obs - 1) + multiple * (max_len_obs - 1) + remainder

    return Variable(event), Variable(time), \
    Variable(post), Variable(duration), \
    Variable(dtime_sampling), Variable(index_of_hidden_sampling), Variable(mask_sampling), \
    Variable(event_obs), Variable(time_obs), \
    Variable(dtime_backward), Variable(index_of_hidden_backward), \
    Variable(dtime_backward_sampling), Variable(index_of_hidden_backward_sampling)

    #return Variable(event.detach().data), Variable(time.detach().data), \
    #Variable(post.detach().data), Variable(duration.detach().data), \
    #Variable(dtime_sampling.detach().data), Variable(index_of_hidden_sampling.detach().data), Variable(mask_sampling.detach().data), \
    #Variable(event_obs.detach().data), Variable(time_obs.detach().data), \
    #Variable(dtime_backward.detach().data), Variable(index_of_hidden_backward.detach().data), \
    #Variable(dtime_backward_sampling.detach().data), Variable(index_of_hidden_backward_sampling.detach().data)

class DataProcessorBase(object):
    def __init__(self, mode, idx_BOS, idx_EOS, idx_PAD,
        sampling=1, use_gpu=False, missing_types=[]):
        self.mode = mode
        self.idx_BOS = idx_BOS
        self.idx_EOS = idx_EOS
        self.idx_PAD = idx_PAD
        self.sampling = sampling
        self.use_gpu = use_gpu
        self.device = torch.cuda if use_gpu else torch
        self.missing_types = missing_types
        #self.batch_size = batch_size
        if self.mode == 'NeuralHawkes':
            self.funcBatch = processBatchParticles
            self.sampleForIntegral = sampleForIntegral
        else:
            raise Exception('Unknown mode: {}'.format(mode))

    def getSeq(self, input, weights):
        return getSeq(input, weights,
            idx_BOS=self.idx_BOS, idx_EOS=self.idx_EOS, idx_PAD=self.idx_PAD)

    def orgSeq(self, input, duration=None):
        return orgSeq(input,
            idx_BOS=self.idx_BOS, idx_EOS=self.idx_EOS, idx_PAD=self.idx_PAD,
            missing_types=self.missing_types, duration=duration)

    def processSeq(self, input, duration=None):
        return processSeq(input, duration,
            idx_BOS=self.idx_BOS, idx_EOS=self.idx_EOS, idx_PAD=self.idx_PAD,
            missing_types=self.missing_types, device=self.device)

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

class DataProcessorNaive(DataProcessorBase):
    def __init__(self, *args, **kwargs):
        super(DataProcessorNaive, self).__init__('Naive', *args, **kwargs)


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
