# -*- coding: utf-8 -*-
"""

neural Hawkes process (nhp) and continuous-time LSTM

@author: hongyuan
"""

import math
import warnings
import time
import numpy
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from nhpf.utils.wrapper import Arange # Variable, Arange


class CTLSTMCell(nn.Module):

    def __init__(self, hidden_dim, beta=1.0, device=torch):
        super(CTLSTMCell, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(hidden_dim * 2, hidden_dim * 7, bias=True)

        self.beta = beta

    def forward(
        self, rnn_input,
        hidden_t_i_minus, cell_t_i_minus, cell_bar_im1):

        dim_of_hidden = rnn_input.dim() - 1

        input_i = torch.cat((rnn_input, hidden_t_i_minus), dim=dim_of_hidden)
        output_i = self.linear(input_i)

        gate_input, \
        gate_forget, gate_output, gate_pre_c, \
        gate_input_bar, gate_forget_bar, gate_decay = output_i.chunk(
            7, dim_of_hidden)

        gate_input = F.sigmoid(gate_input)
        gate_forget = F.sigmoid(gate_forget)
        gate_output = F.sigmoid(gate_output)
        gate_pre_c = F.tanh(gate_pre_c)
        gate_input_bar = F.sigmoid(gate_input_bar)
        gate_forget_bar = F.sigmoid(gate_forget_bar)
        gate_decay = F.softplus(gate_decay, beta=self.beta)

        cell_i = gate_forget * cell_t_i_minus + gate_input * gate_pre_c
        cell_bar_i = gate_forget_bar * cell_bar_im1 + gate_input_bar * gate_pre_c

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(self, cell_i, cell_bar_i, gate_decay, gate_output, dtime):
        # no need to consider extra_dim_particle here
        # cuz this function is applicable to any # of dims
        if dtime.dim() < cell_i.dim():
            dtime = dtime.unsqueeze(cell_i.dim()-1).expand_as(cell_i)

        cell_t_ip1_minus = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(
            -gate_decay * dtime)
        hidden_t_ip1_minus = gate_output * F.tanh(cell_t_ip1_minus)

        return cell_t_ip1_minus, hidden_t_ip1_minus


class NeuralHawkes(nn.Module):

    def __init__(
        self, total_num,
        hidden_dim=32, time_eps=0.001, time_max=1000.0, beta=1.0, #strength=0.0,
        use_gpu=False):
        super(NeuralHawkes, self).__init__()

        self.total_num = total_num

        self.hidden_dim = hidden_dim

        self.idx_BOS = self.total_num
        self.idx_EOS = self.total_num + 1
        self.idx_PAD = self.total_num + 2

        self.time_eps = time_eps
        self.time_max = time_max
        self.beta = beta

        self.use_gpu = use_gpu
        self.device = torch.cuda if use_gpu else torch

        self.Emb = nn.Embedding(
            self.total_num + 3, hidden_dim)

        self.rnn_cell = CTLSTMCell(hidden_dim, beta=beta, device=self.device)
        self.hidden_lambda = nn.Linear(
            hidden_dim, self.total_num, bias=False)

        self.init_h = Variable(
            self.device.FloatTensor(hidden_dim).fill_(0.0) )
        self.init_c = Variable(
            self.device.FloatTensor(hidden_dim).fill_(0.0) )
        self.init_cb = Variable(
            self.device.FloatTensor(hidden_dim).fill_(0.0) )

        self.eps = numpy.finfo(float).eps
        self.max = numpy.finfo(float).max


    def getStates(self, event, dtime):
        r"""
        go through the sequences and get all the states and gates
        we assume there is always a dim for particles but it can be 1
        """
        batch_size, num_particles, T_plus_2 = event.size()
        cell_t_i_minus = self.init_c.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.hidden_dim)
        cell_bar_im1 = self.init_cb.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.hidden_dim)
        hidden_t_i_minus = self.init_h.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_particles, self.hidden_dim)

        all_cell, all_cell_bar = [], []
        all_gate_output, all_gate_decay = [], []
        all_hidden = []
        all_hidden_after_update = []

        for i in range(T_plus_2 - 1):
            # only BOS to last event update LSTM
            # <s> CT-LSTM

            emb_i = self.Emb(event[:, :, i ])
            dtime_i = dtime[:, :, i + 1 ] # need to carefully check here
            r"""
            i or i + 1, that is a question
            i : seems working because the log-prob increasing from -2 to -1
            i + 1 : seems not working because log-prob quickly stuck in -2
            but technically, i believe i + 1 is correct, because
            after updating LSTM with i-th event, we should use the dtime of
            i + 1 -th event to decay the hidden states (used for next step)
            how to decide ? --- benchmark it !!!
            i can use Xutai's PyTorch code to run and test on same dataset
            see the results
            and then run my code on the same data --- whichever can match !!!
            """

            #import ipdb; ipdb.set_trace()

            cell_i, cell_bar_i, gate_decay_i, gate_output_i = self.rnn_cell(
                emb_i, hidden_t_i_minus, cell_t_i_minus, cell_bar_im1
            )
            _, hidden_t_i_plus = self.rnn_cell.decay(
                cell_i, cell_bar_i, gate_decay_i, gate_output_i,
                torch.zeros_like(dtime_i)
            )
            cell_t_ip1_minus, hidden_t_ip1_minus = self.rnn_cell.decay(
                cell_i, cell_bar_i, gate_decay_i, gate_output_i,
                dtime_i
            )
            all_cell.append(cell_i)
            all_cell_bar.append(cell_bar_i)
            all_gate_decay.append(gate_decay_i)
            all_gate_output.append(gate_output_i)
            all_hidden.append(hidden_t_ip1_minus)
            all_hidden_after_update.append(hidden_t_i_plus)
            cell_t_i_minus = cell_t_ip1_minus
            cell_bar_im1 = cell_bar_i
            hidden_t_i_minus = hidden_t_ip1_minus
            # </s> CT-LSTM
        # these tensors shape : batch_size, num_particles, T+1, hidden_dim
        # cells and gates right after BOS, 1st event, ..., N-th event
        # hidden right before 1st event, ..., N-th event, End event (PAD)
        all_cell = torch.stack( all_cell, dim=2)
        all_cell_bar = torch.stack( all_cell_bar, dim=2)
        all_gate_decay = torch.stack( all_gate_decay, dim=2)
        all_gate_output = torch.stack( all_gate_output, dim=2)
        all_hidden = torch.stack( all_hidden, dim=2 )
        all_hidden_after_update = torch.stack( all_hidden_after_update, dim=2)
        #assert all_gate_decay.data.cpu().numpy().all() >= 0.0, "Decay > 0"
        return batch_size, num_particles, T_plus_2, \
        all_cell, all_cell_bar, all_gate_decay, all_gate_output, \
        all_hidden, all_hidden_after_update

    def getTarget(self, event, dtime):
        r"""
        make target variable and masks
        """
        batch_size, num_particles, T_plus_2 = event.size()
        mask_complete = torch.ones_like(dtime[:, :, 1:])
        target_data = event[:, :, 1:].detach().data.clone()

        mask_complete[target_data >= self.total_num] = 0.0
        target_data[target_data >= self.total_num] = 0
        target = Variable( target_data )

        return target, mask_complete


    def getLogLambda(
        self, batch_size, num_particles, T_plus_2,
        target, mask_complete, all_hidden):
        r"""
        we output log_lambda for one case:
        1. complete, including obs, unobs (note that for now we don't have unobs)
        """

        all_lambda= F.softplus(self.hidden_lambda(all_hidden), beta=self.beta)
        log_lambda= torch.log(all_lambda+ self.eps)


        log_lambda_target = log_lambda.view(
            batch_size * num_particles * (T_plus_2 - 1), self.total_num
        )[
            Arange(batch_size * num_particles * (T_plus_2 - 1), device=self.device),
            target.view( batch_size * num_particles * (T_plus_2 - 1) )
        ].view(batch_size, num_particles, T_plus_2 - 1)

        log_lambda_target_complete = log_lambda_target * mask_complete

        lambda_sum_complete = torch.sum(all_lambda, dim=3)

        log_lambda_sum_complete = torch.log(lambda_sum_complete + self.eps)
        log_lambda_sum_complete *= mask_complete

        return log_lambda_target_complete, log_lambda_sum_complete


    def getSampledStates(
        self, dtime_sampling, index_of_hidden_sampling,
        all_cell, all_cell_bar, all_gate_output, all_gate_decay):
        r"""
        we output the sampled hidden states of the left-to-right machine
        """
        r"""
        states shape : batch_size * num_particles * T+1 * hidden_dim
        dtime_sampling : batch_size * num_particles * max_len_sampling
        index_of_hidden_sampling : batch_size * num_particles * max_len_sampling
        """
        batch_size, num_particles, T_plus_1, _ = all_cell.size()
        _, _, max_len_sampling = dtime_sampling.size()

        all_cell_sampling = all_cell.view(
            batch_size * num_particles * T_plus_1, self.hidden_dim )[
                index_of_hidden_sampling.view(-1), :].view(
                    batch_size, num_particles, max_len_sampling, self.hidden_dim)
        all_cell_bar_sampling = all_cell_bar.view(
            batch_size * num_particles * T_plus_1, self.hidden_dim )[
                index_of_hidden_sampling.view(-1), :].view(
                    batch_size, num_particles, max_len_sampling, self.hidden_dim)
        all_gate_output_sampling = all_gate_output.view(
            batch_size * num_particles * T_plus_1, self.hidden_dim )[
                index_of_hidden_sampling.view(-1), :].view(
                    batch_size, num_particles, max_len_sampling, self.hidden_dim)
        all_gate_decay_sampling = all_gate_decay.view(
            batch_size * num_particles * T_plus_1, self.hidden_dim )[
                index_of_hidden_sampling.view(-1), :].view(
                    batch_size, num_particles, max_len_sampling, self.hidden_dim)

        cy_sample, hy_sample = self.rnn_cell.decay(
            all_cell_sampling, all_cell_bar_sampling,
            all_gate_decay_sampling, all_gate_output_sampling,
            dtime_sampling
        )

        return hy_sample


    def getIntegral(
        self, hy_sample, mask_sampling, duration):
        r"""
        we output integral for one case:
        1. complete
        """
        r"""
        mask_sampling : batch_size * num_particles * max_len_sampling
        duration : batch_size * num_particles
        """
        lambda_sample = F.softplus(
            self.hidden_lambda(hy_sample), beta=self.beta )

        lambda_sample_complete_sum = lambda_sample.sum(3)
        lambda_sample_complete_mean = torch.sum(
            lambda_sample_complete_sum * mask_sampling, dim=2 ) / torch.sum(
            mask_sampling, dim=2 )
        integral_complete = lambda_sample_complete_mean * duration

        return integral_complete


    def forward(self, input, mode=1, weight=None):

        event, dtime, post, duration, \
        dtime_sampling, index_of_hidden_sampling, mask_sampling = input

        r"""
        event, dtime : batch_size, M, T+2
        post(erior of incomplete unobserved) : batch_size, M (not used)
        duration : batch_size, M
        dtime_sampling : batch_size, M, T_sample
        mode ==
        1 : complete log likelihood
        Note: for log_likelihood, check if weight==None
        if False, do arithmetic average over particles
        """

        batch_size, num_particles, T_plus_2, \
        all_cell, all_cell_bar, all_gate_decay, all_gate_output, \
        all_hidden, all_hidden_after_update = self.getStates(event, dtime)

        target, mask_complete = self.getTarget(
            event, dtime)

        sampled_hidden = self.getSampledStates(
            dtime_sampling, index_of_hidden_sampling,
            all_cell, all_cell_bar, all_gate_output, all_gate_decay
        )

        # <s> \lambda_{k_i}(t_i) for scheduled actions
        log_lambda_target_complete, log_lambda_sum_complete = self.getLogLambda(
            batch_size, num_particles, T_plus_2,
            target, mask_complete, all_hidden)
        # batch_size * num_particles * T_plus_2-1
        # </s> \lambda_{k_i}(t_i) for scheduled actions


        # <s> int_{0}^{T} lambda_sum dt for scheduled actions
        integral_complete = self.getIntegral(
            sampled_hidden, mask_sampling, duration)
        # batch_size * num_particles
        # </s> int_{0}^{T} lambda_sum dt for scheduled actions


        # <s> log likelihood computation
        logP_complete = log_lambda_target_complete.sum(2) - integral_complete
        # batch_size * num_particles
        # </s> log likelihood computation

        r"""
        for any seq in a batch, there are some particles (can be 1)
        there are (normalized) weight for each particle
        if weight is not given, i.e. None
        we assume uniform weight, so average particles
        """

        if weight is None:
            weight = Variable(
                self.device.FloatTensor(
                    batch_size, num_particles).fill_(1.0) )
            weight = weight / torch.sum(weight, dim=1, keepdim=True)

        # For now we only have one mode
        if mode == 1:
            # complete log likelihood
            objective = -torch.sum( logP_complete * weight )
            num_events = torch.sum( mask_complete )
        else:
            raise Exception( "Unknown mode : ".format(mode) )

        return objective, num_events
