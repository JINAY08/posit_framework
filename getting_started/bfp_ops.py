

import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import pdb
import itertools as it
import logging
import unittest

def unpack_bfp_args(kwargs):
    """
    Set up the bfp arguments
    """
    bfp_args = {}
    bfp_argn = [('num_format', 'fp32'),
                ('rounding_mode', 'stoc'),
                ('epsilon', 1e-8),
                ('mant_bits', 0),
                ('bfp_tile_size', 0),
                ('weight_mant_bits', 0),
                ('device', 'cpu')]

    for arg, default in bfp_argn:
        if arg in kwargs:
            bfp_args[arg] = kwargs[arg]
            del kwargs[arg]
        else:
            bfp_args[arg] = default
    return bfp_args

class PositConv2d(torch.nn.Conv2d):
    """
    posit convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.bfp_args = unpack_bfp_args(kwargs)
        self.num_format = self.bfp_args['num_format']

    def conditions_l1(self, xx):
        conditions, values = [], []
        x = torch.where(xx < 0, -1*xx, xx)
        for e in range(16):
            for j in range(8):
                if(j==0): 
                    conditions.append(torch.logical_and(x > 0, x < (1.0625/pow(2,16))))
                    mid = 1/pow(2,16)
                    mid = torch.where(xx < 0, -1*mid, mid)
                    values.append(mid)
                else:
                    lower_bound = (1.0625 + 0.125 * (j - 1)) / pow(2, 16 - e)
                    upper_bound = (1.0625 + 0.125 * j) / pow(2, 16 - e)
                    mid = (lower_bound+upper_bound) / 2
                    condition = torch.logical_and(x > lower_bound, x < upper_bound)
                    conditions.append(condition)
                    mid = torch.where(xx < 0, -1*mid, mid)
                    values.append(mid)
        return conditions, values

    def conditions_g1(self, xx):
        conditions, values = [], []
        x = torch.where(xx < 0, -1*xx, xx)
        for e in range(16):
            for j in range(8):
                if(j==8): 
                    conditions.append(x > 1.8125*(pow(2,15)))
                    mid = 1.875*(pow(2,15))
                    mid = torch.where(xx < 0, -1*mid, mid)
                    values.append(mid)
                else:
                    lower_bound = (1.0625 + 0.125 * (j - 1)) * pow(2, e)
                    upper_bound = (1.0625 + 0.125 * j) * pow(2, e)
                    mid = (lower_bound+upper_bound) / 2
                    condition = torch.logical_and(x > lower_bound, x < upper_bound)
                    conditions.append(condition)
                    mid = torch.where(xx < 0, -1*mid, mid)
                    values.append(mid)
        return conditions, values

    def forward(self, input):
        if self.num_format == 'posit':
            # print('going')
            conditions_l1, values_l1 = self.conditions_l1(input)
            conditions_g1, values_g1 = self.conditions_g1(input)
            conditions_l1_w, values_l1_w = self.conditions_l1(self.weight)
            conditions_g1_w, values_g1_w = self.conditions_g1(self.weight)
            for i in range(len(conditions_l1)):
                input = torch.where(torch.abs(input) < 1, torch.where(conditions_l1[i], values_l1[i], input), torch.where(conditions_g1[i], values_g1[i], input))
            for i in range(len(conditions_l1_w)):
                self.weight = torch.nn.Parameter(torch.where(torch.abs(self.weight) < 1, torch.where(conditions_l1_w[i], values_l1_w[i], self.weight), torch.where(conditions_g1_w[i], values_g1_w[i], self.weight)))
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        elif self.num_format == 'fp32':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
            

        else:
            raise NotImplementedError('NumFormat not implemented')


class PositLinear(torch.nn.Linear):
    """
    posit linear layer
    """
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias)
        self.bfp_args = unpack_bfp_args(kwargs)
        self.num_format = self.bfp_args['num_format']
        self.linear_op = _get_bfp_op(F.linear, 'linear', self.bfp_args)

    def forward(self, input):
        if self.num_format == 'fp32' or self.num_format == 'posit':
            # print('going')
            # self.weight = torch.nn.Parameter(torch.where(self.weight < 0, torch.tensor(-1).float(), torch.tensor(0).float()))
            return F.linear(input, self.weight, self.bias)
        elif self.num_format == 'bfloat16':
            weight = self.weight.to(torch.bfloat16)
            input = input.to(torch.bfloat16)
            if(self.bias is not None):
                bias = self.bias.to(torch.bfloat16)
            else:
                bias = None

            return F.linear(input, self.weight, self.bias)

        else:
            raise NotImplementedError('NumFormat not implemented')



if __name__ == '__main__':
    unittest.main(verbosity=2)

