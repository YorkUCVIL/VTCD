'''
Special functions with altered gradient behavior.
Created by Basile Van Hoorick, Jun 2022.
'''

from __init__ import *
import torch


class StraightThroughThreshold(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        y = torch.round(input)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
