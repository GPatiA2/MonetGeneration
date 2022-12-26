import sys, os

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
import Blocks as bk

class Generator(nn.Module):
    '''
    Generator Class
    A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to 
    transform an input image into an image from the other class, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(Generator, self).__init__()
        self.upfeature = bk.FeatureMapBlock(input_channels, hidden_channels*2)
        self.upfeature2 = bk.FeatureMapBlock(hidden_channels*2, hidden_channels)
        self.contract1 = bk.ContractingBlock(hidden_channels)
        self.contract2 = bk.ContractingBlock(hidden_channels * 2)
        self.contract3 = bk.ContractingBlock(hidden_channels * 4)
        res_mult = 8

        self.resBlocks = bk.ResidualBlockss(hidden_channels, res_mult)
        self.resBlocks2 = bk.ResidualBlockss(hidden_channels, res_mult)

        self.expand1 = bk.ExpandingBlock(hidden_channels * 8)
        self.expand2 = bk.ExpandingBlock(hidden_channels * 4)
        self.expand3 = bk.ExpandingBlock(hidden_channels * 2)
        self.downfeature = bk.FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        Function for completing a forward pass of Generator: 
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x00 = self.upfeature(x)
        x0 = self.upfeature2(x00)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x3 = self.resBlocks(x3)
        x11 = self.resBlocks2(x3)
        x11 = self.expand1(x11)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)