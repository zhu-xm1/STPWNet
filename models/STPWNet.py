#!/usr/bin/env python
"""
@author : liang
@email  : dear_dao@163.com
@time   : 11/6/2019 9:34 AM
@desc   : pw_test.py.py
"""

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BN_MOMENTUM = 0.1


class PWBlock(nn.Module):
    expansion = 1

    def __init__(self, num_rank, group_channels, out_channels, dropout=0.,downsample=None):
        super(PWBlock, self).__init__()

        inner_channels=group_channels*self.expansion
        self.bn1 = nn.BatchNorm2d(group_channels, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(group_channels, inner_channels, kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(inner_channels, out_channels,kernel_size=1,bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.dropout=dropout
        self.downsample = downsample

        self.inplanes = num_rank*group_channels
        self.outplanes = self.inplanes + group_channels
        self.in_channels = group_channels

        print('==>',self.inplanes,self.outplanes)

    def forward(self, x):
        inputs = x[:,self.inplanes:self.outplanes,:,:]
        out = self.conv1(self.relu(self.bn1(inputs)))
        if self.dropout>0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out=self.conv2(self.relu(self.bn2(out)))
        if self.dropout>0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        residual = x[:,self.outplanes:,:,:]+out[:,:-self.in_channels,:,:]
        out = torch.cat([x[:,:self.outplanes,:,:],residual,out[:,-self.in_channels:,:,:]],1)

        return out

class PWModule(nn.Module):
    def __init__(self, block, num_group,in_channels):
        super(PWModule, self).__init__()

        self.out_channels = in_channels*2

        num_group_channels=int(in_channels//num_group)

        self.layers=self._make_layer(block,num_group,num_group_channels,in_channels)

        print('---'*10)

    def _make_layer(self,block,num_layer,num_group_channels,num_chanels):
        layers=[]
        for i in range(num_layer):
            layers.append(block(i,num_group_channels,num_chanels))
        return nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class PWNetUnit(nn.Module):
    def __init__(self,in_flow,out_flow,init_channels=64,num_group=8,droprate=0):
        super(PWNetUnit, self).__init__()
        if in_flow==0:return
        self.conv =nn.Conv2d(in_flow,init_channels,kernel_size=3,padding=1,bias=False)

        self.module=PWModule(PWBlock, num_group,init_channels)
        out_channels=self.module.out_channels
        self.trans = TransitionBlock(out_channels, out_flow)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return self.trans(self.module(out))

class PWNet(nn.Module):
    def __init__(self,in_flow,nb_flow,init_channels=[128],group_list=[8,8,8],droprate=0):
        super(PWNet, self).__init__()

        self.in_flow=in_flow

        self.close_feature1 = PWNetUnit(in_flow,in_flow,init_channels[0], group_list[0])

        self.close_feature2 = PWNetUnit(in_flow, nb_flow, init_channels[0], group_list[0])

        #self.close_feature3 = PWNetUnit(in_flow, nb_flow, init_channels[0], group_list[0])


        # self.period_feature = PWNetUnit(in_flow[1],nb_flow,init_channels[1], group_list[1])
        # self.trend_feature = PWNetUnit(in_flow[2],nb_flow,init_channels[2], group_list[2])


    def forward(self, inputs):

        out = self.close_feature1(inputs[0])

        out = self.close_feature2(out)

       # out = self.close_feature3(out)

        # if self.in_flow[1] > 0:
        #     out += self.period_feature(inputs[1])
        # if self.in_flow[2] > 0:
        #     out += self.trend_feature(inputs[2])

        return torch.sigmoid(out)

if __name__=='__main__':
    from thop import clever_format
    from thop import profile

    net = PWNet(3,1)
    inputs1 = torch.Tensor(np.random.random((1, 3, 16, 8)))
    inputs = [inputs1]
    flops, params = profile(net, inputs=(inputs,))
    print('==>',flops,params)
    flops, params = clever_format([flops, params], "%.3f")
    print('==>',flops,params)

    out = net(inputs)
    print('shape:',out.shape)