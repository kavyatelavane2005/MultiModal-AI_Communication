#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pose-TGCN Model for Sign Language Recognition.

Adapted from the WLASL repository (https://github.com/dxli94/WLASL)
Modified for CPU-only inference.
"""
from __future__ import absolute_import
from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolution_att(nn.Module):
    """Graph Convolution layer with learnable adjacency matrix.
    
    Uses a learnable 55x55 attention matrix instead of a fixed
    adjacency matrix, allowing the model to learn inter-keypoint
    relationships during training.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(55, 55))  # 55 keypoints
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)   # H @ W
        output = torch.matmul(self.att, support)      # A @ (H @ W)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    """Residual Graph Convolution Block.
    
    Two GraphConvolution layers with BatchNorm, Tanh activation,
    Dropout, and a residual skip connection.
    """

    def __init__(self, in_features, p_dropout, bias=True, is_resi=True):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_resi = is_resi

        self.gc1 = GraphConvolution_att(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)

        self.gc2 = GraphConvolution_att(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        if self.is_resi:
            return y + x
        else:
            return y


class GCN_muti_att(nn.Module):
    """Multi-stage GCN with learnable attention for sign language recognition.
    
    Architecture:
        1. Initial GCN layer: input_feature -> hidden_feature
        2. num_stage residual GC_Blocks
        3. Mean pooling across 55 keypoints
        4. Final FC layer: hidden_feature -> num_class
    
    Args:
        input_feature: Input feature dimension (num_samples * 2 for x,y coords)
        hidden_feature: Hidden layer dimension
        num_class: Number of sign language classes to predict
        p_dropout: Dropout probability
        num_stage: Number of residual GCN blocks
        is_resi: Whether to use residual connections
    """

    def __init__(self, input_feature, hidden_feature, num_class,
                 p_dropout, num_stage=1, is_resi=True):
        super(GCN_muti_att, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution_att(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(55 * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, is_resi=is_resi))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch, 55, input_feature)
               where 55 = number of keypoints
               and input_feature = num_samples * 2 (x,y concatenated across time)
        
        Returns:
            Tensor of shape (batch, num_class) — raw logits (no softmax)
        """
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        # Mean pooling across keypoints: (batch, 55, hidden) -> (batch, hidden)
        out = torch.mean(y, dim=1)
        out = self.fc_out(out)

        return out


if __name__ == '__main__':
    # Quick test
    num_samples = 50
    model = GCN_muti_att(
        input_feature=num_samples * 2,
        hidden_feature=64,
        num_class=100,
        p_dropout=0.3,
        num_stage=20
    )
    x = torch.ones([2, 55, num_samples * 2])
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {model(x).shape}")
    print("Model test passed!")
