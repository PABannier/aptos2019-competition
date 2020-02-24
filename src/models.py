import numpy as np

import torch 
import torch.nn as nn
from torch.nn import functional as F

import pretrainedmodels


class ResNet34(nn.Module):
    def __init__(self, pretrained):

        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=False)

        self.l = nn.Linear(512, 1)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        return self.l(x)