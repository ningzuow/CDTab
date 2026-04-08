# cdtab/modules/contrastive_module.py
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionMLP(nn.Module): 

    def __init__(self, in_dim, out_dim=256, hidden_dim=1024, num_layers=3, use_bn=True, normalize=True):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden_dim, bias=True)] 
        if use_bn:
            layers += [nn.BatchNorm1d(hidden_dim)] 
        layers += [nn.SiLU(inplace=True)] 

        if num_layers == 3:
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=True)]
            if use_bn:
                layers += [nn.BatchNorm1d(hidden_dim)]
            layers += [nn.SiLU(inplace=True)]

        layers += [nn.Linear(hidden_dim, out_dim, bias=True)]
        if use_bn:
            layers += [nn.BatchNorm1d(out_dim, affine=False)]

        self.net = nn.Sequential(*layers)
        self.normalize = normalize


    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1) if self.normalize else z

class PredictionMLP(nn.Module):

    def __init__(self, in_dim=256, hidden_dim=1024, out_dim=256, use_bn=True, normalize=True):
        super().__init__()

        layers = [
            nn.Linear(in_dim, hidden_dim, bias=True),
        ]
        layers += [nn.BatchNorm1d(hidden_dim)]
        layers += [nn.SiLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, out_dim, bias=True)]
        
        self.net = nn.Sequential(*layers)
        self.normalize = normalize

    def forward(self, z):
        p = self.net(z)
        return F.normalize(p, dim=-1) if self.normalize else p
