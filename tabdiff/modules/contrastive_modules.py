# tabdiff/modules/contrastive_module.py
# -*- coding: utf-8 -*-
"""
基础对比学习模块：
只实现 ProjectionMLP 和 PredictionMLP，
后续的组装（ContrastiveModel）可另放文件实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# GPT：通用MLP公式，Input → [Linear + (BN) + Activation] × (L - 1) → [Linear + (BN)]
# 我自己看了一下，大概也是这样


# 投影头：把 encoder 的表征 h 映射到对比空间 z
# MLP就是几层全连接+激活+BN的堆叠
# 变换空间，让特征可比较。
class ProjectionMLP(nn.Module):  # 在PyTorch里，nn.Module是所有神经网络模块的基类

    def __init__(self, in_dim, out_dim=256, hidden_dim=1024, num_layers=2, use_bn=True, normalize=True):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, hidden_dim, bias=True)] # 普通的全连接层
        if use_bn:
            layers += [nn.BatchNorm1d(hidden_dim)] # 批归一化层，让训练更稳定
        layers += [nn.SiLU(inplace=True)] # 一种激活函数，类似ReLU但更平滑

        # 如果是3层网络，中间再加一层
        # 中间层维度都是 hidden_dim，因为我们希望中间层都在一个“统一的隐空间”里工作。
        if num_layers == 3:
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=True)]
            if use_bn:
                layers += [nn.BatchNorm1d(hidden_dim)]
            layers += [nn.SiLU(inplace=True)]

        # GPT: 最后一层映射到输出维度
        # 最后再加一个 BatchNorm（但不学习仿射参数），
        # 这是 SimCLR/MoCo 常见做法，可以让输出空间的分布更稳定。
        # 有待验证
        layers += [nn.Linear(hidden_dim, out_dim, bias=True)]
        if use_bn:
            layers += [nn.BatchNorm1d(out_dim, affine=False)]

        # 把一个 Python 列表里的层（layers）组合成一个完整的网络结构
        self.net = nn.Sequential(*layers)
        # 一个标志位，用于控制是否执行 L2 归一化。
        self.normalize = normalize

    # 为什么每个神经网络类里面都有这个函数？
    # forward() 定义了 输入数据是如何流经这个网络的。
    # 其实 PyTorch 内部执行的是：model.forward(x)
    # 它会自动处理梯度、注册参数、反向传播等一切工作。
    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1) if self.normalize else z

# 预测头：仅主分支使用，预测目标分支特征，便于学习更好的表征
# 稳定训练，让 q 能预测 k
class PredictionMLP(nn.Module):

    def __init__(self, in_dim=256, hidden_dim=1024, out_dim=256, use_bn=True, normalize=True):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim, bias=True),
        ]
        if use_bn:
            layers += [nn.BatchNorm1d(hidden_dim)]
        layers += [nn.SiLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, out_dim, bias=True)]
        self.net = nn.Sequential(*layers)
        self.normalize = normalize

    def forward(self, z):
        p = self.net(z)
        return F.normalize(p, dim=-1) if self.normalize else p
