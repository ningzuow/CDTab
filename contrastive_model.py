# tabdiff/modules/contrastive_model.py
# -*- coding: utf-8 -*-
"""
独立版对比学习网络（基于 TabDiff 的 Tokenizer + Transformer + 我实现的 MLP）

结构:
    Tokenizer → Transformer(Encoder) → ProjectionMLP → PredictionMLP
输出:
    anchor/positive 的投影向量 (z_q, z_k) 和预测向量 (p_q)
    z和p的shape均为 (batch_size, proj_out_dim)
用法:
    from tabdiff.modules.contrastive_model import ContrastiveModel

    model = ContrastiveModel(
        num_features=输入维度, 
        num_categories=类别特征数, 
        transformer_dim=128
    )
    out = model(x_num, x_cat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pandas as pd

# 从 TabDiff 主体导入 Tokenizer 和 Transformer
from tabdiff.modules.transformer import Transformer
from tabdiff.modules.transformer import Tokenizer  # 部分版本中在 tabdiff/modules/transformer.py 里一起定义

# 从你刚写的 contrastive_module 导入两个 MLP
from tabdiff.modules.contrastive_modules import ProjectionMLP, PredictionMLP


# 对比学习网络: Tokenizer + Transformer + ProjectionMLP + PredictionMLP
class ContrastiveModel_Main(nn.Module):

    def __init__(
        self,
        num_continuous: int,
        categories: list,
        d_token: int = 64,
        transformer_dim: int = 128, # Transformer 内部隐层维度
        n_heads: int = 4, # Multi-Head Attention 的头数
        num_layers: int = 2, # Transformer 堆叠层数
        proj_out_dim: int = 256,
        pred_hidden_dim: int = 1024,
    ):
        super().__init__()

        # === 1. Tokenizer ===
        self.tokenizer = Tokenizer(
            d_numerical=num_continuous,
            categories=categories,
            d_token=d_token,
            bias = True
        )

        # === 2. Transformer Encoder ===
        # 直接用 TabDiff 已实现的 Transformer 类
        self.encoder = Transformer(
            n_layers=num_layers,# Transformer 堆叠层数，每一层包含 Multi-Head Attention + Feed Forward
            d_token=d_token,
            n_heads=n_heads,    # 多头注意力的“视角数量”
            d_out=d_token,         # 输出维度与输入相同即可
            d_ffn_factor=4,        # FFN 扩展因子，一般用 2~4
            attention_dropout=0.1,
            ffn_dropout=0.1,
            residual_dropout=0.1,
        )

        # === 3. Projection MLP ===
        self.projection = ProjectionMLP(
            in_dim=d_token,         # 输入维度与 Transformer 输出一致
            out_dim=proj_out_dim,
            hidden_dim=transformer_dim,
            num_layers=2,
            use_bn=True,
            normalize=True,
        )

        # === 4. Prediction MLP ===
        self.prediction = PredictionMLP(
            in_dim=proj_out_dim,
            hidden_dim=pred_hidden_dim,
            out_dim=proj_out_dim,
            use_bn=True,
            normalize=True,
        )

    def forward(self, x_num, x_cat=None, timesteps=None):
        # Tokenize
        tokens = self.tokenizer(x_num, x_cat)
        tokens = tokens[:, 1:, :]           # 与 TabDiff 保持一致
        enc_out = self.encoder(tokens)

        # 池化成全局向量
        h = enc_out[:, 0, :] if enc_out.shape[1] > 1 else enc_out.squeeze(1)

        # Projection + Prediction
        z = self.projection(h)
        p = self.prediction(z)
        return p, z
    
# 对比学习中的 “Momentum / Key Encoder”（无 Prediction MLP）
class ContrastiveModel_Momentum(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        categories: list,
        d_token: int = 64,
        transformer_dim: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        proj_out_dim: int = 256,
    ):
        super().__init__()

        # === 1. Tokenizer ===
        self.tokenizer = Tokenizer(
            d_numerical=num_continuous,
            categories=categories,
            d_token=d_token,
            bias=True
        )

        # === 2. Transformer Encoder ===
        self.encoder = Transformer(
            n_layers=num_layers,
            d_token=d_token,
            n_heads=n_heads,
            d_out=d_token,
            d_ffn_factor=4,
            attention_dropout=0.1,
            ffn_dropout=0.1,
            residual_dropout=0.1,
        )

        # === 3. Projection MLP ===
        self.projection = ProjectionMLP(
            in_dim=d_token,
            out_dim=proj_out_dim,
            hidden_dim=transformer_dim,
            num_layers=2,
            use_bn=True,
            normalize=True,
        )

    def forward(self, x_num, x_cat=None):
        # Tokenize
        tokens = self.tokenizer(x_num, x_cat)
        tokens = tokens[:, 1:, :]
        enc_out = self.encoder(tokens)

        # 池化（取 [CLS] 向量）
        h = enc_out[:, 0, :] if enc_out.shape[1] > 1 else enc_out.squeeze(1)

        # Projection 输出表示向量 z_k
        z_k = self.projection(h)
        return z_k

