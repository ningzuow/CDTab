# -*- coding: utf-8 -*-
"""
使用真实 adult 数据的对比学习模型测试脚本
依赖：
    - make_pairs.py 已生成 ./contrastive_batches/adult_batch_000.csv 等文件
    - info.json 位于 data/adult/info.json
    - contrastive_model.py 已正确实现
"""

import os
import json
import pandas as pd
import torch
from contrastive_model import ContrastiveModel_Main

# 导入 TabDiff 的预处理类
from utils_train import TabDiffDataset
import json

# ========= 配置 =========
dataname = "adult"
batch_path = f"contrastive_batches/{dataname}_batch_000.csv"
info_path = f"data/{dataname}/info.json"

assert os.path.exists(batch_path), f"未找到批次文件：{batch_path}"
assert os.path.exists(info_path), f"未找到 info.json：{info_path}"

'''
# ========= 1. 使用 TabDiff 内置预处理加载 =========
with open(info_path, "r", encoding="utf-8") as f:
    info = json.load(f)

dataset = TabDiffDataset(
    dataname=dataname,
    data_dir=f"data/{dataname}",
    info=info,
    isTrain=True
)

# 提取处理后的数值 / 类别张量
d_numerical = dataset.d_numerical
categories = dataset.categories.tolist()

import torch.nn.functional as F

X = dataset.X
# 数值部分：直接取
x_num = X[:, :d_numerical].float()

# 类别部分：当前是“每列一个整数编码”，且第 0 列是 y（基数=2），需要先剔除 y，再逐列 one-hot 展开
cat_codes = X[:, d_numerical:].long()      # [N, n_cat + 1]，第 0 列是 y
cat_codes = cat_codes[:, 1:]               # 去掉 y
categories = categories[1:]                # 同步去掉 y 的基数（很关键！）

# 逐列 one-hot，并按列块拼接成 Tokenizer 期望的大矩阵
oh_blocks = [F.one_hot(cat_codes[:, j], num_classes=K).float()
             for j, K in enumerate(categories)]
x_cat = torch.cat(oh_blocks, dim=1).float()   # [N, sum(categories)]


print(f"[INFO] 使用 TabDiffDataset 预处理后：x_num={x_num.shape}, x_cat={x_cat.shape}")
print(f"[INFO] 类别列取值数：{categories}")
'''

# ========= 1. 从 contrastive batch 加载 =========
df = pd.read_csv(batch_path)
with open(info_path, "r", encoding="utf-8") as f:
    info = json.load(f)

num_idx = info["num_col_idx"]
cat_idx = info["cat_col_idx"]

# === 数值部分 ===
x_num = torch.tensor(df.iloc[:, num_idx].values, dtype=torch.float32)

# === 类别部分（转整数编码） ===
import torch.nn.functional as F

# === 计算每个类别特征的取值数（Tokenizer 要用） ===
categories = [df.iloc[:, idx].nunique(dropna=True) for idx in cat_idx]

# === 类别部分（逐列 one-hot，再按列块拼起来） ===
onehot_blocks = []
fixed_categories = []  # 若有缺失，会把 K+1 记录到这里，供 Tokenizer 使用
for j, col_idx in enumerate(cat_idx):
    s = df.iloc[:, col_idx].astype("category")
    codes = torch.tensor(pd.Categorical(s).codes, dtype=torch.long)   # -1 代表 NaN
    
    K = categories[j]
    if (codes == -1).any():
        # 若存在 NaN/缺失，让 NaN 占用一个额外的 code（把 -1 映射到 0），并将 K 增 1
        K = K + 1
        codes = torch.clamp(codes + 1, min=0)

    onehot = F.one_hot(codes, num_classes=K).float()
    onehot_blocks.append(onehot)
    fixed_categories.append(int(K))

x_cat = torch.cat(onehot_blocks, dim=1)  # [batch, sum(K_j)]
categories = fixed_categories            # 用修正后的 K 列表传给 Tokenizer


# === 计算每个类别特征的取值数（Tokenizer 要用） ===
categories = [df.iloc[:, idx].nunique() for idx in cat_idx]

d_numerical = len(num_idx)

print(f"[INFO] 从对比学习 batch 加载：x_num={x_num.shape}, x_cat={x_cat.shape}")
print(f"[INFO] 类别列取值数: {categories}")



# ========= 3. 初始化模型 =========
model = ContrastiveModel_Main(
    num_continuous=d_numerical,
    categories=categories,
    d_token=64,
    transformer_dim=128,
    n_heads=4,
    num_layers=2,
    proj_out_dim=256,
    pred_hidden_dim=1024,
)


# ========= 4. 前向传播 =========
with torch.no_grad():
    p, z = model(x_num, x_cat)

print(f"[OK] 前向传播成功！")
print(f"p shape: {p.shape}, z shape: {z.shape}")


# ========= 5. 测试 Momentum Encoder =========
from contrastive_model import ContrastiveModel_Momentum

momentum_model = ContrastiveModel_Momentum(
    num_continuous=d_numerical,
    categories=categories,
    d_token=64,
    transformer_dim=128,
    n_heads=4,
    num_layers=2,
    proj_out_dim=256,
)

with torch.no_grad():
    z_k = momentum_model(x_num, x_cat)

print(f"[OK] Momentum Encoder 前向传播成功！")
print(f"z_k shape: {z_k.shape}")

# ========= 6. 参数同步检查（可选）=========
# 若你计划做 MoCo，可以复制主模型参数到动量模型
for param_q, param_k in zip(model.parameters(), momentum_model.parameters()):
    param_k.data.copy_(param_q.data)
    param_k.requires_grad = False  # 动量分支不反向传播

print("[INFO] 参数同步完成，可用于 MoCo 风格训练。")
