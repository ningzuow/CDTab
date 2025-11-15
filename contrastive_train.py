# -*- coding: utf-8 -*-
"""
Train contrastive model (InfoNCE) on your pre-made batches (adult_batch_XXX.csv).
- 多个正样本对：逐对计算；负样本=当前 batch 中所有 __role=='neg'
- 对称交叉：loss = InfoNCE(q(a), k(p)) + InfoNCE(q(p), k(a))
- 动量编码器：仅更新主网；动量网用 EMA 同步
"""

import os, json, math, glob
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim

# === 需要的自定义模块 ===
from contrastive_model import ContrastiveModel_Main, ContrastiveModel_Momentum
# Tokenizer 需要每个类别列的“全集基数”，所以要从完整 train.csv 统计
from tabdiff.modules.transformer import Tokenizer  # 仅作类型提示，不直接用

# ============== 配置（只改这里）==============
DATANAME = "adult"
INFO_PATH = f"data/{DATANAME}/info.json"
FULL_TRAIN_CSV = f"data/{DATANAME}/train.csv"   # 用它来统计每个类别列的全集基数
BATCH_DIR = "contrastive_batches"               # 你保存 batch_000.csv 的目录
BATCH_GLOB = os.path.join(BATCH_DIR, f"{DATANAME}_batch_*.csv")

EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 0.0
BATCH_TEMPERATURE = 0.07      # tau,InfoNCE温度
MOMENTUM = 0.99               # EMA 系数 m
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_TOKEN = 64
N_LAYERS = 2
N_HEADS = 4
PROJ_DIM = 256
PRED_HID = 1024
TRANS_HID = 128
# ==========================================


# 工具：获取数据集信息
def load_info(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 工具：用整份 train.csv 统计每个类别列的全集基数
def get_category_cardinalities_from_full(train_csv, info):
    df_full = pd.read_csv(train_csv)
    cat_idx = info["cat_col_idx"]
    # 注意：TabDiff 的 Tokenizer 不包含 y；adult 的 y 在 target_col_idx=14，不在 cat_idx 里
    Ks = [df_full.iloc[:, j].nunique() for j in cat_idx] # 类别列的全集基数列表
    return Ks

def codes_to_onehot(x_cat_codes: torch.Tensor, Ks):
    """x_cat_codes: [N, n_cat] 的整数编码；返回拼接后的 one-hot: [N, sum(Ks)]"""
    blocks = [F.one_hot(x_cat_codes[:, i], num_classes=Ks[i]) for i in range(len(Ks))]
    return torch.cat(blocks, dim=1).float()


# ---------- 新增：从完整 train.csv 建立每个类别列的全局映射 ----------
def build_global_category_encoders(train_csv, info):
    """为每个类别列建立 '类别名 -> 整数索引' 的全局映射"""
    df_full = pd.read_csv(train_csv)
    cat_idx = info["cat_col_idx"]
    encoders = []
    for idx in cat_idx:
        col = df_full.iloc[:, idx].astype("category")
        mapping = {cat: i for i, cat in enumerate(col.cat.categories)}
        encoders.append(mapping)
    return encoders




# 工具：把一个 batch CSV 转成 (x_num, x_cat_codes, pairs, neg_idx) 
"""
    返回：
      x_num: FloatTensor [N, d_num]
      x_cat_codes: LongTensor [N, n_cat]  —— 每列是类别整数编码(0..K-1)
      pairs: list of (anchor_idx, pos_idx)
      neg_idx: LongTensor [N_neg]
"""
def load_contrastive_batch(batch_csv, info, encoders):
    
    df = pd.read_csv(batch_csv)

    # 丢掉我们自己加的辅助列之后再做编码
    meta_cols = ["__role", "__pair_id", "__class", "__orig_row"]
    assert all(c in df.columns for c in meta_cols), "batch CSV 缺少辅助标签列"
    info_num = info["num_col_idx"]
    info_cat = info["cat_col_idx"]

    # 数值
    x_num = torch.tensor(df.iloc[:, info_num].values, dtype=torch.float32)

    # 使用全局映射编码每列（避免 batch 缺类导致编码不一致）
    x_cat_list = []
    for i, col_idx in enumerate(info_cat):
        mapping = encoders[i]
        col = df.iloc[:, col_idx].map(mapping).fillna(0).astype(int)
        x_cat_list.append(torch.tensor(col.values, dtype=torch.int64).unsqueeze(1))
    x_cat_codes = torch.cat(x_cat_list, dim=1)


    # 组正样本对：每个 pair_id 恰好有 anchor 和 pos
    anchors = df.index[df["__role"] == "anchor"].tolist()
    pos_map = df[df["__role"] == "pos"].set_index("__pair_id").index.to_series()
    # 用 pair_id 对齐 anchor 和 pos
    pairs = []
    for rid in anchors:
        pid = int(df.loc[rid, "__pair_id"])
        # 找到同 pid 的正样本的行号
        pos_rid = int(df.index[(df["__pair_id"] == pid) & (df["__role"] == "pos")][0])
        pairs.append((rid, pos_rid))

    # 负样本索引
    neg_idx = torch.tensor(df.index[df["__role"] == "neg"].tolist(), dtype=torch.long)

    return x_num, x_cat_codes, pairs, neg_idx

# ---------- 新增：从完整 train.csv 建立每个类别列的全局映射 ----------
def build_global_category_encoders(train_csv, info):
    """为每个类别列建立 '类别名 -> 整数索引' 的全局映射"""
    df_full = pd.read_csv(train_csv)
    cat_idx = info["cat_col_idx"]
    encoders = []
    for idx in cat_idx:
        col = df_full.iloc[:, idx].astype("category")
        mapping = {cat: i for i, cat in enumerate(col.cat.categories)}
        encoders.append(mapping)
    return encoders



# InfoNCE：单个 query 对一组 keys（第0个为正，其余为负）
# 这个函数中有一个[None]的操作，此操作前是一个标量（零维张量），此操作后才是向量，以符合 cross_entropy 的输入要求
# 加一个维度
def info_nce_one(q, k_pos, k_neg, tau):
    """
    q: [C]
    k_pos: [C]
    k_neg: [M, C]
    """
    q = F.normalize(q, dim=0)
    k_pos = F.normalize(k_pos, dim=0)
    k_neg = F.normalize(k_neg, dim=1) if k_neg.ndim == 2 else k_neg

    # logits = [ q·k_pos, q·k_neg_1, ... ]
    pos_logit = torch.matmul(q, k_pos)            # 标量
    # 没有负样本时的特殊处理，不可能发生，GPT写的保险
    if k_neg.numel() == 0:
        logits = pos_logit[None]
        labels = torch.tensor([0], device=logits.device)
    else:
        neg_logits = torch.matmul(k_neg, q)        # [M]
        logits = torch.cat([pos_logit[None], neg_logits], dim=0)  # [1+M]
        labels = torch.tensor([0], device=logits.device)

    loss = F.cross_entropy(logits[None] / tau, labels)  # [1, 1+M]
    return loss

# ---------- 训练主函数 ----------
def train():
    # 1) 准备 info & 类别全集基数
    info = load_info(INFO_PATH)
    Ks = get_category_cardinalities_from_full(FULL_TRAIN_CSV, info)
    
    encoders = build_global_category_encoders(FULL_TRAIN_CSV, info)
    
    d_num = len(info["num_col_idx"])
    print(f"[INFO] 类别全集基数 Ks={Ks}")

    # 2) 初始化主网 & 动量网
    model_q = ContrastiveModel_Main(
        num_continuous=d_num,
        categories=Ks,
        d_token=D_TOKEN,
        transformer_dim=TRANS_HID,
        n_heads=N_HEADS,
        num_layers=N_LAYERS,
        proj_out_dim=PROJ_DIM,
        pred_hidden_dim=PRED_HID,
    ).to(DEVICE)

    model_k = ContrastiveModel_Momentum(
        num_continuous=d_num,
        categories=Ks,
        d_token=D_TOKEN,
        transformer_dim=TRANS_HID,
        n_heads=N_HEADS,
        num_layers=N_LAYERS,
        proj_out_dim=PROJ_DIM,
    ).to(DEVICE)

    # 初始把 q 参数拷到 k，并冻结 k 的梯度
    for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
        p_k.data.copy_(p_q.data)
        p_k.requires_grad_(False)

    # 优化器opt只绑定了主网的参数，所以反向传播时只会更新主网
    opt = optim.AdamW(model_q.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 读取所有 batch 文件
    batch_files = sorted(glob.glob(BATCH_GLOB))
    assert batch_files, f"未找到 {BATCH_GLOB}"

    global_step = 0
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        for bpath in batch_files:
            # 3) 读取一个 batch
            x_num, x_cat_codes, pairs, neg_idx = load_contrastive_batch(bpath, info, encoders)
            x_num = x_num.to(DEVICE)
            x_cat_codes = x_cat_codes.to(DEVICE)
            neg_idx = neg_idx.to(DEVICE)
            x_cat_oh = codes_to_onehot(x_cat_codes, Ks).to(DEVICE)
            
            # 启用训练模式，更新参数
            model_q.train()
            # 关闭训练行为，固定参数
            model_k.eval()

            # 4) 一次性前向：主网给 query 向量 p；动量网给 key 向量 z
            with torch.no_grad():
                z_all = model_k(x_num, x_cat_oh)             # [N, C] keys
            p_all, _ = model_q(x_num, x_cat_oh)               # [N, C] queries

            # 5) 逐对累加 InfoNCE（对称交叉）
            loss_total = torch.zeros((), device=DEVICE)
            if neg_idx.numel() > 0:
                k_neg = z_all[neg_idx]                           # [M, C]
            else:
                k_neg = z_all[:0]                                # 空张量

            for a_idx, p_idx in pairs:
                a_idx = int(a_idx); p_idx = int(p_idx)
                q1 = p_all[a_idx]    # a -> query (主网)
                q2 = p_all[p_idx]    # p -> query (主网)
                k1 = z_all[a_idx]    # a -> key   (动量网)
                k2 = z_all[p_idx]    # p -> key   (动量网)

                # 对称：q(a) 对 k(p)，以及 q(p) 对 k(a)
                loss_ap = info_nce_one(q1, k2, k_neg, BATCH_TEMPERATURE)
                loss_pa = info_nce_one(q2, k1, k_neg, BATCH_TEMPERATURE)
                loss_total = loss_total + (loss_ap + loss_pa)


            loss_total = loss_total / len(pairs)
            # 6) 反传，只更新主网
            opt.zero_grad()
            loss_total.backward()
            opt.step()

            # 7) EMA 同步到动量网
            with torch.no_grad():
                m = MOMENTUM
                for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
                    p_k.data.mul_(m).add_(p_q.data, alpha=(1.0 - m))

            global_step += 1
            print(f"[step {global_step:05d}] {os.path.basename(bpath)}  pairs={len(pairs)}  "
                  f"neg={int(neg_idx.numel())}  loss={loss_total.item():.4f}")

    # 8) 训练结束，保存主网权重
    os.makedirs("ckpts", exist_ok=True)
    torch.save(model_q.state_dict(), f"ckpts/contrastive_main.pt")
    torch.save(model_k.state_dict(), f"ckpts/contrastive_momentum.pt")
    print("\n[OK] Saved to ckpts/contrastive_main.pt / contrastive_momentum.pt")


if __name__ == "__main__":
    train()
