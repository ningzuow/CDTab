# make_pairs.py
# -*- coding: utf-8 -*-
"""
根据“全体少数类上做 1-NN → 取前 16 对 → 再采样 224 个多数类 → 组成一个 256 的 batch”的流程,
为 TabDiff 项目(adult 数据集）构造正负样本批次。

用法示例：
python make_pairs.py --dataname adult \
    --csv data/adult/train.csv \
    --info data/adult/info.json \
    --pairs 16 --neg 224 --seed 42 \
    --out_dir contrastive_batches

如果不显式提供 --csv/--info,将按以下顺序自动寻找:
- CSV:data/{dataname}/train.csv 或 synthetic/{dataname}/train.csv
- INFO:data/{dataname}/info.json
"""
import os
import json
import argparse
import random
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# 智能解析数据与信息文件路径
def resolve_paths(dataname: str, csv_path: str = None, info_path: str = None) -> Tuple[str, str]:
    
    # 自动查找 CSV 与 info.json 路径（若未提供）
    if csv_path is None:
        cands = [f"data/{dataname}/train.csv", f"synthetic/{dataname}/train.csv"]
        csv_path = next((p for p in cands if os.path.exists(p)), None)
    if info_path is None:
        cand = f"data/{dataname}/info.json"
        info_path = cand if os.path.exists(cand) else None

    assert csv_path and os.path.exists(csv_path), f"找不到训练 CSV：{csv_path or '未提供'}"
    assert info_path and os.path.exists(info_path), f"找不到 info.json：{info_path or '未提供'}"
    return csv_path, info_path

# 读取 info.json（TabDiff 的列位置信息与任务类型
def load_info(info_path: str) -> Dict:

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    # 统一为 int 索引（有些 info 的 key 可能是字符串）
    for k in ["num_col_idx", "cat_col_idx", "target_col_idx"]:
        info[k] = [int(i) for i in info[k]]
    return info

# 按 info.json 划分数值/类别/目标列
def split_num_cat_y(df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    
    num_idx = info["num_col_idx"]
    cat_idx = info["cat_col_idx"]
    y_idx = info["target_col_idx"]
    
    assert len(y_idx) == 1, "当前脚本假定是二分类（单一目标列）。"
    X_num = df.iloc[:, num_idx].copy() if len(num_idx) else pd.DataFrame(index=df.index)
    X_cat = df.iloc[:, cat_idx].copy() if len(cat_idx) else pd.DataFrame(index=df.index)
    y = df.iloc[:, y_idx[0]].copy()
    
    return X_num, X_cat, y

"""
将特征转为可做距离的向量：
- 数值:StandardScaler
- 类别:OneHotEncoder
方便后续做 1-NN 搜索
返回:np.ndarray (N, D)、以及用于以后复现/调试的变换器。
"""
def fit_transform_features(X_num: pd.DataFrame, X_cat: pd.DataFrame):

    parts = []
    scal = None
    ohe = None
    # 处理数值与类别特征
    if X_num.shape[1] > 0:
        scal = StandardScaler()
        parts.append(scal.fit_transform(X_num.values))
    if X_cat.shape[1] > 0:
        # 统一转 string，避免 int/float 杂糅导致 OHE 出错
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        parts.append(ohe.fit_transform(X_cat.astype(str).values))
   
    if not parts:
        raise ValueError("没有可用特征用于相似度计算。")
    
    # 拼接为统一向量
    X = np.concatenate(parts, axis=1)
    return X, scal, ohe

# 识别少数类与多数类标签
def find_minority_label(y: pd.Series) -> Tuple[object, object]:
    vc = y.value_counts()
    assert len(vc) == 2, f"当前脚本仅支持二分类，实际类别数：{len(vc)}"
    minority = vc.idxmin()
    majority = vc.idxmax()
    
    return minority, majority


"""
在“全体少数类子集”上做 1-NN（排除自身），输出字典：global_row_id → nearest_global_row_id。
说明：
我们仅用少数类内部的最近邻，避免把多数类当“正样本”
这里使用欧几里得距离（数值标准化 + OHE 类别）
"""
def build_1nn_within_class(X_embed: np.ndarray, class_idx: np.ndarray) -> Dict[int, int]:
    """
    在某一类内部做 1-NN（排除自身），返回: 全局行号 -> 最近邻的全局行号
    """
    X_cls = X_embed[class_idx]

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_cls)
    dist, ind = nn.kneighbors(X_cls, return_distance=True)

    mapping = {}
    for i_local, row_global in enumerate(class_idx):
        nn_local = ind[i_local, 1]          # 第一个是自己，第二个才是最近邻
        nn_global = class_idx[nn_local]
        mapping[int(row_global)] = int(nn_global)
    return mapping



# 从“1-NN 字典”里，按顺序挑出前 k 个样本对（anchor → positive）
def pick_k_pairs(pair_dict: Dict[int, int], k: int, deterministic_keys: List[int] = None) -> List[Tuple[int, int]]:

    keys = deterministic_keys if deterministic_keys is not None else list(pair_dict.keys())
    
    # 确保key是整数
    keys = [int(x) for x in keys]
    pairs = []
    for k_idx in keys[:k]:
        pairs.append((k_idx, pair_dict[k_idx]))
    return pairs

# 从多数类中均匀采样 n_neg 个样本
def sample_majority(majority_idx: np.ndarray, n_neg: int, seed: int, batch_id: int = 0) -> List[int]:
    
    # 随机种子
    rng = random.Random(seed + batch_id)
    majority_idx = list(map(int, majority_idx))
    # 随机有放回采样
    return rng.choices(majority_idx, k=n_neg)

"""
组装最终批次（共 256 行）并打上辅助标签，方便下游 InfoNCE：
- __role: anchor / pos / neg
- __pair_id: [0..len(pairs)-1]（仅 anchor/pos 有值，neg 为 -1）
- __class: minority / majority
- __orig_row: 原始行号（便于回溯）
"""
def assemble_batch(df: pd.DataFrame,
                   min_pairs: List[Tuple[int, int]],
                   maj_pairs: List[Tuple[int, int]]) -> pd.DataFrame:
    """
    将少数类 / 多数类的正样本对一起组成一个 batch。
    每个 pair 生成两行数据，并打上 __pair_id 与 __orig_row 方便后续计算 loss。
    不再使用 __role / __class。
    """
    rows = []
    # 少数类 pair 先编号
    pid = 0
    for (a, p) in min_pairs:
        for rid in (a, p):
            row = df.loc[rid].copy()
            row["__pair_id"] = pid
            row["__orig_row"] = rid
            rows.append(row)
        pid += 1

    # 多数类 pair 接着编号
    for (a, p) in maj_pairs:
        for rid in (a, p):
            row = df.loc[rid].copy()
            row["__pair_id"] = pid
            row["__orig_row"] = rid
            rows.append(row)
        pid += 1

    batch_df = pd.DataFrame(rows)
    batch_df = batch_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return batch_df



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", type=str, default="adult")
    ap.add_argument("--csv", type=str, default=None, help="训练集 CSV 路径（不提供则自动查找）")
    ap.add_argument("--info", type=str, default=None, help="info.json 路径（不提供则自动查找）")
    ap.add_argument("--pairs", type=int, default=493, help="正样本对数量(anchor-pos 的对数)")
    ap.add_argument("--neg", type=int, default=3110, help="负样本（多数类）数量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="contrastive_batches")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    csv_path, info_path = resolve_paths(args.dataname, args.csv, args.info)
    info = load_info(info_path)

    # (1) 读取数据并拆分列
    df = pd.read_csv(csv_path)
    X_num, X_cat, y = split_num_cat_y(df, info)
    minority_label, majority_label = find_minority_label(y)

    # (2) 特征嵌入（标准化数值 + OHE 类别），用于距离计算
    X_embed, scal, ohe = fit_transform_features(X_num, X_cat)

    # (3) 构造“全体少数类上的 1-NN 字典”
    idx_all = np.arange(len(df))
    min_mask = (y.values == minority_label)
    maj_mask = ~min_mask
    min_idx = idx_all[min_mask]
    maj_idx = idx_all[maj_mask]

    # 为少数类 / 多数类分别建立“类内 1-NN”映射
    nn_min = build_1nn_within_class(X_embed, min_idx)
    nn_maj = build_1nn_within_class(X_embed, maj_idx)

    min_keys = sorted(list(nn_min.keys()))
    maj_keys = sorted(list(nn_maj.keys()))


        # 生成固定数量的随机 batch（20 个）
    num_batches = 20

    # 每个类在一个 batch 中的 pair 数
    pairs_per_class = args.pairs  # 建议你用 --pairs 1024，则 batch_size = 4 * 1024 = 4096

    for b_id in range(num_batches):
        # 随机从少数类 key 中选 pairs_per_class 个作为 anchor
        cur_min_keys = random.sample(
            min_keys,
            k=min(pairs_per_class, len(min_keys))
        )
        min_pairs = pick_k_pairs(nn_min, k=len(cur_min_keys), deterministic_keys=cur_min_keys)

        # 随机从多数类 key 中选 pairs_per_class 个作为 anchor
        cur_maj_keys = random.sample(
            maj_keys,
            k=min(pairs_per_class, len(maj_keys))
        )
        maj_pairs = pick_k_pairs(nn_maj, k=len(cur_maj_keys), deterministic_keys=cur_maj_keys)

        # 组装 batch（少数 / 多数类的 pairs 一起）
        batch_df = assemble_batch(df, min_pairs, maj_pairs)

        print(f"[INFO] batch {b_id}: size={len(batch_df)} (minor pairs={len(min_pairs)}, major pairs={len(maj_pairs)})")

        # 保存当前 batch
        out_csv = os.path.join(args.out_dir, f"{args.dataname}_batch_{b_id:03d}.csv")
        batch_df.to_csv(out_csv, index=False)
        print(f"[OK] 保存 batch {b_id+1}/{num_batches}: {out_csv}")




if __name__ == "__main__":
    main()
