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
def build_1nn_on_minority(X_embed: np.ndarray, minority_idx: np.ndarray) -> Dict[int, int]:

    # 映射到“少数类子空间”的行号与向量
    X_min = X_embed[minority_idx]
    
    # n_neighbors=2：第一个是自身，第二个才是最近的其他样本
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_min)
    dist, ind = nn.kneighbors(X_min, return_distance=True)
    
    # ind 的形状 (n_minority, 2)，ind[i, 0] == i（自身），ind[i, 1] 是最近邻在少数类子空间的下标
    mapping = {}
    for i_local, row_global in enumerate(minority_idx):
        nn_local = ind[i_local, 1]
        nn_global = minority_idx[nn_local]
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
                   pairs: List[Tuple[int, int]],
                   neg_ids: List[int]) -> pd.DataFrame:

    rows = []
    for pid, (a, p) in enumerate(pairs):
        for rid, role in [(a, "anchor"), (p, "pos")]:
            row = df.loc[rid].copy()
            row["__role"] = role
            row["__pair_id"] = pid
            row["__class"] = "minority"
            row["__orig_row"] = rid
            rows.append(row)
    
    for rid in neg_ids:
        row = df.loc[rid].copy()
        row["__role"] = "neg"
        row["__pair_id"] = -1
        row["__class"] = "majority"
        row["__orig_row"] = rid
        rows.append(row)
    batch_df = pd.DataFrame(rows)
    
    # 打乱一下行顺序，避免固定排列产生学习偏置；如需稳定顺序可注释下一行
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

    nn_map = build_1nn_on_minority(X_embed, min_idx)
    deterministic_keys = sorted(list(nn_map.keys()))

    # (4) 选择前 K=16 个 key 生成有向对（anchor→pos）
    # === 新增：计算可生成的 batch 数 ===
    total_minority = len(deterministic_keys)
    batch_size_pairs = args.pairs                 # 每批 16 对（32 少数类样本）
    num_batches = int(np.ceil(total_minority / batch_size_pairs))

    print(f"[INFO] 共 {total_minority} 个少数类样本，可生成 {num_batches} 个 batch。")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.dataname}_minority_1nn.json"), "w", encoding="utf-8") as f:
        json.dump(nn_map, f, ensure_ascii=False, indent=2)

    deterministic_keys = sorted(list(nn_map.keys()))

    # 生成固定数量的随机 batch（200 个）
    num_batches = 10

    for b_id in range(num_batches):
    # 从少数类中随机选出 2 * pairs 个（anchor+pos 各一）
        key_chunk = random.sample(deterministic_keys, k=min(args.pairs, len(deterministic_keys)))


        # 选取当前批次的 anchor→positive 对
        pairs = pick_k_pairs(nn_map, k=len(key_chunk), deterministic_keys=key_chunk)

        # 从多数类里按顺序采样（循环复用）
        neg_ids = sample_majority(maj_idx, n_neg=args.neg, seed=args.seed, batch_id=b_id)


        # 组装 batch
        batch_df = assemble_batch(df, pairs, neg_ids)

        # 保存当前 batch
        out_csv = os.path.join(args.out_dir, f"{args.dataname}_batch_{b_id:03d}.csv")
        batch_df.to_csv(out_csv, index=False)

        print(f"[OK] 保存 batch {b_id+1}/{num_batches}: {out_csv}")



if __name__ == "__main__":
    main()
