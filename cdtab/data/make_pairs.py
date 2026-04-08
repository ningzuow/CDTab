import os
import json
import argparse
import random
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Parse the data and information file paths.
def resolve_paths(dataname: str, csv_path: str = None, info_path: str = None) -> Tuple[str, str]:
    
    if csv_path is None:
        cands = [f"data/{dataname}/train.csv", f"synthetic/{dataname}/train.csv"]
        csv_path = next((p for p in cands if os.path.exists(p)), None)
    if info_path is None:
        cand = f"data/{dataname}/info.json"
        info_path = cand if os.path.exists(cand) else None

    assert csv_path and os.path.exists(csv_path), f"找不到训练 CSV：{csv_path or '未提供'}"
    assert info_path and os.path.exists(info_path), f"找不到 info.json：{info_path or '未提供'}"
    return csv_path, info_path

def load_info(info_path: str) -> Dict:

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    for k in ["num_col_idx", "cat_col_idx", "target_col_idx"]:
        info[k] = [int(i) for i in info[k]]
    return info


def split_num_cat_y(df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    
    num_idx = info["num_col_idx"]
    cat_idx = info["cat_col_idx"]
    y_idx = info["target_col_idx"]
    
    assert len(y_idx) == 1, "The current script assumes binary classification."
    X_num = df.iloc[:, num_idx].copy() if len(num_idx) else pd.DataFrame(index=df.index)
    X_cat = df.iloc[:, cat_idx].copy() if len(cat_idx) else pd.DataFrame(index=df.index)
    y = df.iloc[:, y_idx[0]].copy()
    
    return X_num, X_cat, y

def fit_transform_features(X_num: pd.DataFrame, X_cat: pd.DataFrame):

    parts = []
    scal = None
    ohe = None

    if X_num.shape[1] > 0:
        scal = StandardScaler()
        parts.append(scal.fit_transform(X_num.values))
    if X_cat.shape[1] > 0:
        
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        parts.append(ohe.fit_transform(X_cat.astype(str).values))
   
    if not parts:
        raise ValueError("There are no available features for similarity calculation.")
    
    X = np.concatenate(parts, axis=1)
    return X, scal, ohe


def find_minority_label(y: pd.Series) -> Tuple[object, object]:
    vc = y.value_counts()
    assert len(vc) == 2, f"The current script only supports binary classification, and the actual number of categories:{len(vc)}"
    minority = vc.idxmin()
    majority = vc.idxmax()
    
    return minority, majority



def build_1nn_within_class(X_embed: np.ndarray, class_idx: np.ndarray) -> Dict[int, int]:

    X_cls = X_embed[class_idx]

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_cls)
    dist, ind = nn.kneighbors(X_cls, return_distance=True)

    mapping = {}
    for i_local, row_global in enumerate(class_idx):
        nn_local = ind[i_local, 1]          
        nn_global = class_idx[nn_local]
        mapping[int(row_global)] = int(nn_global)
    return mapping


def pick_k_pairs(pair_dict: Dict[int, int], k: int, deterministic_keys: List[int] = None) -> List[Tuple[int, int]]:

    keys = deterministic_keys if deterministic_keys is not None else list(pair_dict.keys())
    
    keys = [int(x) for x in keys]
    pairs = []
    for k_idx in keys[:k]:
        pairs.append((k_idx, pair_dict[k_idx]))
    return pairs


def sample_majority(majority_idx: np.ndarray, n_neg: int, seed: int, batch_id: int = 0) -> List[int]:
    rng = random.Random(seed + batch_id)
    majority_idx = list(map(int, majority_idx))
    return rng.choices(majority_idx, k=n_neg)


def assemble_batch(df: pd.DataFrame,
                   min_pairs: List[Tuple[int, int]],
                   maj_pairs: List[Tuple[int, int]]) -> pd.DataFrame:

    rows = []
    pid = 0
    for (a, p) in min_pairs:
        for rid in (a, p):
            row = df.loc[rid].copy()
            row["__pair_id"] = pid
            row["__orig_row"] = rid
            rows.append(row)
        pid += 1

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

    df = pd.read_csv(csv_path)
    X_num, X_cat, y = split_num_cat_y(df, info)
    minority_label, majority_label = find_minority_label(y)

    X_embed, scal, ohe = fit_transform_features(X_num, X_cat)

    idx_all = np.arange(len(df))
    min_mask = (y.values == minority_label)
    maj_mask = ~min_mask
    min_idx = idx_all[min_mask]
    maj_idx = idx_all[maj_mask]

    nn_min = build_1nn_within_class(X_embed, min_idx)
    nn_maj = build_1nn_within_class(X_embed, maj_idx)

    min_keys = sorted(list(nn_min.keys()))
    maj_keys = sorted(list(nn_maj.keys()))

    num_batches = 40

    pairs_per_class = args.pairs  

    for b_id in range(num_batches):
        cur_min_keys = random.sample(
            min_keys,
            k=min(pairs_per_class, len(min_keys))
        )
        min_pairs = pick_k_pairs(nn_min, k=len(cur_min_keys), deterministic_keys=cur_min_keys)

        cur_maj_keys = random.sample(
            maj_keys,
            k=min(pairs_per_class, len(maj_keys))
        )
        maj_pairs = pick_k_pairs(nn_maj, k=len(cur_maj_keys), deterministic_keys=cur_maj_keys)

        batch_df = assemble_batch(df, min_pairs, maj_pairs)

        print(f"[INFO] batch {b_id}: size={len(batch_df)} (minor pairs={len(min_pairs)}, major pairs={len(maj_pairs)})")

        out_csv = os.path.join(args.out_dir, f"{args.dataname}_batch_{b_id:03d}.csv")
        batch_df.to_csv(out_csv, index=False)
        print(f"[OK] batch {b_id+1}/{num_batches}: {out_csv}")




if __name__ == "__main__":
    main()
