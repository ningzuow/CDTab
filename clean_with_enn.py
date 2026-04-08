import os
import sys
import math
import time
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


REAL_DATA_DIR = "/root/autodl-tmp/Program_2/ConTabDiff/data/adult"
REAL_TRAIN_FILE = os.path.join(REAL_DATA_DIR, "train.csv")

GEN_DATA_DIR = "/root/autodl-tmp/Program_2/ConTabDiff/eval/report_runs/learnable_schedule/adult"

CONTRASTIVE_Q_CKPT_FILE = "/root/autodl-tmp/Program_2/ConTabDiff/tabdiff/ckpt/adult/learnable_schedule/contrastive_q_full_4670.pt"

CONTRASTIVE_CKPT_DIR = os.path.dirname(CONTRASTIVE_Q_CKPT_FILE)

LABEL_COL = "income"
CLEAN_OUT_DIR = os.path.join(GEN_DATA_DIR, "enn_cleaned")
os.makedirs(CLEAN_OUT_DIR, exist_ok=True)


import os
import pickle
import torch

from tabdiff.modules.main_modules import UniModMLP
from tabdiff.modules.contrastive_modules import ProjectionMLP, PredictionMLP



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONTRASTIVE_CONFIG_FILE = os.path.join(os.path.dirname(CONTRASTIVE_Q_CKPT_FILE), "config.pkl")
with open(CONTRASTIVE_CONFIG_FILE, "rb") as f:
    cfg = pickle.load(f)


backbone = UniModMLP(**cfg["unimodmlp_params"]).to(device)

tokenizer_q = backbone.tokenizer
encoder_q = backbone.encoder

d_token = cfg["unimodmlp_params"]["d_token"]
d_num   = cfg["unimodmlp_params"]["d_numerical"]
d_cat   = len(cfg["unimodmlp_params"]["categories"])
encoder_out_dim = d_token * (d_num + d_cat)

proj_q = ProjectionMLP(in_dim=encoder_out_dim).to(device)
pred_q = PredictionMLP(in_dim=256).to(device)


q_state = torch.load(CONTRASTIVE_Q_CKPT_FILE, map_location=device)

tokenizer_q.load_state_dict(q_state["tokenizer"], strict=True)
encoder_q.load_state_dict(q_state["encoder"], strict=True)
proj_q.load_state_dict(q_state["proj"], strict=True)
pred_q.load_state_dict(q_state["pred"], strict=True)


backbone.eval()
proj_q.eval()
pred_q.eval()


import json
import torch
import torch.nn.functional as F

from utils_train import TabDiffDataset, preprocess
info_path = os.path.join(REAL_DATA_DIR, "info.json")
with open(info_path, "r") as f:
    info = json.load(f)

num_col_idx = info["num_col_idx"]
cat_col_idx = info["cat_col_idx"]


train_data = TabDiffDataset(
    dataname="adult",
    data_dir=REAL_DATA_DIR,
    info=info,
    isTrain=True,
    y_only=False,
    dequant_dist=cfg["data"]["dequant_dist"],
    int_dequant_factor=cfg["data"]["int_dequant_factor"],
)

X_real = train_data.X.to(device)                    
y_real = train_data.y.cpu().numpy().astype(np.int64)

num_dim = train_data.d_numerical
categories = train_data.categories.tolist()         

import src
from utils_train import make_dataset

T_dict = {}
T_dict["normalization"] = "quantile"
T_dict["num_nan_policy"] = "mean"
T_dict["cat_nan_policy"] = None
T_dict["cat_min_frequency"] = None
T_dict["cat_encoding"] = None              
T_dict["y_policy"] = "default"
T_dict["dequant_dist"] = cfg["data"]["dequant_dist"]
T_dict["int_dequant_factor"] = cfg["data"]["int_dequant_factor"]

T = src.Transformations(**T_dict)

real_dataset_with_T = make_dataset(
    data_path=REAL_DATA_DIR,
    T=T,
    task_type=info["task_type"],
    change_val=False,
    concat=True,
    y_only=False,
)

num_transform = getattr(real_dataset_with_T, "num_transform", None)
int_transform = getattr(real_dataset_with_T, "int_transform", None)
cat_transform = getattr(real_dataset_with_T, "cat_transform", None)



num_classes_list = categories


def forward_pq_from_X(x: torch.Tensor) -> torch.Tensor:
    x_num = x[:, :num_dim].float()
    x_cat_idx = x[:, num_dim:].long()

    one_hot_list = []
    for j, Kj in enumerate(num_classes_list):
        col = x_cat_idx[:, j]
        one_hot = F.one_hot(col, num_classes=Kj)
        one_hot_list.append(one_hot.float())
    x_cat_oh = torch.cat(one_hot_list, dim=1)

    with torch.no_grad():
        t_q = tokenizer_q(x_num, x_cat_oh)[:, 1:, :]
        h_q = encoder_q(t_q).reshape(x.size(0), -1)
        p_q = pred_q(proj_q(h_q))
    return p_q

def load_and_encode_gen_csv(gen_csv_path: str) -> Tuple[pd.DataFrame, torch.Tensor]:
    gen_df_raw = pd.read_csv(gen_csv_path)

    gen_x_df = gen_df_raw.drop(columns=[LABEL_COL], errors="ignore")

    if len(num_col_idx) > 0:
        X_gen_num = gen_x_df.iloc[:, num_col_idx].to_numpy(dtype=np.float32)
        if num_transform is not None:
            X_gen_num = num_transform.transform(X_gen_num)
        if int_transform is not None:
            X_gen_num = int_transform.transform(X_gen_num)
    else:
        X_gen_num = np.zeros((len(gen_x_df), 0), dtype=np.float32)

    if len(cat_col_idx) > 0:
        if cat_transform is None:
            raise RuntimeError("cat_transform is empty.")
        X_gen_cat_raw = gen_x_df.iloc[:, cat_col_idx].astype(str).to_numpy()
        X_gen_cat = cat_transform.transform(X_gen_cat_raw)
        X_gen_cat = np.asarray(X_gen_cat, dtype=np.int64)
    else:
        X_gen_cat = np.zeros((len(gen_x_df), 0), dtype=np.int64)

    X_gen_num_t = torch.from_numpy(X_gen_num.astype(np.float32, copy=False)).to(device)
    X_gen_cat_t = torch.from_numpy(X_gen_cat.astype(np.int64, copy=False)).to(device)
    X_gen = torch.cat([X_gen_num_t, X_gen_cat_t], dim=1)
    return gen_df_raw, X_gen



p_real = forward_pq_from_X(X_real)


import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

def enn_filter_generated(
    p_real: torch.Tensor,
    y_real: np.ndarray,
    gen_df: pd.DataFrame,
    p_gen: torch.Tensor,
    k: int = 3,
    majority_label: int = 0,
    metric: str = "euclidean",
):
    assert p_real.ndim == 2 and p_gen.ndim == 2
    assert p_real.size(1) == p_gen.size(1)
    assert len(y_real) == p_real.size(0)
    assert len(gen_df) == p_gen.size(0)
    assert k >= 1

    p_real_np = p_real.detach().cpu().numpy().astype(np.float32, copy=False)
    p_gen_np  = p_gen.detach().cpu().numpy().astype(np.float32, copy=False)
    y_real_np = np.asarray(y_real, dtype=np.int64)

    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn.fit(p_real_np)

    knn_idx = nn.kneighbors(p_gen_np, return_distance=False)  # [N_gen, k]


    maj_cnt = (y_real_np[knn_idx] == majority_label).sum(axis=1)  # [N_gen]
    remove_mask = maj_cnt > (k / 2.0)

    removed_idx = np.where(remove_mask)[0]
    kept_idx = np.where(~remove_mask)[0]

    gen_df_kept = gen_df.iloc[kept_idx].reset_index(drop=True)
    return gen_df_kept, removed_idx

import glob


gen_files = sorted(glob.glob(os.path.join(GEN_DATA_DIR, "samples_*.csv")))


for gen_path in gen_files:

    gen_df_raw, X_gen = load_and_encode_gen_csv(gen_path)
    p_gen = forward_pq_from_X(X_gen)
    gen_df_clean, removed_idx = enn_filter_generated(
        p_real=p_real,
        y_real=y_real,
        gen_df=gen_df_raw,
        p_gen=p_gen,
        k=3,
        majority_label=0,
    )

    save_path = os.path.join(CLEAN_OUT_DIR, os.path.basename(gen_path))
    gen_df_clean.to_csv(save_path, index=False)



