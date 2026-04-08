# build_pair_info_adult.py  
import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors

DATA_DIR = "data/abalone_13VR"
INFO_PATH = os.path.join(DATA_DIR, "info.json")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
OUT_PATH = os.path.join(DATA_DIR, "pair_info.json")


with open(INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)

num_idx = info["num_col_idx"]
cat_idx = info["cat_col_idx"]
target_idx = info["target_col_idx"][0]
col_names = info["column_names"]


df = pd.read_csv(TRAIN_PATH)


df = df[col_names]


label_col = col_names[target_idx]
y = df[label_col].values

df_feat = df.drop(columns=[label_col])

num_cols = [col_names[i] for i in num_idx]
cat_cols = [col_names[i] for i in cat_idx]

df_num = df_feat[num_cols]
df_cat = df_feat[cat_cols]

scaler = MinMaxScaler()
X_num = scaler.fit_transform(df_num)

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(df_cat.astype(str))

X = np.concatenate([X_num, X_cat], axis=1)
print("Final feature shape:", X.shape)

pair_dict = {}
unique_labels = np.unique(y)

for cls in unique_labels:
    idx_cls = np.where(y == cls)[0]
    X_cls = X[idx_cls]

    if len(idx_cls) < 2:
        continue

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_cls)
    _, indices = nn.kneighbors(X_cls)

    for i_local, i_global in enumerate(idx_cls):
        j_local = indices[i_local, 1]
        j_global = idx_cls[j_local]
        pair_dict[int(i_global)] = int(j_global)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(pair_dict, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(pair_dict)} pairs → {OUT_PATH}")
