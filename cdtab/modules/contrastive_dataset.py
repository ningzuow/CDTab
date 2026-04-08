import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class ContrastiveBatchDataset(Dataset):
    

    def __init__(self, csv_file, info, cdtab_dataset):
        df = pd.read_csv(csv_file)
        self.info = info

       
        num_idx = info["num_col_idx"]

       
        raw_path = info.get("data_path")
        if raw_path is None:
            raw_path = os.path.join(cdtab_dataset.data_dir, "adult.data")

        raw_train = pd.read_csv(raw_path, header=None)
        raw_train.columns = info["column_names"]


        x_num = torch.tensor(df.iloc[:, num_idx].values, dtype=torch.float32)  # [B, d_num]

        target_col_name = info["column_names"][info["target_col_idx"][0]]

        train_raw_labels = raw_train[target_col_name].astype(str).values
        label_vocab = pd.Series(train_raw_labels).unique().tolist()
        label2id = {s: i for i, s in enumerate(label_vocab)}

        y_raw_vals = df[target_col_name].astype(str).values
        y_encoded = np.array([label2id[v] for v in y_raw_vals], dtype=np.int64)
        x_y = torch.tensor(y_encoded, dtype=torch.long).unsqueeze(1)   # [B, 1]


        self.labels = torch.tensor(y_encoded, dtype=torch.long)        # [B]

        x_cat_cols = []
        for col in info["cat_col_idx"]:
            col_name = info["column_names"][col]

            train_raw_vals = raw_train[col_name].astype(str).values
            vocab = pd.Series(train_raw_vals).unique().tolist()
            str2id = {s: i for i, s in enumerate(vocab)}

            col_vals = df[col_name].astype(str).values
            encoded = np.array([str2id[v] for v in col_vals], dtype=np.int64)

            x_cat_cols.append(torch.tensor(encoded, dtype=torch.long).unsqueeze(1))  # [B, 1]

        
        x_cat = torch.cat([x_y] + x_cat_cols, dim=1)  # [B, 9]
        self.x_tensor = torch.cat([x_num, x_cat], dim=1)  # [B, d_num + 9]

        
        pairs = []
        if "__pair_id" in df.columns:
            group = df.groupby("__pair_id").indices
            for pid, idxs in group.items():
                if len(idxs) == 2:
                    a, b = idxs
                    pairs.append((int(a), int(b)))
        self.pairs = pairs

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "x": self.x_tensor,
            "pairs": self.pairs,
            "labels": self.labels
        }
