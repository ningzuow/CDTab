import pandas as pd
import json
import os

def generate_info_json(data_path, test_path=None, val_path=None, task_type="binclass"):
    df = pd.read_csv(data_path)
    column_names = list(df.columns)
    num_col_idx = []
    cat_col_idx = []
    column_info = {}

    for idx, column in enumerate(df.columns):
        if df[column].dtype in ['int64', 'float64']:
            num_col_idx.append(idx)
            column_info[column] = "float"
        elif df[column].dtype == 'object':
            cat_col_idx.append(idx)
            column_info[column] = "str"
        else:
            column_info[column] = "unknown"

    target_col_idx = [len(df.columns) - 1]
    train_num = len(df)
    test_num = 0
    if test_path:
        df_test = pd.read_csv(test_path)
        test_num = len(df_test)

    val_num = 0
    if val_path:
        df_val = pd.read_csv(val_path)
        val_num = len(df_val)

    info = {
        "name": os.path.basename(data_path).split('.')[0],  # 基于文件名提取数据集名称
        "task_type": task_type,
        "header": None,
        "column_names": column_names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": data_path,
        "val_path": val_path,
        "test_path": test_path,
        "column_info": column_info,
        "train_num": train_num,
        "test_num": test_num,
    }

    info_json_path = data_path.replace(".csv", "_info.json")
    with open(info_json_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

    print(f"info.json has been saved to {info_json_path}")


