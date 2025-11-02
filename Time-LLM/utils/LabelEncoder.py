import pandas as pd
import json
import numpy as np
import torch

label_dict = 'C:\project\Time-LLM\dataset\label.json'

def encode_labels(label_dict, df_dataLabel):
    with open(label_dict, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)

    # 使用自定义的字典进行标签编码
    df_dataLabel['label'] = df_dataLabel['label'].apply(lambda x: label_dict.get(x, -1))
    return df_dataLabel

    # 数字替换为标签(是batch）


def decode_labels(label_dict_path, df_dataLabel):
    # 读取标签字典
    with open(label_dict_path, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)

    # 创建反向字典
    reverse_label_dict = {v: k for k, v in label_dict.items()}

    # 确保 df_dataLabel 是一个在 CPU 上的 Tensor
    if isinstance(df_dataLabel, torch.Tensor):
        df_dataLabel = df_dataLabel.cpu().numpy()  # 将 GPU Tensor 转换为 NumPy 数组

    # 使用 NumPy 向量化操作进行解码
    decoded_labels = np.vectorize(reverse_label_dict.get)(df_dataLabel)
    return decoded_labels