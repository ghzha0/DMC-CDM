import numpy as np
import json
import os
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve

def record_res(config, final_res, save_name='experiment'):
    new_data = {
        "dataset": config['dataset'],
        "test_results": {
            "acc": np.mean(final_res['final_acc']),
            "auc": np.mean(final_res['final_auc']),
            "f1": np.mean(final_res['final_f1'])
        }
    }

    if os.path.exists(f"{save_name}.json"):
        with open(f"{save_name}.json", "r") as file:
            data = json.load(file)
    else:
        data = []

    # Append new data
    data.append(new_data)

    # Write back to the JSON file
    with open(f"{save_name}.json", "w") as file:
        json.dump(data, file, indent=4)

def compute_metrics(y_true, y_pred):
    # 计算每个阈值下的精确率和召回率
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    # 计算每个阈值下的F1分数
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    # 找到F1分数最大的阈值
    optimal_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_index]
    Pred = (y_pred >= optimal_threshold).astype(int)
    return accuracy_score(y_true=y_true, y_pred=Pred), roc_auc_score(y_true=y_true, y_score=y_pred), \
            f1_score(y_true=y_true, y_pred=Pred), np.sqrt(np.mean((Pred - y_true) ** 2))