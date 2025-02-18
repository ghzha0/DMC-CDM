import json
import torch
from random import sample, shuffle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def left_pad_sequences(sequences, padding_value=0):
    max_length = min(max(seq.shape[-1] for seq in sequences), 256)
    padded_sequences = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        if seq.shape[-1] > max_length:
            seq = seq[:max_length]
        padded_sequences[i, -seq.shape[-1]:] = seq
    return padded_sequences

def get_dataset(cfg):
    if cfg['dataname'] == 'moocradar':
        total = 3
        test_num = cfg['test_num']
        dataname = cfg['dataname']        
        exercise0 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+1)%total}.csv")
        exercise1 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+2)%total}.csv")
        exercise2 = pd.read_csv(f"./data/{dataname}/exercise{(test_num)%total}.csv")

        user_num = -1
        with open(f'./data/{dataname}/user_id_mapping.csv', 'r', encoding='utf-8') as rf:
            for _ in rf.readlines():
                user_num += 1

        item_num = -1
        with open(f'./data/{dataname}/problem_id_mapping.csv', 'r', encoding='utf-8') as rf:
            for _ in rf.readlines():
                item_num += 1

        exercise = pd.concat([exercise0, exercise1, exercise2]).groupby('student_id').apply(lambda x: [x['question_id'].to_numpy(), x['correct'].to_numpy()]).to_dict()
    elif cfg['dataname'] == 'ifly':
        total = 6
        test_num = cfg['test_num']
        dataname = cfg['dataname']
        exercise0 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+1)%total}.csv")
        exercise1 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+2)%total}.csv")
        exercise2 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+3)%total}.csv")
        exercise3 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+4)%total}.csv")
        exercise4 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+5)%total}.csv")
        exercise5 = pd.read_csv(f"./data/{dataname}/exercise{(test_num)%total}.csv")

        user_num = -1
        with open(f'./data/{dataname}/user_id_mapping.csv', 'r', encoding='utf-8') as rf:
            for _ in rf.readlines():
                user_num += 1

        item_num = -1
        with open(f'./data/{dataname}/problem_id_mapping.csv', 'r', encoding='utf-8') as rf:
            for _ in rf.readlines():
                item_num += 1
        exercise = pd.concat([exercise0, exercise1, exercise2, exercise3, exercise4, exercise5]).groupby('user_id').apply(lambda x: [x['exer_id'].to_numpy(), x['score'].to_numpy()]).to_dict()
    elif cfg['dataname'] == 'junyi':
        total = 3
        test_num = cfg['test_num']
        dataname = cfg['dataname']
        exercise0 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+1)%total}.csv")
        exercise1 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+2)%total}.csv")
        exercise2 = pd.read_csv(f"./data/{dataname}/exercise{(test_num+3)%total}.csv")
        user_num = -1
        with open(f'./data/{dataname}/user_id_mapping.csv', 'r', encoding='utf-8') as rf:
            for _ in rf.readlines():
                user_num += 1

        item_num = -1
        with open(f'./data/{dataname}/problem_id_mapping.csv', 'r', encoding='utf-8') as rf:
            for _ in rf.readlines():
                item_num += 1

        exercise = pd.concat([exercise0, exercise1, exercise2]).groupby('student_id').apply(lambda x: [x['problem_id'].to_numpy(), x['correct'].to_numpy()]).to_dict()

    train_seq, train_ans, eval_seq, eval_ans, test_seq, test_ans = [], [], [], [], [], []
    for user in range(user_num):
        index = np.arange(len(exercise[user][0]))
        shuffle(index)
        exercise[user][0] = exercise[user][0][index]
        exercise[user][1] = exercise[user][1][index]

        all_length = len(exercise[user][0])
        eval_len = max(int(all_length * 0.1), 1)
        train_len = all_length - 2 * eval_len

        if cfg['sparsity'] > 0:
            train_len = max(train_len - int(train_len * cfg['sparsity']), 1)

        train_seq.append(torch.LongTensor(exercise[user][0][:train_len]))
        train_ans.append(torch.LongTensor([i - abs(i-1) for i in exercise[user][1]][:train_len]))
        eval_seq.append(torch.LongTensor(exercise[user][0][train_len: train_len + eval_len]))
        eval_ans.append(torch.LongTensor([i - abs(i-1) for i in exercise[user][1]][train_len: train_len + eval_len]))
        test_seq.append(torch.LongTensor(exercise[user][0][train_len + eval_len:]))
        test_ans.append(torch.LongTensor([i - abs(i-1) for i in exercise[user][1]][train_len + eval_len:]))
        
    train_seq = left_pad_sequences(train_seq, padding_value=item_num)
    eval_seq = left_pad_sequences(eval_seq, padding_value=item_num)
    test_seq = left_pad_sequences(test_seq, padding_value=item_num)
    train_ans = left_pad_sequences(train_ans, padding_value=0)
    eval_ans = left_pad_sequences(eval_ans, padding_value=0)
    test_ans = left_pad_sequences(test_ans, padding_value=0)
    return train_dataset(train_seq, eval_seq, test_seq, train_ans, eval_ans, test_ans, user_num), user_num, item_num

class train_dataset(Dataset):
    def __init__(self, train_seq, eval_seq, test_seq, train_ans, eval_ans, test_ans, length):
        super().__init__()
        self.train_seq = train_seq
        self.train_ans = train_ans
        self.eval_seq = eval_seq
        self.eval_ans = eval_ans
        self.test_seq = test_seq
        self.test_ans = test_ans
        self.length = length

    def __getitem__(self, index):
        return {
            'train_seq': self.train_seq[index],
            'train_ans': self.train_ans[index],
            'eval_seq': self.eval_seq[index],
            'eval_ans': self.eval_ans[index],
            'test_seq': self.test_seq[index],
            'test_ans': self.test_ans[index]
        }

    def __len__(self):
        return self.length