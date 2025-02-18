# The part of code is adapted from https://github.com/liuqidong07/LLM-ESR.
import numpy as np
import os
import torch
import torch.nn as nn
import pickle
from .cross_attention import Backbone

class collabrative_extractor(nn.Module):
    def __init__(self, ce_embed_size, ce_dropout, item_num):
        super(collabrative_extractor, self).__init__()
        self.item_num = item_num
        self.item_emb = torch.nn.Embedding(self.item_num+1, ce_embed_size, padding_idx=item_num)
        self.emb_dropout = torch.nn.Dropout(p=ce_dropout)

    def get_problem_embedding(self, log_seqs):
        item_seq_emb = self.item_emb(log_seqs)
        return item_seq_emb
