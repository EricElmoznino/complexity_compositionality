#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:29:42 2019

@author: xiayezi
"""

import argparse
import random
import numpy as np
import math
import os
import itertools

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch import autograd


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="the path to save the results")
parser.add_argument(
    "--noresets", action="store_true", help="whether to use iterated learning"
)
parser.add_argument("--wandb_entity", type=str, default="yoshua-bengio")
parser.add_argument("--tau", type=int, default=2, help="for gumble softmax")
parser.add_argument("--seed", type=int, default=12345, help="random seed")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument(
    "--lr_ratio", type=float, default=1, help="lr for listener is lr*lr_ratio"
)
parser.add_argument(
    "--sel_candi", type=int, default=15, help="candiates for the listene"
)
parser.add_argument("--num_attr", type=int, default=2, help="number of attributes")
parser.add_argument("--num_sys", type=int, default=8, help="number of attribute values")
parser.add_argument("--voc_len_add", type=int, default=0, help="added vocabulary size")
parser.add_argument("--msg_len_add", type=int, default=0, help="added msg length size")
parser.add_argument("--repr_dim", type=int, default=128, help="representation dim")
parser.add_argument("--Ia", type=int, default=1200, help="pretrain rounds of Alice")
parser.add_argument("--Ib", type=int, default=200, help="pretrain rounds of Bob")
parser.add_argument("--Ig", type=int, default=3000, help="rounds of interaction")
parser.add_argument("--valid_num", type=int, default=8, help="size of validation set")
parser.add_argument("--max_gen", type=int, default=80, help="max generations")
parser.add_argument(
    "--pairs_teach", type=int, default=100, help="sampled pairs for listener pretrain"
)
parser.add_argument("--n_object_save_repeats", type=int, default=10)

args = parser.parse_args()


setup_seed(args.seed)  # 12345 is valid for N_B=100, SEL_CAN = 5


"""
for training model
"""
RESETS = False if args.noresets else True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = args.lr  # learning rate
CLIP = 50.0  # max after clipping gradient
DECODER_LEARING_RATIO = args.lr_ratio
OPTIMISER = optim.Adam


"""
hyperparameters of model
"""
SEL_CANDID = args.sel_candi  # Number of candidate when selecting
ATTRI_SIZE = args.num_attr  # Number of attributes, i.e., number of digits
NUM_SYSTEM = args.num_sys  # Number system, usually just use decimal
HIDDEN_SIZE = args.repr_dim  # Hidden size of the model
BATCH_SIZE = 64
MSG_MAX_LEN = ATTRI_SIZE + args.msg_len_add  # Controlled by ourselves
VALID_NUM = args.valid_num  # Ratio of valid set to train set

# Size of vocabulary this is available for communication
MSG_VOCSIZE = NUM_SYSTEM + args.voc_len_add
MSG_MODE = "REINFORCE"  # 'GUMBEL' or 'REINFORCE'
MSG_HARD = True  # Discretized as one-hot vectors

WANDB_ENTITY = args.wandb_entity
