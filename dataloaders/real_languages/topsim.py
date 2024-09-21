import os
from itertools import combinations
from tqdm import tqdm
import numpy as np
import torch

# Script arguments
data_dir = "/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/real_languages/coco-captions/english/"


def edit_dist(str1, str2):
    """
    Calculate the edit distance of two strings.
    Insert/delete/replace all cause 1.
    """
    len1, len2 = len(str1), len(str2)
    DM = [0]
    for i in range(len1):
        DM.append(i + 1)

    for j in range(len2):
        DM_new = [j + 1]
        for i in range(len1):
            tmp = 0 if str1[i] == str2[j] else 1
            new = min(DM[i + 1] + 1, DM_new[i] + 1, DM[i] + tmp)
            DM_new.append(new)
        DM = DM_new

    return DM[-1]


with open(data_dir + "sentences.txt", "r", encoding="utf-8") as f:
    w = f.read().split("\n")
z = torch.load(data_dir + "z.pt").cpu().numpy()

pair_indices = list(combinations(range(len(w)), 2))
w_distmat = np.zeros(len(pair_indices))
z_distmat = np.zeros(len(pair_indices))

for pair in tqdm(pair_indices):
    i, j = pair
    w_distmat[pair] = edit_dist(w[i], w[j])
    z_distmat[pair] = np.linalg.norm(z[i] - z[j])

topsim = np.corrcoef(w_distmat, z_distmat)[0, 1]
print(f"Topological similarity: {topsim:0.5g}")
