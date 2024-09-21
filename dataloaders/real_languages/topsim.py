import sys
from itertools import combinations
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

# Script arguments
num_samples = 1000
num_repeats = 10
languages = ["english", "french", "german", "spanish", "japanese"]
dataset = "coco-captions"


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


def language_topsim(language):
    print(f"Language: {language}")

    data_dir = f"/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/real_languages/{dataset}/{language}/"

    with open(data_dir + "sentences.txt", "r", encoding="utf-8") as f:
        w = f.read().split("\n")
    z = torch.load(data_dir + "z.pt").cpu().numpy()

    topsims = []
    for i in range(num_repeats):
        print(f"Repeat {i + 1}/{num_repeats}")

        idxs = np.random.choice(len(w), num_samples, replace=False)
        w_sample = [w[i] for i in idxs]
        z_sample = z[idxs]

        num_pair_entries = len(w_sample) * (len(w_sample) - 1) // 2
        w_distmat = np.zeros(num_pair_entries)
        z_distmat = np.zeros(num_pair_entries)

        for i, (i1, i2) in tqdm(
            enumerate(combinations(range(len(w_sample)), 2)),
            total=num_pair_entries,
        ):
            w_distmat[i] = edit_dist(w_sample[i1], w_sample[i2])
            z_distmat[i] = np.linalg.norm(z_sample[i1] - z_sample[i2])

        topsim = np.corrcoef(w_distmat, z_distmat)[0, 1]
        topsims.append(topsim)

    return topsims


all_language_topsims = []
for language in languages:
    topsims = language_topsim(language)
    for t in topsims:
        all_language_topsims.append({"language": language, "topsim": t})
    print(f"Topological similarity for language: {language}")
    print(f"Average: {np.mean(topsims):0.5g} ± {np.std(topsims):0.5g}")
    print(f"All values: {topsims}")
    print("\n")
all_language_topsims = pd.DataFrame(all_language_topsims)
all_language_topsims.to_csv("data/real_languages/topsims.csv", index=False)
