from simulated.data_generators import UniformDataGenerator
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

scratch = os.environ["SCRATCH"]
data_folder_path = f"{scratch}/complexity_compositionality/data/simulated/uniform"
os.makedirs(data_folder_path, exist_ok=True)

n_samples = 100000
compositionality_dict = {}
for vocab_size in [10]:
for vocab_size in [10, 100]:
    for d in [256, 512, 1024]:
        for k in [2, 4, 8]:
            if k > d:
                continue
            data_generator = UniformDataGenerator(k=k, d=d, vocab_size=vocab_size)
            w, z = data_generator.sample(n_samples)
            z_train, z_test, w_train, w_test = train_test_split(z, w, test_size=0.2)
            key_string = f"vocab_size={vocab_size}_d={d}_k={k}_nsamples={n_samples}"
            compositionality_dict[key_string] = data_generator.compositionality(z, w)
            os.makedirs(f"{data_folder_path}/{key_string}", exist_ok=True)

            torch.save(
                torch.tensor(w_train), f"{data_folder_path}/{key_string}/w_train.pt"
            )
            torch.save(
                torch.tensor(z_train), f"{data_folder_path}/{key_string}/z_train.pt"
            )
            torch.save(
                torch.tensor(w_test), f"{data_folder_path}/{key_string}/w_test.pt"
            )
            torch.save(
                torch.tensor(z_test), f"{data_folder_path}/{key_string}/z_test.pt"
            )

df = pd.DataFrame.from_dict(compositionality_dict, orient="index")
df.to_csv(f"{data_folder_path}/compositionality_dict.csv")
