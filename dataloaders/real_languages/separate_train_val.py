import torch

w = torch.load("/home/mila/t/thomas.jiralerspong/kolmogorov/scratch/complexity_compositionality/data/real_languages/coco-captions/w.pt")
z = torch.load("/home/mila/t/thomas.jiralerspong/kolmogorov/scratch/complexity_compositionality/data/real_languages/coco-captions/z.pt")
import os
from sklearn.model_selection import train_test_split

# Define the directory path
data_dir = "/home/mila/t/thomas.jiralerspong/kolmogorov/scratch/complexity_compositionality/data/real_languages/coco-captions/"

# Define the validation set size (e.g., 20% of the data)
val_size = 0.2

# Split the data
w_train, w_val, z_train, z_val = train_test_split(w, z, test_size=val_size, random_state=42)

# Save the train and validation sets
torch.save(w_train, os.path.join(data_dir, "w_train.pt"))
torch.save(w_val, os.path.join(data_dir, "w_val.pt"))
torch.save(z_train, os.path.join(data_dir, "z_train.pt"))
torch.save(z_val, os.path.join(data_dir, "z_val.pt"))

print(f"Train set size: {len(w_train)}")
print(f"Validation set size: {len(w_val)}")

breakpoint()
print(w.shape)
print(z.shape)

