import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


# Script arguments
save_dir = "/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/real_languages"
dataset_id = "sentence-transformers/coco-captions"
model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
save_text = True


def dataset_preprocess(dataset):
    # You can change this based on different dataset_id's
    dataset = dataset["caption1"]
    dataset = [d.strip() for d in dataset]
    dataset = [d for d in dataset if "\n" not in d]
    for i in range(len(dataset)):
        if dataset[i][-1] == ".":
            dataset[i] = dataset[i][:-1]
    return dataset


# Load dataset
dataset_name = dataset_id.split("/")[-1]
assert not os.path.exists(f"{save_dir}/{dataset_name}")
dataset = load_dataset(dataset_id, split="train")
dataset = dataset_preprocess(dataset)

# Build w
tokenizer = AutoTokenizer.from_pretrained(model_id)
w = tokenizer(dataset, padding=True, truncation=True, return_tensors="pt")["input_ids"]
unique = torch.unique(w)
w_short = torch.zeros_like(w)
for i, u in enumerate(unique):
    w_short[w == u] = i

# Build z
model = SentenceTransformer(model_id)
z = model.encode(dataset, convert_to_numpy=False)
z = torch.stack(z).cpu()


# Save results
os.mkdir(f"{save_dir}/{dataset_name}")
torch.save(w, f"{save_dir}/{dataset_name}/w.pt")
torch.save(w_short, f"{save_dir}/{dataset_name}/w_short.pt")
torch.save(z, f"{save_dir}/{dataset_name}/z.pt")
if save_text:
    with open(f"{save_dir}/{dataset_name}/sentences.txt", "w") as f:
        f.write("\n".join(dataset))
