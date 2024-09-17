from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import os
import torch
import sys
from argparse import ArgumentParser
from tqdm import tqdm

SAVE_PATH = "/home/mila/t/thomas.jiralerspong/kolmogorov/scratch/complexity_compositionality/data/real_languages/coco-captions"
LANGUAGES = ["deu_Latn", "fra_Latn", "jpn_Jpan", "spa_Latn", "zho_Hans"]


def get_sentences(language, save_path):
    filename = f"{save_path}/sentences_{language}.txt"
    with open(filename, "r") as file:
        return [line.strip() for line in file]


print("Getting Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
print("Getting SentenceTransformer")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

for LANGUAGE in LANGUAGES:
    print(f"Processing {LANGUAGE}")
    print(f"Getting sentences for {LANGUAGE}")
    sentences = get_sentences(LANGUAGE, SAVE_PATH)

    # Process sentences in batches
    # Process all sentences at once
    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    w = tokens["input_ids"]
    z = model.encode(sentences)
    print("w shape", w.shape)
    print("z shape", z.shape)

    # Save results
    torch.save(w, f"{SAVE_PATH}/w_{LANGUAGE}.pt")
    torch.save(z, f"{SAVE_PATH}/z_{LANGUAGE}.pt")

    print(f"Finished processing {LANGUAGE}")

# # Clear GPU memory
# del tokens, w, z
# torch.cuda.empty_cache()

# # Combine batches after processing
# print("Combining batches...")

# # Combine w tensors
# w_combined = []
# for i in range(0, len(sentences), batch_size):
#     w_batch = torch.load(f"{SAVE_PATH}/w_{LANGUAGE}_batch_{i//batch_size}.pt")
#     w_combined.append(w_batch)
#     os.remove(
#         f"{SAVE_PATH}/w_{LANGUAGE}_batch_{i//batch_size}.pt"
#     )  # Remove individual batch file
# w_combined = torch.cat(w_combined, dim=0)
# torch.save(w_combined, f"{SAVE_PATH}/w_{LANGUAGE}_combined.pt")

# # Combine z tensors
# z_combined = []
# for i in range(0, len(sentences), batch_size):
#     z_batch = torch.load(f"{SAVE_PATH}/z_{LANGUAGE}_batch_{i//batch_size}.pt")
#     z_combined.append(z_batch)
#     os.remove(
#         f"{SAVE_PATH}/z_{LANGUAGE}_batch_{i//batch_size}.pt"
#     )  # Remove individual batch file
# z_combined = torch.cat(z_combined, dim=0)
# torch.save(z_combined, f"{SAVE_PATH}/z_{LANGUAGE}_combined.pt")

# print(f"Combined tensors saved for {LANGUAGE}")
