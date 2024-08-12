from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from get_data import get_sentences
import os
import torch

LANGUAGE = "english"
BATCH_SIZE = 100
SAVE_PATH = f"/home/mila/t/thomas.jiralerspong/kolmogorov/scratch/kolmogorov/results/real_languages/{LANGUAGE}"
sentences = get_sentences(LANGUAGE)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
breakpoint()
tokens = tokenizer(sentences, padding=True, truncation=True)
w = torch.tensor(tokens['input_ids'])
zs = []
for i in range(0, len(sentences), BATCH_SIZE):
    end = min(i+BATCH_SIZE, len(sentences))
    batch = sentences[i:end]
    z = model.encode(batch)
    zs.append(torch.tensor(z))
    print(f"Finished batch {i}-{end}")
zs = torch.cat(zs, dim=0)

os.makedirs(SAVE_PATH, exist_ok=True)
torch.save(w, f"{SAVE_PATH}/w.pt")
torch.save(zs, f"{SAVE_PATH}/z.pt")
