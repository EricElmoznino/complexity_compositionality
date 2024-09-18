from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import argparse
import os

language_code_to_name = {
    "fra_Latn": "french",
    "deu_Latn": "german",
    "spa_Latn": "spanish",
    "jpn_Jpan": "japanese",
}


def translate_sentences(
    english_sentences,
    language,
    translation_model_id,
    batch_size,
    max_translated_tokens,
):
    tokenizer = AutoTokenizer.from_pretrained(translation_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    translated_sentences = []
    print("Translating sentences...")
    for i in tqdm(range(0, len(english_sentences), batch_size)):
        translated_batch = translate_batch(
            english_sentences[i : i + batch_size],
            tokenizer,
            model,
            device,
            language,
            max_translated_tokens,
        )
        translated_sentences += translated_batch

    for i in range(len(translated_sentences)):
        if translated_sentences[i][-1] == ".":
            translated_sentences[i] = translated_sentences[i][:-1]

    return translated_sentences


def translate_batch(batch, tokenizer, model, device, language, max_translated_tokens):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    translated = model.generate(
        **inputs,
        max_length=max_translated_tokens,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(language),
    )
    translated_sentences = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_sentences


def embed_sentences(sentences, sentence_embedding_model_id):
    # Build w
    tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model_id)
    w = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")[
        "input_ids"
    ]
    unique = torch.unique(w)
    w_short = torch.zeros_like(w)
    for i, u in enumerate(unique):
        w_short[w == u] = i

    # Build z
    print("Embedding sentences...")
    model = SentenceTransformer(sentence_embedding_model_id)
    z = model.encode(sentences, convert_to_numpy=False)
    z = torch.stack(z).cpu()

    return w, w_short, z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate and embed sentences using Hugging Face models"
    )
    parser.add_argument(
        "--language", help="Target language code (e.g., 'fra_Latn' for French)"
    )
    parser.add_argument(
        "--source_folder",
        help="Path to the source data containing an english folder to translate from",
        default="/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/real_languages/coco-captions/",
    )
    parser.add_argument(
        "--translation_model_id",
        help="Name of the Hugging Face model to use for translation",
        default="facebook/nllb-200-3.3B",
    )
    parser.add_argument(
        "--sentence_embedding_model_id",
        help="Name of the Hugging Face model to use for sentence embeddings",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    parser.add_argument(
        "--batch_size", type=int, default=25, help="Batch size for translation"
    )
    parser.add_argument(
        "--max_translated_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens per translated sentence",
    )
    args = parser.parse_args()

    target_dir = os.path.join(args.source_folder, language_code_to_name[args.language])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Translate sentences
    with open(
        os.path.join(args.source_folder, "english", "sentences.txt"),
        "r",
        encoding="utf-8",
    ) as f:
        english_sentences = f.read().split("\n")
    translated_sentences = translate_sentences(
        english_sentences,
        args.language,
        args.translation_model_id,
        args.batch_size,
        args.max_translated_tokens,
    )
    with open(os.path.join(target_dir, "sentences.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(translated_sentences))

    # Embed translated sentences
    w, w_short, z = embed_sentences(
        translated_sentences,
        args.sentence_embedding_model_id,
    )
    torch.save(w, os.path.join(target_dir, "w.pt"))
    torch.save(w_short, os.path.join(target_dir, "w_short.pt"))
    torch.save(z, os.path.join(target_dir, "z.pt"))

    print(f"Translation completed. Output saved to {target_dir}")
