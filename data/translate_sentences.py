from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
import argparse


def translate_sentences(
    model_name, source_file, target_file, batch_size=100, language="en"
):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Open the target file in append mode
    with open(target_file, "a", encoding="utf-8") as f_out:
        # Read and translate sentences in batches
        with open(source_file, "r", encoding="utf-8") as f_in:
            batch = []
            for sentence in tqdm(f_in, desc="Translating"):
                batch.append(sentence.strip())
                if len(batch) == batch_size:
                    translated_batch = translate_batch(
                        batch, tokenizer, model, device, language
                    )
                    f_out.writelines(translated_batch)
                    batch = []

            # Translate any remaining sentences
            if batch:
                translated_batch = translate_batch(batch, tokenizer, model, device)
                f_out.writelines(translated_batch)


def translate_batch(batch, tokenizer, model, device, language):
    inputs = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    translated = model.generate(
        **inputs,
        max_length=128,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(language),
    )
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return [text + "\n" for text in translated_texts]


# langdetect

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate sentences using a Hugging Face model"
    )
    parser.add_argument("--model_name", help="Name of the Hugging Face model to use")
    parser.add_argument(
        "--language", help="Target language code (e.g., 'fr' for French)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for translation"
    )
    args = parser.parse_args()

    source_file = "/home/mila/t/thomas.jiralerspong/kolmogorov/scratch/complexity_compositionality/data/real_languages/coco-captions/sentences.txt"
    target_file = f"/home/mila/t/thomas.jiralerspong/kolmogorov/scratch/complexity_compositionality/data/real_languages/coco-captions/sentences_{args.language}.txt"

    translate_sentences(
        args.model_name, source_file, target_file, args.batch_size, args.language
    )
    print(f"Translation completed. Output saved to {target_file}")
