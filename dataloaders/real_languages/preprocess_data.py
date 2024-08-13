from constants import PROMPTS_FOLDER
def translate_sentence(sentence, language):
    translated_sentence = sentence
    # TODO: Add DeepL API to translate
    return translated_sentence

def translate_sentences(sentences, language):
    translated_sentences = []
    for sentence in sentences:
        translated_sentence = translate_sentence(sentence, language)
        translated_sentences.append(translated_sentence)
    return translated_sentences

def save_sentences(sentences, language):
    with open(f'{PROMPTS_FOLDER}/{language}_sentences.txt', 'w') as outfile:
        for sentence in sentences:
            outfile.write(sentence)

def get_source_sentences(fname):
    with open(fname) as infile:
        sentences = infile.readlines()
    return sentences


target_languages = ["fr", "de", "jp"]
sentences = get_source_sentences(f"{PROMPTS_FOLDER}/english_sentences.txt")

for language in target_languages:
    translated_sentences = translate_sentences(sentences, language)
    save_sentences(translated_sentences, language)