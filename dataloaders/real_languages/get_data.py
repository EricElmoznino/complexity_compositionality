from constants import PROMPTS_FOLDER
def get_sentences(language):
    with open(f'{PROMPTS_FOLDER}/{language}_sentences.txt') as infile:
        sentences = infile.readlines()
    return sentences