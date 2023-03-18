import os
import shutil

import langid

bible_files = '/Users/widdows/Data/BibleTexts/train/'
corpus_dir = '/Users/widdows/Data/LangidTrain/'


def make_corpus():
    for bible_text in os.listdir(bible_files):
        lang = bible_text.split(".")[0]
        os.makedirs(os.path.join(corpus_dir, lang), exist_ok=True)
        shutil.copyfile(os.path.join(bible_files, bible_text), os.path.join(corpus_dir, lang, bible_text))

