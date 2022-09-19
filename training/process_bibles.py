"""
Creates term and character files from many translations of bibles to build a classifier.

Uses data and some example instructions from https://github.com/christos-c/bible-corpus.

Includes several hard-coded paths from Dominic's machine.
"""
from collections import defaultdict
import logging
import numpy as np
import os
from typing import TextIO

import xml.etree.ElementTree as ET

from lplangid import count_utils
from lplangid import language_classifier as lc
from lplangid.tokenizer import tokenize_fast as tokenize
from process_wiki import MIN_WORD_LENGTH, SKIP_WORDS_WITH_DIGITS, WIKI_TEXT_ROOT


def process_bibles_xml_to_text(corpus_dir='/Users/widdows/Code/bible-corpus/bibles',
                               out_dir='/Users/widdows/Data/BibleTexts'):
    all_files = os.listdir(corpus_dir)
    plain_files = [fn for fn in all_files if '-tok' not in fn and '-WEB' not in fn]
    langs = defaultdict(list)
    for fn in plain_files:
        root = ET.fromstring(open(os.path.join(corpus_dir, fn)).read())
        lang_ele = root.find('.//language')
        lang_id = lang_ele.attrib['id']
        lang_name = lang_ele.text.strip()
        langs[lang_id].append(fn)
        if len(langs[lang_id]) > 1:
            logging.warning(f"Already seen language '{lang_id}' ({lang_name}) in files {langs[lang_id]}")
        with open(os.path.join(out_dir, lang_id + '.txt'), 'w', encoding='utf-8') as out:
            for n in root.iter('seg'):
                try:
                    out.write(n.text.strip() + '\n')
                except AttributeError:
                    logging.warning(f"Problem in file {fn} with element {str(n)}")


def count_text_in_input(filehandle: TextIO):
    term_freq_dict = {}
    char_freq_dict = {}
    for line in filehandle:
        for char in [char for char in line if char.isalpha()]:
            if char.isalpha():
                char_freq_dict[char] = char_freq_dict.get(char, 0) + 1
        for word in tokenize(line):
            if len(word) < MIN_WORD_LENGTH:
                continue
            if SKIP_WORDS_WITH_DIGITS and any([x.isdigit() for x in word]):
                continue
            term_freq_dict[word.lower()] = term_freq_dict.get(word.lower(), 0) + 1
    return term_freq_dict, char_freq_dict


def text_files_to_freq_files(input_dir='/Users/widdows/Data/BibleTexts', output_dir=lc.FREQ_DATA_DIR + '_bible'):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for infile in os.listdir(input_dir):
        lang = infile.split('.')[0]
        with open(os.path.join(input_dir, infile)) as filehandle:
            term_freq_dict, char_freq_dict = count_text_in_input(filehandle)

        term_rank_list = [kv[0] for kv in sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)]

        # Write out word in ranked order. The "+ 1000" is in case some are filtered out later.
        count_utils.write_rank_file(os.path.join(
            output_dir, f"{lang}_term_rank.csv"), term_rank_list, max_records=lc.MAX_WORDS_PER_LANG + 1000)
        count_utils.write_freq_file(os.path.join(
            output_dir, f"{lang}_char_freq.csv"), char_freq_dict)
        logging.info(f"\tWrote {min(lc.MAX_WORDS_PER_LANG + 1000, len(term_rank_list))} bible word ranks and "
                     f"{len(char_freq_dict)} character frequencies for language '{lang}' into directory {output_dir}.")


def run_wikipedia_tests(num_trials=10000, restrict_to_wiki_langs=False):
    all_term_ranks, all_char_weights = lc.prepare_scoring_tables(lc.FREQ_DATA_DIR + '_bible')
    wiki_langs = os.listdir(WIKI_TEXT_ROOT)

    if restrict_to_wiki_langs:
        all_term_ranks = {k: v for k, v in all_term_ranks.items() if k in wiki_langs}
        all_char_weights = {k: [(lang, score) for lang, score in vals if lang in wiki_langs]
                            for k, vals in all_char_weights.items()}
    classifier = lc.RRCLanguageClassifier(all_term_ranks, all_char_weights)

    min_line_length = 32
    correct = 0
    attempted = 0

    for _ in range(num_trials):
        sample_lines = []
        lang = np.random.choice(wiki_langs)
        while len(sample_lines) == 0:
            try:
                wiki_dir = np.random.choice(os.listdir(os.path.join(WIKI_TEXT_ROOT, lang, 'test')))
                wiki_path = os.path.join(WIKI_TEXT_ROOT, lang, 'test', wiki_dir)
                wiki_file = np.random.choice(os.listdir(wiki_path))
                sample_lines = [line for line in open(os.path.join(wiki_path, wiki_file)).readlines()[1:]
                                if len(line) >= min_line_length and not line.startswith('<')]
            except FileNotFoundError:
                logging.warning(f"Failed to find Wiki test files for language: {lang}")
        this_line = np.random.choice(sample_lines).strip()
        results = classifier.get_language_scores(this_line)[:5]
        logging.debug(f'{lang}: {this_line}')
        logging.debug(results)
        if len(results) > 0:
            attempted += 1
            if results[0][0] == lang:
                correct += 1

    print(f'Trials: {num_trials}. Attempted: {attempted}. Correct: {correct}. '
          f'Precision: {correct/attempted:0.3f}. Recall: {correct/num_trials}')


def main():
    # Some of these steps are optional, depending on what you're trying to do.
    logging.basicConfig(level=logging.INFO)
    process_bibles_xml_to_text()
    text_files_to_freq_files()
    print("Restricting to Wiki languages:")
    run_wikipedia_tests(restrict_to_wiki_langs=True)
    print("Selecting from all available languages:")
    run_wikipedia_tests()


if __name__ == '__main__':
    main()
