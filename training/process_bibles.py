"""
Creates term and character files from many translations of bibles to build a classifier.

Uses data and some example instructions from https://github.com/christos-c/bible-corpus.

Includes several hard-coded paths from Dominic's machine.
"""
from collections import defaultdict
import logging
import numpy as np
import os
from pathlib import Path
from typing import TextIO

import xml.etree.ElementTree as ET

from lplangid import count_utils
from lplangid import language_classifier as lc
from lplangid.tokenizer import tokenize_fast as tokenize
from process_wiki import MIN_WORD_LENGTH, SKIP_WORDS_WITH_DIGITS, WIKI_TEXT_ROOT

# The directory with the unzipped files from https://github.com/christos-c/bible-corpus
BIBLE_XML_DIR = '/Users/widdows/Code/bible-corpus/bibles'

# The directory where these will be extracted to raw text files, in full, train, and test directories.
BIBLE_TXT_ROOT = '/Users/widdows/Data/BibleTexts'
SUBDIRS = ['full', 'train', 'test']


def process_bibles_xml_to_text(corpus_dir=BIBLE_XML_DIR,
                               out_dir_root=BIBLE_TXT_ROOT):
    all_files = os.listdir(corpus_dir)
    plain_files = [fn for fn in all_files if '-tok' not in fn and '-WEB' not in fn]
    langs = defaultdict(list)

    Path(BIBLE_TXT_ROOT).mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        Path(os.path.join(BIBLE_TXT_ROOT, subdir)).mkdir(parents=True, exist_ok=True)

    for fn in plain_files:
        root = ET.fromstring(open(os.path.join(corpus_dir, fn)).read())
        lang_ele = root.find('.//language')
        lang_id = lang_ele.attrib['id']
        lang_name = lang_ele.text.strip()
        langs[lang_id].append(fn)
        if len(langs[lang_id]) > 1:
            logging.warning(f"Already seen language '{lang_id}' ({lang_name}) in files {langs[lang_id]}")

        out_full, out_train, out_test = [
            open(os.path.join(BIBLE_TXT_ROOT, subdir, lang_id + '.txt'), 'w', encoding='utf-8')
            for subdir in SUBDIRS
        ]
        for i, n in enumerate(root.iter('seg')):
            try:
                out_full.write(n.text.strip() + '\n')
                # Implements an 80:20 train:test split.
                if i % 5 == 4:
                    out_test.write(n.text.strip() + '\n')
                else:
                    out_train.write(n.text.strip() + '\n')
            except AttributeError:
                logging.warning(f"Problem in file {fn} with element {str(n)}")


def count_text_in_input(filehandle: TextIO):
    term_freq_dict = defaultdict(int)
    char_freq_dict = defaultdict(int)
    for line in filehandle:
        for char in [char for char in line if char.isalpha()]:
            char_freq_dict[char] += 1
            if char.lower() != char:
                char_freq_dict[char.lower()] += 1
        for word in tokenize(line):
            if len(word) < MIN_WORD_LENGTH:
                continue
            if SKIP_WORDS_WITH_DIGITS and any([x.isdigit() for x in word]):
                continue
            term_freq_dict[word.lower()] += 1
    return term_freq_dict, char_freq_dict


def text_files_to_freq_files(input_dir, output_dir=lc.FREQ_DATA_DIR + '_bible'):
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


def run_bible_tests(test_dir, num_trials_per_lang=1000):
    all_term_ranks, all_char_weights = lc.prepare_scoring_tables(lc.FREQ_DATA_DIR + '_bible')
    classifier = lc.RRCLanguageClassifier(all_term_ranks, all_char_weights)
    filenames = os.listdir(test_dir)

    total_tests, total_attempted, total_correct = 0, 0, 0
    for filename in filenames:
        attempted, correct = 0, 0
        lang = filename.split('.')[0]

        available_test_lines = open(os.path.join(test_dir, filename)).readlines()
        if len(available_test_lines) < num_trials_per_lang:
            logging.warning(f"Only {len(available_test_lines)} test lines for language {lang} from file {filename}.")
        test_lines = np.random.choice(available_test_lines, num_trials_per_lang)

        for test_line in test_lines:
            results = classifier.get_language_scores(test_line)[:5]
            if len(results) > 0:
                attempted += 1
                if results[0][0] == lang:
                    correct += 1
        print(f'Language: {lang} Trials: {len(test_lines)}. Attempted: {attempted}. Correct: {correct}. '
              f'Precision: {correct / attempted:0.3f}. Recall: {correct / len(test_lines)}')
        total_tests += len(test_lines)
        total_attempted += attempted
        total_correct += correct

    print(f'All languages. Trials: {num_trials_per_lang}. Attempted: {total_attempted}. Correct: {total_correct}. '
          f'Precision: {total_correct / total_attempted:0.3f}. Recall: {total_correct / total_tests}')


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
    retrain = False
    if retrain:
        logging.basicConfig(level=logging.INFO)
        process_bibles_xml_to_text()
        text_files_to_freq_files(os.path.join(BIBLE_TXT_ROOT, SUBDIRS[1]))

    run_bible_tests(os.path.join(BIBLE_TXT_ROOT, SUBDIRS[2]))

    # print("Restricting to Wiki languages:")
    # run_wikipedia_tests(restrict_to_wiki_langs=True)
    # print("Selecting from all available languages:")
    # run_wikipedia_tests()


if __name__ == '__main__':
    main()
