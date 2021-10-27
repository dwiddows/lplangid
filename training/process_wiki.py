"""
This file contains data preparation / utilities.
It is mainly used when adding new languages to the lplangid classifier.

See the README.md file in this directory for more detailed instructions.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, List, Tuple

from training.data_overrides import TOP_DATA_OVERRIDES, RANKED_DATA_OVERRIDES
from lplangid import language_classifier as lc, count_utils as cu
from lplangid.tokenizer import tokenize_fast as tokenize

"""This should be set to wherever the Wikipedia files are on your system."""
WIKI_TEXT_ROOT = os.path.expanduser('~/Data/Wikipedia/')

SKIP_WORDS_WITH_DIGITS = True
MIN_WORD_LENGTH = 2
MAX_NUM_WORDS_PER_LANGUAGE = 5000

NEW_FREQ_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_freq_data")
ALL_WORDS = -1  # triggers use of all words
ALL_CHARS = -1  # triggers use of all chars
ALL_FILES = -1  # triggers use of all files


def train_test_split(requested_languages=()):
    """Processes the output of WikiExtractor.py (see https://github.com/attardi/wikiextractor) into train and test.

    This is potentially brittle: you should only do this once when adding a new language or updating everything.

    :param requested_languages: List of languages to process. If empty, all available on filesystem will be processed.
    """
    test_proportion = 0.2
    lang_dirs = [lang for lang in os.listdir(WIKI_TEXT_ROOT) if len(lang) == 2]
    if requested_languages:
        missing_languages = [ml for ml in requested_languages if ml not in lang_dirs]
        if missing_languages:
            raise ValueError(
                f"Language '{missing_languages}' was requested but there is no such directory in {WIKI_TEXT_ROOT}")
        lang_dirs = requested_languages
    logging.info(f"Making test train split for languages: {', '.join(lang_dirs)}")

    os.chdir(WIKI_TEXT_ROOT)
    for lang in lang_dirs:
        logging.info(f"\tMaking test train split for language: {lang}")
        os.chdir(lang)
        lang_dir = os.getcwd()
        text_dirs = [x for x in os.listdir(lang_dir) if os.path.isdir(x)]
        if text_dirs != ["text"]:
            logging.info(f"\tExpected just 'text' directory in {os.getcwd()} - instead got {text_dirs}. "
                         f"Skipping the rest of test / train split for language {lang}.")
            continue
        os.mkdir("train")
        os.mkdir("test")
        for subdir in os.listdir("text"):
            os.mkdir(os.path.join("train", subdir))
            os.mkdir(os.path.join("test", subdir))
            all_text_files = sorted(os.listdir(os.path.join("text", subdir)))
            if len(all_text_files) == 0:
                continue
            cutoff = math.floor(len(all_text_files) * (1 - test_proportion))
            train_files = all_text_files[:cutoff]
            for train_file in train_files:
                os.rename(os.path.join("text", subdir, train_file),
                          os.path.join("train", subdir, train_file))
            test_files = all_text_files[cutoff:]
            for testfile in test_files:
                os.rename(os.path.join("text", subdir, testfile),
                          os.path.join("test", subdir, testfile))
            os.rmdir(os.path.join("text", subdir))
        os.rmdir("text")
        os.chdir("../..")


def count_from_text_root(wiki_text_dir):
    """Counts the word and character frequencies in text files beneath the given wiki_text_dir.

    Words are normalized to lowercase, but upper and lowercase characters are counted separately.
    """
    term_freq_dict = {}
    char_freq_dict = {}
    for path, _, filenames in os.walk(wiki_text_dir):
        for filename in filenames:
            filepath = os.path.join(path, filename)
            with open(filepath) as filehandle:
                for line in filehandle:
                    if line.startswith('<'):
                        continue
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


def count_wiki_text_contents(requested_languages: Tuple[str] = ()):
    """
    For each wiki directory (identified by a 2-character language code), counts the words and characters
    and outputs to appropriate rank files (for words) and frequency files (for characters).

    Each wiki directory must have a "train" or a "text" directory with the text files to be counted.

    :param requested_languages: List of languages to process. If empty, all available on filesystem will be processed.
    """
    lang_dirs = [lang for lang in os.listdir(WIKI_TEXT_ROOT) if len(lang) == 2]
    if requested_languages:
        missing_languages = [ml for ml in requested_languages if ml not in lang_dirs]
        if missing_languages:
            raise ValueError(
                f"Language '{missing_languages}' was requested but there is no such directory in {WIKI_TEXT_ROOT}")
        lang_dirs = requested_languages
    logging.info(f"Making count resources for languages: {', '.join(lang_dirs)}")

    for lang in lang_dirs:
        logging.info(f"\tStarting term and character counting for language '{lang}' ...")
        lang_dir_full = os.path.join(WIKI_TEXT_ROOT, lang)
        lang_dir_contents = os.listdir(lang_dir_full)
        text_dir = "train" if "train" in lang_dir_contents else "text" if "text" in lang_dir_contents else None
        if not text_dir:
            raise ValueError(f"No 'train' or 'text' directory in {lang_dir_full}. Please investigate. "
                             f"Check that Wiki archive was uncompressed using bunzip2 "
                             f"and text extracted using WikiExtractor.")

        term_freq_dict, char_freq_dict = count_from_text_root(os.path.join(lang_dir_full, text_dir))
        term_rank_list = [kv[0] for kv in sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)]

        # Write out word in ranked order. The "+ 1000" is in case some are filtered out later.
        cu.write_rank_file(os.path.join(
            lc.FREQ_DATA_DIR, f"{lang}_term_rank.csv"), term_rank_list, max_records=MAX_NUM_WORDS_PER_LANGUAGE+1000)
        cu.write_freq_file(os.path.join(
            lc.FREQ_DATA_DIR, f"{lang}_char_freq.csv"), char_freq_dict)
        logging.info(f"\tFinished counting and writing wiki word ranks and character frequencies for language '{lang}'"
                     f"into directory {lc.FREQ_DATA_DIR}.")


def get_char_lag(all_char_ranks: Dict[str, List[Tuple[str, float]]], lang: str, term: str):
    """Returns a score comparing the given term's char score for the given language
    to the best char score for that term in any language."""
    term_scores = lc.score_chars(all_char_ranks, term)
    if lang not in term_scores or len(term_scores) == 0 or max(term_scores.values()) == 0:
        return 0
    return term_scores[lang] / max(term_scores.values())


def filter_terms_by_chars(requested_languages=()):
    """Rewrites each language's term rank file with only terms whose relative char score is above a threshold.

    :param requested_languages: List of languages to process. If empty, all available in freq_data will be processed.
    """
    char_lag_threshold = 0.2  # Determined empirically by staring at results.

    all_term_ranks, all_char_freqs = lc.prepare_scoring_tables()
    if not requested_languages:
        requested_languages = all_term_ranks.keys()
    for lang in requested_languages:
        term_ranks = all_term_ranks[lang]
        filtered_term_ranks = {}
        for term in term_ranks:
            if get_char_lag(all_char_freqs, lang, term) > char_lag_threshold:
                filtered_term_ranks[term] = term_ranks[term]
        ranked_terms = [kv[0] for kv in sorted(filtered_term_ranks.items(), key=lambda x: x[1])]
        cu.write_rank_file(os.path.join(
            lc.FREQ_DATA_DIR, f"{lang}_term_rank.csv"), ranked_terms, max_records=MAX_NUM_WORDS_PER_LANGUAGE)
        logging.info(f"Rewrote {lang}_term_rank.csv after filtering out {len(term_ranks) - len(ranked_terms)} "
                     f"out of {len(term_ranks)} terms for having unusual characters for language {lang}.")


def add_data_overrides_to_term_ranks(requested_languages=()):
    """Adds data from the data_overrides files to the term rank files in-place.

    Tries to avoid doing this twice by accident, though this is clumsy and should be checked by developers.

    :param requested_languages: List of languages to process. If empty, all available in freq_data will be processed.
    """
    all_term_ranks, _ = lc.prepare_scoring_tables()
    if not requested_languages:
        requested_languages = all_term_ranks.keys()
    for lang in requested_languages:
        if lang not in RANKED_DATA_OVERRIDES and lang not in TOP_DATA_OVERRIDES:
            logging.info(f"There are no listed data_overrides for language {lang}. Skipping this step.")
            continue

        old_ranks = all_term_ranks[lang]
        # Turn the rank dictionary into a list for easier manipulation rather than fast lookup.
        term_list = [x[0] for x in sorted(old_ranks.items(), key=lambda kv: kv[1])]
        if lang in RANKED_DATA_OVERRIDES:
            if not all([term in old_ranks for term in RANKED_DATA_OVERRIDES[lang]]):
                for new_term, rank in sorted(RANKED_DATA_OVERRIDES[lang].items(), key=lambda kv: kv[1], reverse=True):
                    term_list.insert(rank, new_term)
            else:
                logging.info(f"It is likely that ranked overrides were already added for language {lang}. Skipping.")

        if lang in TOP_DATA_OVERRIDES:
            if term_list[0] != TOP_DATA_OVERRIDES[lang][0]:
                term_list = TOP_DATA_OVERRIDES[lang] + term_list
            else:
                logging.info(f"It is likely that ranked overrides were already added for language {lang}. Skipping.")

        cu.write_rank_file(os.path.join(lc.FREQ_DATA_DIR, f"{lang}_term_rank.csv"), term_list)
        logging.info(f"Rewrote {lang}_term_rank.csv after adding data overrides.")


def main(argv):
    """Main function that takes a directory of text contents (e.g., from an uncompressed wiki archive), and
    - Splits the text files into train and test directories.
    - Counts the terms and characters in each language directory.
    - Filters out terms that don't fit the character profile of a language.
    - Adds known extra terms (data_overrides) to the term rank tables.
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", help="Comma separated list of language codes, e.g., --languages=en,es,pt",
                        type=str)

    args = parser.parse_args(argv)
    requested_languages = args.languages.split(',') if args.languages else None

    logging.info("Starting processing wiki files.")
    if requested_languages:
        logging.info(f"Explicitly requested these languages: {', '.join(requested_languages)}")

    train_test_split(requested_languages=requested_languages)
    count_wiki_text_contents(requested_languages=requested_languages)
    filter_terms_by_chars(requested_languages=requested_languages)
    add_data_overrides_to_term_ranks(requested_languages=requested_languages)


if __name__ == '__main__':
    main(sys.argv[1:])
