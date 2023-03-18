import logging
import numpy as np
import os

from lplangid import language_classifier as lc
from experiments import fasttext_client
from experiments.classification_report import nullsafe_classification_report

# The directory with the unzipped files from https://github.com/christos-c/bible-corpus
BIBLE_XML_DIR = "/Users/widdows/Code/bible-corpus/bibles"

# The directory where these will be extracted to raw text files, in full, train, and test directories.
BIBLE_TXT_ROOT = "/Users/widdows/Data/BibleTexts"
SUBDIRS = ["full", "train", "test"]


def run_bible_tests(test_dir, num_trials_per_lang=1000):
    top_languages = {x[:2] for x in os.listdir(lc.FREQ_DATA_DIR)}
    all_term_ranks, all_char_weights = lc.prepare_scoring_tables(lc.FREQ_DATA_DIR)
    rrc_classifier = lc.RRCLanguageClassifier(all_term_ranks, all_char_weights)
    ft_classifier = fasttext_client.FastTextLangID()
    filenames = os.listdir(test_dir)

    total_tests, total_attempted, total_correct = 0, 0, 0
    y_labels, y_pred = [], []
    for filename in filenames:
        attempted, correct = 0, 0
        lang = filename.split(".")[0]

        if lang not in top_languages:
            continue

        available_test_lines = open(os.path.join(test_dir, filename)).readlines()
        if len(available_test_lines) < num_trials_per_lang:
            logging.warning(f"Only {len(available_test_lines)} test lines for language {lang} from file {filename}.")
        test_lines = np.random.choice(available_test_lines, num_trials_per_lang)

        for test_line in test_lines:
            result = rrc_classifier.get_winner(test_line)
            result = ft_classifier.predict_lang(test_line)

            y_pred.append(result)
            y_labels.append(lang)
            if result:
                attempted += 1
                if result == lang:
                    correct += 1
        if correct == 0:
            print(f"Skipping missing language {lang}")
            continue
        print(
            f"Language: {lang} Trials: {len(test_lines)}. Attempted: {attempted}. Correct: {correct}. "
            f"Precision: {correct / attempted:0.3f}. Recall: {correct / len(test_lines)}"
        )
        total_tests += len(test_lines)
        total_attempted += attempted
        total_correct += correct

    print(
        f"All languages. Trials: {num_trials_per_lang}. Attempted: {total_attempted}. Correct: {total_correct}. "
        f"Precision: {total_correct / total_attempted:0.3f}. Recall: {total_correct / total_tests}"
    )

    print(nullsafe_classification_report(y_labels, y_pred))


def main():
    run_bible_tests(os.path.join(BIBLE_TXT_ROOT, SUBDIRS[2]))


if __name__ == "__main__":
    main()
