import json
import logging
import numpy as np
import os

import langid

from lplangid import language_classifier as lc
from experiments import fasttext_client
from experiments.classification_report import nullsafe_classification_report


def langid_classify(text: str):
    return langid.classify(text)[0]


def run_twituser_tests():
    rrc_classifier = lc.RRCLanguageClassifier.many_language_bible_instance()
    ft_classifier = fasttext_client.FastTextLangID()

    fn_labels = [
        [rrc_classifier.get_winner, "RRC"],
        [ft_classifier.predict_lang, "FastText"],
        [langid_classify, "LangID"],
    ]

    for fn, label in fn_labels:
        print(f"Classifying with {label}")
        y_labels, y_pred = [], []
        with open("twituser_data/twituser") as twituser_data:
            for line in twituser_data:
                record = json.loads(line)
                result = fn(record["text"])
                y_pred.append(result)
                y_labels.append(record["lang"])

        print(nullsafe_classification_report(y_labels, y_pred))


def main():
    run_twituser_tests()


if __name__ == "__main__":
    main()
