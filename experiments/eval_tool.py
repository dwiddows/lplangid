import json
from pathlib import Path
import os
import time

import pandas as pd
import langid

from lplangid import language_classifier as lc
from experiments import fasttext_client, huggingface_client
from experiments.classification_report import nullsafe_classification_report

wiki_root = Path.home() / "Data" / "WikipediaLindemann"
bibles_root = Path.home() / "Data" / "bibles" / "BibleTexts/"

def langid_classify(text: str):
    return langid.classify(text)[0]


def load_twituser_test_data():
    with open("twituser_data/twituser") as twituser_data:
        records = [json.loads(line) for line in twituser_data]
    texts = [record["text"] for record in records]
    labels = [record["lang"] for record in records]
    return texts, labels


def sample_from_big_string(big_str: str, text_len: int, num_samples: int):
    texts = []
    for i in range(num_samples):
        region_start = (i * len(big_str)) // num_samples
        start = big_str.find(" ", region_start)
        if start == -1:
            continue
        end = big_str.find(" ", start + text_len)
        if end == -1:
            end = 0
        texts.append(big_str[start:end])
    return texts


def sample_texts_from_dir(text_dir, min_length, samples_per_file):
    texts, labels = [], []
    for lang in os.listdir(text_dir):
        with open(Path(text_dir) / lang, encoding='utf-8') as lang_fh:
            lang_contents = lang_fh.read()
        lang_samples = sample_from_big_string(lang_contents, min_length, samples_per_file)
        texts.extend(lang_samples)
        if lang.endswith(".txt"):
            lang = lang[:-4]
        labels.extend([lang] * len(lang_samples))
    return texts, labels


def run_tests():
    rrc_bibles= lc.RRCLanguageClassifier.many_language_bible_instance()
    rrc_smallwiki = lc.RRCLanguageClassifier(*lc.prepare_scoring_tables(data_dir=lc.FREQ_DATA_DIR + "_smallwiki"))
    ft_classifier = fasttext_client.FastTextLangID()
    hg_classifier = huggingface_client.HuggingfaceLangID()
    hg_classifier_xlm = huggingface_client.HuggingfaceLangID(huggingface_client.HUGGINGFACE_XLM_MODEL_PATH)

    fn_tags = [
        [lambda texts: [rrc_bibles.get_winner(text) for text in texts], "RRC bibles"],
        [lambda texts: [rrc_smallwiki.get_winner(text) for text in texts], "RRC smallwiki"],
        [lambda texts: [ft_classifier.predict_lang(text) for text in texts], "FastText"],
        [lambda texts: [langid_classify(text) for text in texts], "LangID"],
        [hg_classifier.predict_lang_batch, "DistilMBert Lang ID"],
        [hg_classifier_xlm.predict_lang_batch, "XLM Roberta Lang ID"],
    ]

    strlens = [16, 64, 256]
    test_texts = []
    for strlen in strlens:
        texts, y_labels = sample_texts_from_dir(Path(wiki_root) / "test", strlen, 20)
        test_texts.append([strlen, texts, y_labels])

    tagged_reports = []
    for fn, tag in fn_tags:
        for strlen, texts, y_labels in test_texts:
            y_pred = fn(texts)
            df_report = nullsafe_classification_report(y_labels, y_pred)
            df_report = df_report.loc[["macro avg", "weighted avg"]][["precision", "recall", "f1-score"]]

            # Create tuples and MultiIndex that include both the strlen/tag and the 'macro avg'/'weighted avg'
            index_tuples = [(tag, strlen, 'macro avg'), (tag, strlen, 'weighted avg')]
            multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['StrLen', 'Classifier', 'Metric'])
            df_report = pd.DataFrame(df_report.values, index=multi_index, columns=df_report.columns)
            tagged_reports.append(df_report)

    final_df = pd.concat(tagged_reports)
    print(final_df)
    final_df.to_csv("results/wiki_results_df.csv")


def main():
    run_tests()


if __name__ == "__main__":
    main()
