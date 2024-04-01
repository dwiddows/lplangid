import json

import langid

from lplangid import language_classifier as lc
from experiments import fasttext_client, huggingface_client
from experiments.classification_report import nullsafe_classification_report


def langid_classify(text: str):
    return langid.classify(text)[0]


def run_twituser_tests():
    rrc_classifier = lc.RRCLanguageClassifier.default_instance()
    rrc_bibles= lc.RRCLanguageClassifier.many_language_bible_instance()
    rrc_smallwiki = lc.RRCLanguageClassifier(*lc.prepare_scoring_tables(data_dir=lc.FREQ_DATA_DIR + "_smallwiki"))
    ft_classifier = fasttext_client.FastTextLangID()
    hg_classifier = huggingface_client.HuggingfaceLangID()

    fn_labels = [
        [lambda texts: [rrc_classifier.get_winner(text) for text in texts], "RRC default"],
        [lambda texts: [rrc_bibles.get_winner(text) for text in texts], "RRC bibles"],
        [lambda texts: [rrc_smallwiki.get_winner(text) for text in texts], "RRC smallwiki"],
        [lambda texts: [ft_classifier.predict_lang(text) for text in texts], "FastText"],
        [lambda texts: [langid_classify(text) for text in texts], "LangID"],
        [hg_classifier.predict_lang_batch, "HuggingFace"],
    ]

    # fn_labels = [[hg_classifier.predict_lang, "HuggingFace"]]

    for fn, label in fn_labels:
        print(f"Classifying with {label}")
        y_labels = []
        with open("twituser_data/twituser") as twituser_data:
            input_texts = []
            for line in twituser_data:
                record = json.loads(line)
                # if record["lang"] not in rrc_classifier.term_ranks:
                #     continue
                input_texts.append(record["text"])
                y_labels.append(record["lang"])

        y_pred = fn(input_texts)

        print(nullsafe_classification_report(y_labels, y_pred))


def main():
    run_twituser_tests()


if __name__ == "__main__":
    main()
