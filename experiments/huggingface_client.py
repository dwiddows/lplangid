import os
from pathlib import Path
import re
from typing import List
import warnings

import numpy as np
from scipy.special import softmax
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate.*")

HUGGINGFACE_DEFAULT_MODEL_ROOT = Path(os.path.dirname(__file__)) / "distilmbert_lc_model_800"

HUGGINGFACE_XLM_MODEL_PATH = "papluca/xlm-roberta-base-language-detection"


def get_latest_model_from_dir(model_path):
    # If it's not a directory, hopefully it's a huggingface tag ...
    if not os.path.exists(model_path):
        return model_path

    # If it's a directory with a config.json, it's probably what we're looking for.
    if os.path.isfile(Path(model_path) / "config.json"):
        return model_path

    # Otherwise look for the last checkpoint in the directory.
    pattern = re.compile(r"checkpoint-\d+")
    dir_items = os.listdir(model_path)
    checkpoints = sorted(filter(pattern.match, dir_items), key=lambda x: int(x.split('-')[-1]))
    if not checkpoints:
        raise ValueError("No checkpoint found in the directory.")
    latest_checkpoint = checkpoints[-1]
    return os.path.join(model_path, latest_checkpoint)


class HuggingfaceLangID:
    def __init__(self, model_root=HUGGINGFACE_DEFAULT_MODEL_ROOT):
        model_path = get_latest_model_from_dir(model_root)
        self.lc_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.evaluator = Trainer(model=self.lc_model)

    def predict_lang_batch(self, texts: List[str], batch_size=100, verbose=False):
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_predicted_labels = []

        for batch_texts in batches:
            tokenized_texts = self.tokenizer(batch_texts, padding=True, return_tensors="pt", truncation=True, max_length=max([len(t) for t in texts]))
            inputs = {k: v.to(self.evaluator.args.device) for k, v in tokenized_texts.items()}
            with torch.no_grad():
                outputs = self.lc_model(**inputs)
            all_logits = outputs.logits.cpu().numpy()

            predicted_labels = []
            for logits in all_logits:
                probs = softmax(logits, axis=-1)
                # Print sorted languages by probability
                if verbose:
                    lang_scores = {self.lc_model.config.id2label[i]: prob for i, prob in enumerate(probs)}
                    for k, v in sorted(lang_scores.items(), key=lambda x: x[1]):
                        print(f"{k}\t{v:0.4f}")
                predicted_index = np.argmax(probs, axis=-1)
                predicted_label = self.lc_model.config.id2label[predicted_index]
                predicted_labels.append(predicted_label)
            all_predicted_labels.extend(predicted_labels)

        return all_predicted_labels

    def predict_lang(self, text: str, verbose=False):
        return self.predict_lang_batch([text])[0]


if __name__ == "__main__":
    hg_classifier = HuggingfaceLangID()
    lang = hg_classifier.predict_lang_batch(["Hello in English", "Bonjour en Francais"])
    print(f"Default prediction: {lang}")
    hg_xlm_classifier = HuggingfaceLangID(HUGGINGFACE_XLM_MODEL_PATH)
    lang = hg_xlm_classifier.predict_lang_batch(["Hello in English", "Bonjour en Francais"])
    print(f"XLM prediction: {lang}")

