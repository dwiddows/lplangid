from pathlib import Path

import fasttext


class FastTextLangID:
    def __init__(self):
        pretrained_lang_model = Path.home() / "Data" / "fasttext" / "lid.176.bin"
        self.model = fasttext.load_model(str(pretrained_lang_model))

    def predict_lang(self, text):
        text = text.replace("\n", " ").strip()
        prediction = self.model.predict(text, k=1)[0]
        return prediction[0].split("__")[-1]


if __name__ == "__main__":
    LANGUAGE = FastTextLangID()
    lang = LANGUAGE.predict_lang("Hello my name is dominic how are you")
    print(lang)
