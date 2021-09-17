# lplangid: Reciprocal Rank Classifier for Language Identification

This package is a python implementation of the classifier described in the paper "Language Identification with a Reciprocal Rank Classifier" (TODO: Publish and link to preprint before release.) 

## Installation and Usage in Classification

You can install the package by running `$ pip install lplangid`, or by cloning this repository and running `pip install -e .` in this directory. 

Basic usage example for language classification:

```
>>> from lplangid.language_classifier import RRCLanguageClassifier
>>> my_classifier = RRCLanguageClassifier.get_default()
>>> my_classifier = RRCLanguageClassifier.default_instance()
>>> my_classifier.get_winner("C'est use teste")

'fr'
``` 

## Data Preparation and Distribution

Throughout this package, languages are identified and referred to using 
[2-letter ISO 339-1 codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes). For example, `en` for English,
`es` (from Español) for Spanish, `zh` (from 中文, Zhōngwén) for Chinese. These are used throughout for directory names,
keys in dictionary tables, and reporting classifier results. 

The classifier uses the datafiles checked in to the `./freq_data` directory here, which is just a few megabytes.
It would be relatively easy to decouple the way these files are distributed. The benefit of combining them is it's
very easy for clients to use. 

The frequency tables in `./lplangid/freq_data` are from Wikipedia data (single shards), tokenized on whitespace.
In addition, a few conversational words from `./training/data_overrides.py` have been added at the top of the 
term rank files.
The `xx_char_freq.csv` files contain characters and sample frequencies. The `xx_term_rank.csv` files contain
only the terms / words. Only the ranks (line numbers) of the words in these files matters.
Unlike most classifier models, you can edit these files directly. For example, the word "bye" and other conversational
terms that are rare in Wikipedia have already been added to the top of the `en_term_rank.csv` file.

See `./training/README.md` for data preparation instructions and tools for adding new languages to the classifier.
