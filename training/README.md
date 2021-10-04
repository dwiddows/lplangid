# Training / Configuring the RRC Language Classifier

This directory contains data gathering and preparation scripts. These are
useful for adding new language to the classifier, or enhancing the behavior for
languages supported already.

The main workflow involves gathering corpus data (e.g., Wikipedia is used here),
and counting tokens and characters.

This is not really "training" in the sense of machine learning of weights with
some objective function to optimize, but it is part of preparing the system to
perform classification, so the package name "training" was chosen to avoid
confusing with preparation in the sense of installation.

# Data Requirements

Each language to be classified needs a `xx_char_freq.csv` and a `xx_term_rank.csv` file in the `lplangid/freq_data`
directory. The data files that ship with this package are mainly built from Wikipedia files, with some conversational
words added at or near the top of the term rank files.

## Adding a New Language from Wikipedia Data

#### Running the Automatic Script

tl;dr: Try running the bash script `./add_wiki_language.sh $LANGUAGE_CODE` and see if it works. It will usually
take several minutes and less then half an hour to add support for a new language.

Requires bash, wget, bunzip2, python wikiextractor (see below), and the tools in this directory.

#### Steps in Detail

In more detail: to create the necessary term rank and character frequency files these for a given language, 
the following steps are used.

* Make a data directory to keep the text data, e.g., `~/Data/Wikipedia` (used here, where `~` refers to your
$HOME directory on Unix-like systems).
  * Each language should hvae its own 2-character ISO 

* Go to https://dumps.wikimedia.org/backup-index.html and download the most recent xxwiki dump for the xx language.
  * You do not need the whole Wiki, a single archive fragment is usually plenty.
  * For example, to add Polish, one would go to https://dumps.wikimedia.org/backup-index.html and click on the first 
`plwiki` link.
  * This typically goes to a page at https://dumps.wikimedia.org/{$LANGUAGE}wiki/{$RECENT_DATE},
  e.g., https://dumps.wikimedia.org/plwiki/20210820/.
  * From there, select and download the first archive of articles, e.g., plwiki-20210820-pages-articles-multistream.xml.bz2
  * This also goes to a parametrized URL, such as https://dumps.wikimedia.org/plwiki/20210820/plwiki-20210820-pages-articles-multistream.xml.bz2
  * Even for single compressed archive files like this, the download will still normally take several minutes.
  * Put the archive in its own language directory in `~/Data/Wikipedia/pl`

* Unzip the archive into XMl format.
  * Navigate to the new language directory, e.g., `cd ~/Data/Wikipedia`
  * Unzip using a command such as `bunzip2 plwiki-20210820-pages-articles-multistream.xml.bz2`

* Extract plain text from the XML file using using https://github.com/attardi/wikiextractor
  * `pip install wikiextractor` followed by `python -m wikiextractor.WikiExtractor <Wikipedia dump file>`
  
* After that, run the `process_wiki.py` script.
  * E.g., in this directory, run `python process_wiki.py --requested_languages=pl`
  * Running the counting process takes 5 to 10 minutes per Wikipedia dump, depending on size.
  * If no `--requested_languages` argument is provided, the script will try to recount for all available languages.
  * The result should be new or updated files in `./lplangid/freq_data`, in this case, `pl_char_freq.csv` and `pl_term_rank.csv`.

In a recent test using Malaysian, downloading and unzipping the wiki archive took ~2 minutes, running wikiextractor.py
took ~10 minutes, running process_wiki.py to count the terms and characters took ~3 minutes.
 