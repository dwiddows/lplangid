#!/usr/bin/env bash

# This script goes through the steps of adding a new language to the RRC classifier based on wiki data.
# See the README.md file for a more thorough explanation of these steps.
# Wikipedia archives are often large (several gigabytes), so the basic download step can take several minutes.

if [[ $(basename `pwd`) != "training" ]]
  then
    echo "$0 must be run from the 'training' directory that contains it."
    exit 1
fi

# Check there is exactly one argument and its a 2-character string.
if [[ $# -ne 1 || ${#1} -ne 2 ]];
then
	echo "Usage: $0 LANGUAGE_CODE"
	echo "The LANGUAGE_CODE argument must be a single 2-letter ISO 639-1 code for the language to be added."
	exit 1
fi

language=$1

target_wiki_dir=~/Data/Wikipedia/$language
freq_data_dir=../lplangid/freq_data

if [[ -f ${freq_data_dir}/${language}_char_freq.csv || -f ${freq_data_dir}/${language}_term_rank.csv ]]
  then
    echo "There are already files ${freq_data_dir}/${language}_char_freq.csv or ${freq_data_dir}/${language}_term_rank.csv"
    echo "These will be overwritten by this process, so please move / remove them first."
    exit 1
fi

if [[ (-d $target_wiki_dir) ]]
  then
    echo "Directory $target_wiki_dir already exists. Please check and move / remove it if you want to try again with this script."
    exit 1
fi

# Predict the date string for the last expected Wikipedia data dump.
# This follows the pattern that data dumps happen on 1st and 20th of each month.
# If this assumption becomes incorrect, you'll need to navigate to the most recent version explicitly.
year_month=$(date +%Y%m)
day=$(date +%d)
last_dump_day=$(printf "%02d" $(( day <= 20 ? 01 : 20 )))
dump_ymd=$year_month$last_dump_day
last_dump_url=https://dumps.wikimedia.org/${language}wiki/${dump_ymd}/${language}wiki-${dump_ymd}-pages-articles-multistream.xml.bz2

if ! wget -q --method=HEAD $last_dump_url; then
  echo "Found no archive at $last_dump_url. Exiting."
  exit 1
fi

mkdir -p $target_wiki_dir
pushd $target_wiki_dir
  echo "Fetching wiki data file from $last_dump_url to directory `pwd`"
  time wget $last_dump_url
  echo "Unzipping wiki archive using bunzip2 ..."
  time bunzip2 ${language}wiki-${dump_ymd}-pages-articles-multistream.xml.bz2
  echo "Extracting text documents from XML using wikiextractor ..."
  time python3 -m wikiextractor.WikiExtractor --quiet ${language}wiki-${dump_ymd}-pages-articles-multistream.xml
popd

echo "Processing text files to create character frequency and term rank data ..."
time python process_wiki_archive.py --languages $language

echo "Finished. Please check that files ${freq_data_dir}/${language}_char_freq.csv and ${freq_data_dir}/${language}_term_rank.csv look to be present and correct."
