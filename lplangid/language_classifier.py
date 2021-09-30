import logging
import math
import os
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple

from lplangid import count_utils as cu
from lplangid.const_data import COMPUTERESE_STARTS
from lplangid.tokenizer import tokenize_fast

# The default Wikipedia + overrides datafiles are shipped in this base directory.
FREQ_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freq_data")

# Weights and biases that can easily be tweaked by hand, or could be learned / optimized.
TERM_PRESENCE_WEIGHT = 0.05
BASELINE_TERM_SCORE = TERM_PRESENCE_WEIGHT / 2
TOP_RANK_DAMPING = 10
LETTERS = set(string.ascii_lowercase)

# If the character score for a language is below this proportion of the winning char score, it's rejected.
# This is tuned towards rejecting Chinese as an option if there are too many Japanese-only characters.
# (In other words, beware, this *is* a dangerously over-tuned hand-written rule that might not generalize.)
CHAR_MIN_TO_PLAY = 0.6

# Maximum number of words read in for each language.
MAX_WORDS_PER_LANG = 10000

CLASSIFY_WORDS_LOWER_CASE = True
CLASSIFY_CHARS_LOWER_CASE = True


class RRCLanguageClassifier:
    """RRCLanguageClassifier is a class that provides language detection scores and predictions.

    It holds the term rank and char rank tables, and runs the pure functions (below) using this state.
    """
    def __init__(self, term_ranks: Dict[str, Dict[str, int]], char_weights: Dict[str, List[Tuple[str, float]]]):
        """
         Construct a new 'RRCLanguageClassifier' object.

         :param term_ranks: dictionary mapping language code -> word/term -> rank.
         :param char_weights: dictionary mapping character -> (language, relative frequency).

         The char_weights table is optimized to score every (character, language) score, whereas the term_ranks
         table is optimized to compute (language, term) scores for languages that pass the character cutoff.
         """
        self.term_ranks: Dict[str, Dict[str, int]] = term_ranks
        self.char_weights: Dict[str, List[Tuple[str, float]]] = char_weights

    @staticmethod
    def default_instance():
        """Gets a default instance populated using the prepare_scoring_tables function."""
        all_term_ranks, all_char_weights = prepare_scoring_tables()
        logging.info(f"Loaded classifier with term ranks and character frequencies for these languages: "
                     f"{', '.join(sorted(all_term_ranks.keys()))}")
        return RRCLanguageClassifier(all_term_ranks, all_char_weights)

    def get_winner(self, text: str) -> str:
        """Returns the language with the single best score. (Ties are very rare.)"""
        return get_winner(self.term_ranks, self.char_weights, text)

    def get_winner_score(self, text: str) -> Tuple[str, float]:
        """Returns the language with the single best score, and its score. (Ties are very rare.)"""
        return get_winner_score(self.term_ranks, self.char_weights, text)

    def get_language_scores(self, text: str) -> List[Tuple[str, float]]:
        """Returns a list of (language code, score) pairs, sorted from highest to lowest score."""
        return score_text(self.term_ranks, self.char_weights, text)


def prepare_scoring_tables() -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[Tuple[str, float]]]]:
    """Reads in term and character ranking data from the files in FREQ_DATA_DIR"""
    all_term_ranks = {}
    all_char_freqs = {}
    for lang_code in set([x[:2] for x in os.listdir(FREQ_DATA_DIR) if x.endswith('.csv') and not x.startswith('.')]):
        tf_file = os.path.join(FREQ_DATA_DIR, f"{lang_code}_term_rank.csv")
        with open(tf_file) as term_freq_file:
            term_freqs = cu.read_rank_file(term_freq_file, MAX_WORDS_PER_LANG)
            all_term_ranks[lang_code] = term_freqs
        with open(os.path.join(FREQ_DATA_DIR, f'{lang_code}_char_freq.csv')) as char_freq_file:
            all_char_freqs[lang_code] = cu.normalize_score_dict(cu.read_freq_file(char_freq_file))
    all_char_weights = invert_char_tables(all_char_freqs)

    logging.debug(f"Prepared term and character ranking tables for languages: {sorted(all_term_ranks.keys())}")
    return all_term_ranks, all_char_weights


def invert_char_tables(lang_to_char_weight: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
    """Inverts a table of lang -> char -> weight to a table of char -> list of (lang, weight) pairs."""
    all_char_weights_dict = {}
    for lang, char_weights in lang_to_char_weight.items():
        for char, weight in char_weights.items():
            if char not in all_char_weights_dict:
                all_char_weights_dict[char] = {}
            all_char_weights_dict[char][lang] = weight
    all_char_weights = {char: sorted(cu.normalize_score_dict(weights).items(), key=lambda x: x[1], reverse=True)
                        for char, weights in all_char_weights_dict.items()}
    return all_char_weights


def score_terms(all_term_ranks: Dict[str, Dict[str, int]], text: str, languages: Tuple[str] = ()) -> Dict[str, float]:
    """Gets a score for each language for the given text based on how common the terms are."""
    tokens = Counter(tokenize_fast(text))
    tokens = {token: count for token, count in tokens.items() if len(token) > 1 or token not in LETTERS}
    scores: Dict[str, float] = {}
    if not languages:
        languages = all_term_ranks.items()

    for lang in languages:
        lang_score = BASELINE_TERM_SCORE
        if lang in all_term_ranks:
            ranks = all_term_ranks[lang]
            for token, count in tokens.items():
                if token in ranks:
                    lang_score += (TERM_PRESENCE_WEIGHT + 1 / math.sqrt(TOP_RANK_DAMPING + ranks[token])) * count
        scores[lang] = lang_score
    return scores


def score_chars(all_char_weights: Dict[str, List[Tuple[str, float]]], text: str) -> Dict[str, float]:
    """Gets a score for each language for the given text based on how common the characters are."""
    chars = Counter(text)
    chars = {char: count for char, count in chars.items() if char.isalpha() and char in all_char_weights}
    scores = {}
    for char, count in chars.items():
        for lang, weight in all_char_weights[char]:
            if lang not in scores:
                scores[lang] = 0
            scores[lang] += weight * count
    return scores


def score_text(all_term_ranks: Dict[str, Dict[str, int]],
               all_char_weights: Dict[str, List[Tuple[str, float]]],
               text: str) -> List[Tuple[str, float]]:
    """Gets the term and character scores and combines them into a single score for each language."""
    if any([text.startswith(x) for x in COMPUTERESE_STARTS]):
        return []

    char_scores = score_chars(all_char_weights, text.lower() if CLASSIFY_CHARS_LOWER_CASE else text)
    if not any(char_scores):
        return []
    char_max = max(char_scores.values())
    char_scores = {k: v for k, v in char_scores.items() if v > char_max * CHAR_MIN_TO_PLAY}
    char_scores = cu.normalize_score_dict(char_scores) if sum(char_scores.values()) > 0 else char_scores
    # Early-out if there is only one contender left (partly to avoid penalizing no term matches without tokenization).
    if len(char_scores) == 1:
        return list(char_scores.items())

    term_scores = score_terms(
        all_term_ranks, text.lower() if CLASSIFY_WORDS_LOWER_CASE else text, languages=tuple(char_scores))
    # If we got this far but have no explicit term matches, then it's usually a spurious classification.
    if not max(term_scores.values()) >= BASELINE_TERM_SCORE + TERM_PRESENCE_WEIGHT:
        return []
    term_scores = cu.normalize_score_dict(term_scores)

    combined_scores = {lang: term_scores.get(lang, BASELINE_TERM_SCORE) * char_scores[lang] for lang in char_scores}
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)


def get_char_winner(all_char_weights: Dict[str, List[Tuple[str, float]]], text: str) -> str:
    """Calls score_chars and returns the language with the highest char score.
    If no scores are greater than zero, returns None."""
    scores = score_chars(all_char_weights, text)
    winner, score = max(scores.items(), key=lambda x: x[1])
    return winner if score > 0 else None


def get_winner_score(term_dict: Dict[str, Dict[str, int]], char_dict: Dict[str, List[Tuple[str, float]]], text: str
                     ) -> Tuple[Optional[str], float]:
    """Calls score_text and returns the winning language and its score.

    No thresholds or tie-breaking is used. If there is a tie, the winner is unpredictable.

    If all scores are zero, the winner is None."""
    combined_scores = score_text(term_dict, char_dict, text)
    if len(combined_scores) == 0:
        return None, 0
    winner, score = max(combined_scores, key=lambda x: x[1])
    return winner if score > 0 else None, score


def get_winner_margin(term_dict, char_dict, text) -> Tuple[Optional[str], float]:
    """Calls score_text and returns the winning language and how much it won by (compared with second highest score).

    If all scores are zero, the winner is None."""
    scores = score_text(term_dict, char_dict, text)
    sorted_scores: List[Tuple[str, float]] = sorted(scores, key=lambda x: x[1], reverse=True)
    if len(sorted_scores) == 0:
        return None, 0
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    winner = sorted_scores[0][0] if sorted_scores[0][1] > 0 else None
    score = sorted_scores[0][1] - sorted_scores[1][1]
    return winner, score


def get_winner(all_term_ranks: Dict[str, Dict[str, int]],
               all_char_weights: Dict[str, List[Tuple[str, float]]],
               text: str) -> Optional[str]:
    """Calls score_text and returns the language with the highest score.
    If no scores are greater than zero, returns None."""
    combined_scores = score_text(all_term_ranks, all_char_weights, text)
    if len(combined_scores) == 0:
        return None
    winner, score = max(combined_scores, key=lambda x: x[1])
    return winner if score > 0 else None


def main():
    """Demonstration main function that you can run just to see what it does with a few phrases."""
    logging.basicConfig(level=logging.INFO)
    rrc_classifier = RRCLanguageClassifier.default_instance()
    texts = ["This is English", "Esto es español"]

    for text in texts:
        winner = rrc_classifier.get_winner(text)
        print(f'{winner}: {text}')


if __name__ == '__main__':
    main()
