# This experiment is on finding which pairs of languages are similar to one another based on the classifier models.
# Requires scipy (e.g. run "pip install scipy")

import logging
from math import log
from typing import Any, Dict, List

from scipy.stats import pearsonr

from lplangid import count_utils as cu
from lplangid import language_classifier as lc


def freq_table_to_ranks_list(input_dict: Dict[str, int]) -> List[str]:
    return [item[0] for _, item in enumerate(sorted(input_dict.items(), key=lambda x: x[1], reverse=True))]


def cos_sim_dicts(dict1, dict2: Dict[Any, float]) -> float:
    dict1, dict2 = cu.normalize_score_dict(dict1, exponent=2), cu.normalize_score_dict(dict2, exponent=2)
    return sum([dict1[x] * dict2[x] for x in set(dict1).intersection(set(dict2))])


def log_weighted_pearson(ranks_list1, ranks_list2):
    ranks1 = {entry: rank + 1 for rank, entry in enumerate(ranks_list1)}
    ranks2 = {entry: rank + 1 for rank, entry in enumerate(ranks_list2)}
    points = [
        (1 / log(ranks1.get(entry, len(ranks_list1)) + 1), 1 / log(ranks2.get(entry, len(ranks_list2)) + 1))
        for entry in set(ranks_list1).union(set(ranks_list2))
    ]
    return pearsonr([point[0] for point in points], [point[1] for point in points])[0]


def ranks_sim_score(list1, list2):
    if len(list1) != len(list2):
        logging.debug(
            f"Lists are of unequal length {len(list1)}, {len(list2)}. "
            f"Truncating to minimum length {min(len(list1), len(list2))}"
        )
        list1 = list1[: min(len(list1), len(list2))]
        list2 = list2[: min(len(list1), len(list2))]

    ranks1 = {entry: rank + 1 for rank, entry in enumerate(list1)}
    ranks2 = {entry: rank + 1 for rank, entry in enumerate(list2)}
    intersection = set(ranks1).intersection(set(ranks2))

    drift_score = 0
    for entry in intersection:
        rank1 = ranks1.get(entry, len(list1))
        rank2 = ranks2.get(entry, len(list1))
        drift_score_delta = abs(1 / (rank2 + lc.TOP_RANK_DAMPING) - 1 / (rank1 + lc.TOP_RANK_DAMPING))
        drift_score += drift_score_delta

    for entry in set(list1).difference(list2):
        drift_score += 1 / ranks1[entry] - 1 / (len(list2) + lc.TOP_RANK_DAMPING)

    for entry in set(list2).difference(list1):
        drift_score += 1 / ranks2[entry] - 1 / (len(list1) + lc.TOP_RANK_DAMPING)

    return drift_score


def main_ranks_sim():
    all_term_ranks, all_char_weights = lc.prepare_scoring_tables()
    all_term_ranks = {lang: freq_table_to_ranks_list(vals)[:1000] for lang, vals in all_term_ranks.items()}
    compare_term_ranks = {lang: ranks for lang, ranks in all_term_ranks.items()}
    for lang1 in ["id", "es", "nl"]:
        scores = [
            (lang2, ranks_sim_score(compare_term_ranks[lang1], compare_term_ranks[lang2]))
            for lang2 in compare_term_ranks
        ]
        print(f"\nNearest to {lang1}:")
        for lang2, score in sorted(scores, key=lambda x: x[1])[:8]:
            print(f"\t{lang2}\t{score:0.3f}")


def main_ranks_cos():
    all_term_ranks, all_char_weights = lc.prepare_scoring_tables()
    my_term_ranks = {
        lang: {word: 1 / (rank + lc.TOP_RANK_DAMPING) for word, rank in ranks.items()}
        for lang, ranks in all_term_ranks.items()
    }
    for lang1 in ["id", "es", "nl"]:  # my_term_ranks:
        scores = [(lang2, cos_sim_dicts(my_term_ranks[lang1], my_term_ranks[lang2])) for lang2 in my_term_ranks]
        print(f"\nNearest to {lang1}:")
        for lang2, score in sorted(scores, key=lambda x: x[1], reverse=True)[:8]:
            print(f"\t{lang2}\t{score:0.3f}")


def _test():
    print(ranks_sim_score(["a", "b", "c"], ["a", "b", "c"]))
    print(ranks_sim_score(["a", "b", "c"], ["a", "c", "b"]))
    print(ranks_sim_score(["a", "b", "c"], ["a", "b", "d"]))
    print(ranks_sim_score(["a", "b", "c"], ["d", "e", "a"]))
    print(ranks_sim_score(["a", "b", "c"], ["d", "e", "f"]))


if __name__ == "__main__":
    main_ranks_cos()
    main_ranks_sim()
