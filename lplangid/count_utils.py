import csv
import logging
import math
from typing import TextIO, Dict, List


def write_freq_file(out_filename: str, freq_dict: Dict[str, int], max_records=-1):
    """Writes the given frequency table to out_filename, in order of descending value.

    Stops at max_records if this optional argument is positive"""
    freq_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    rank = 1
    with open(out_filename, 'w') as out_fh:
        logging.debug(f'Writing frequencies to {out_filename}')
        writer = csv.writer(out_fh)
        for freq_item in freq_items:
            writer.writerow(freq_item)
            rank += 1
            if rank > max_records > 0:
                break
        logging.debug(f'Wrote {len(freq_items)} entries.')


def write_rank_file(out_filename: str, ranked_items: List[str], max_records=-1):
    """Writes the elements of the given list to out_filename, in order.

    Stops at max_records if this optional argument is positive,"""
    if max_records > 0:
        ranked_items = ranked_items[:max_records]
    with open(out_filename, 'w') as out_fh:
        out_fh.write("\n".join(ranked_items))
        logging.debug(f'Wrote {len(ranked_items)} entries to {out_filename}.')


def read_freq_file(input_fh: TextIO, max_records=-1) -> Dict[str, int]:
    """Reads lines of the form "word, count" and returns a dictionary of this word -> count data

    If there are duplicate keys, the corresponding values are added.
    """
    output_dict = {}
    reader = csv.reader(input_fh)
    lines_read = 1
    for row in reader:
        output_dict[row[0]] = output_dict.get(row[0], 0) + int(row[1])
        lines_read += 1
        if lines_read > max_records > 0:
            break
    return output_dict


def read_rank_file(input_fh: TextIO, max_records=-1) -> Dict[str, int]:
    """Reads lines and returns a dictionary where each key is a line and each value is the rank / line number.

    Repeated lines are ignored.
    """
    output_ranks = {}
    line_number = 1
    for line in input_fh:
        text = line.strip()
        if text not in output_ranks:
            output_ranks[text] = line_number
        line_number += 1
        if line_number > max_records > 0:
            break
    return output_ranks


def freq_dict_to_lowercase(input_dict: Dict[str, int]) -> Dict[str, int]:
    """Takes a dictionary of string to value, and returns one where the keys are all mapped to
    lowercase and the values are the sums of all the values mapped in this way."""
    output_dict = {}
    for item in input_dict.items():
        lc = item[0].lower()
        output_dict[lc] = output_dict.get(lc, 0) + item[1]
    return output_dict


def freq_table_to_ranks(input_dict: Dict[str, int]):
    return {item[0]: rank + 1 for rank, item in enumerate(sorted(input_dict.items(), key=lambda x: x[1], reverse=True))}


def normalize_score_dict(input_dict: Dict[str, float]) -> Dict[str, float]:
    """Takes a dictionary of scores and applies L1-normalization (dividing each value by the sum).

    This is the simplest way of turning a collection of scores into a probability distribution.
    """
    total_weight = sum(input_dict.values())
    output_dict = {key: value / total_weight if total_weight > 0 else 1 / len(input_dict)
                   for key, value in input_dict.items()}
    return output_dict


def softmax_score_dict(input_dict: Dict[str, float]) -> Dict[str, float]:
    """Takes a dictionary of scores and applies the softmax function.

    This produces a probability distribution where differences in the inputs are made more pronounced.
    """
    total_weight = sum([math.exp(x) for x in input_dict.values()])
    output_dict = {key: math.exp(value) / total_weight for key, value in input_dict.items()}
    return output_dict


def softmax_to_l1(input_dict: Dict[str, float]) -> Dict[str, float]:
    """Takes a dictionary of scores and renormalizes to the probability distribution of which the input would
    be the softmax. (Doesn't check that the input is actually probability distribution, i.e., values add to one.

    If the softmax was created from a probability distribution, this will recover that distribution.
    """
    log_dict = {key: math.log(value) for key, value in input_dict.items()}
    sum_logs = sum(log_dict.values())
    remainder_shared = (1 - sum_logs) / len(input_dict)
    return {key: value + remainder_shared for key, value in log_dict.items()}


def precision_recall_f(total_positives, attempted, correct, beta=1):
    """Returns the precision, recall, and f1-score given with this number of
    total positives, attempted, and correct items."""
    precision = correct / attempted
    recall = correct / total_positives
    f_measure = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    return precision, recall, f_measure
