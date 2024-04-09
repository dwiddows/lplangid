from typing import List

import pandas as pd
from sklearn.metrics import classification_report


def nullsafe_classification_report(y_label: List[str], y_pred: List[str]):
    """Wrapper round sklearn classification report that is robust to None / empty predictions.

    Implemented by creating a "dummy" class label, and considering each None a "dummy".
    Then the individual label precisions and recalls work correctly, as do the accuracy and weighted average.
    The macro average gets artificially penalized because the denominator is increased by one, so this is
    corrected before returning.
    """
    dummy_val = "DUMMY_VAL"
    label_set = set(y_label)
    num_pred_labels = len({y for y in y_pred if y in label_set})
    y_pred = [y if y in label_set else dummy_val for y in y_pred]
    report = classification_report(y_label, y_pred, output_dict=True, zero_division=0.0)

    if dummy_val in report:
        del report[dummy_val]
        report["macro avg"]["precision"] = report["macro avg"]["precision"] * (num_pred_labels + 1) / num_pred_labels
        report["macro avg"]["recall"] = report["macro avg"]["recall"] * (num_pred_labels + 1) / num_pred_labels
        report["macro avg"]["f1-score"] = report["macro avg"]["f1-score"] * (num_pred_labels + 1) / num_pred_labels
    df = pd.DataFrame(report).transpose()
    return df


def test_nullsafe_classification_report():
    y_label = ["1", "1", "2", "2"]
    y_pred = ["1", None, "2", "2"]
    df = nullsafe_classification_report(y_label, y_pred)
    print()
    print(df.to_latex())
