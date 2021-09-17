from lplangid import count_utils as cu
import pytest


def test_normalize_score_dict():
    assert cu.normalize_score_dict({"foo": 1, "bar": 2}) == {"foo": 1/3, "bar": 2/3}
    assert cu.normalize_score_dict({"foo": 0, "bar": 0}) == {"foo": 1/2, "bar": 1/2}

    # L1 normalization is idempotent.
    assert cu.normalize_score_dict({"foo": 1, "bar": 2}) == cu.normalize_score_dict(
        cu.normalize_score_dict({"foo": 1/3, "bar": 2/3}))


def test_softmax_score_dict():
    assert cu.softmax_score_dict({"foo": 1, "bar": 2}) == pytest.approx({'bar': 0.731058578, 'foo': 0.268941421})
    assert cu.softmax_score_dict({"foo": 2, "bar": 3}) == pytest.approx({'bar': 0.731058578, 'foo': 0.268941421})
    assert cu.softmax_score_dict({"foo": 10, "bar": 11}) == pytest.approx({'bar': 0.731058578, 'foo': 0.268941421})
    assert cu.softmax_score_dict({"foo": 0, "bar": 0}) == {"foo": 1/2, "bar": 1/2}

    # Note that softmax normalization is not idempotent.
    assert not cu.softmax_score_dict({"foo": 1, "bar": 2}) == cu.softmax_score_dict(
        cu.softmax_score_dict({"foo": 1, "bar": 2}))


def test_softmax_score_dict_inverse():
    assert (cu.softmax_to_l1(cu.softmax_score_dict({"foo": 1/3, "bar": 2/3}))
            == pytest.approx({"foo": 1/3, "bar": 2/3}))
    initial = {"foo": 1, "bar": 2}
    softmaxed = cu.softmax_score_dict(initial)
    assert cu.softmax_to_l1(cu.softmax_score_dict(softmaxed)) == pytest.approx(softmaxed)


def test_freq_dict_to_lowercase():
    freq_dict = {"OK": 3, "ok": 4, "Okay": 5}
    freq_dict = cu.freq_dict_to_lowercase(freq_dict)
    assert freq_dict == {"ok": 7, "okay": 5}


def test_freq_table_to_ranks():
    freq_dict = {"OK": 5, "ok": 4, "Okay": 3}
    assert cu.freq_table_to_ranks(freq_dict) == {"OK": 1, "ok": 2, "Okay": 3}


def test_precision_recall_f():
    assert cu.precision_recall_f(100, 80, 70) == pytest.approx((0.875, 0.7, 0.77778), abs=0.0001)
    assert cu.precision_recall_f(100, 80, 70, 0.5) == pytest.approx((0.875, 0.7, 0.83333), abs=0.0001)
    assert cu.precision_recall_f(800, 80, 70) == pytest.approx((0.875, 0.0875, 0.1590909), abs=0.0001)
    assert cu.precision_recall_f(100, 80, 70, 0.5) == pytest.approx((0.875, 0.7, 0.83333), abs=0.0001)
