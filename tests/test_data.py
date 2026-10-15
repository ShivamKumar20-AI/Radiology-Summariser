from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"


def load(name):
    return pd.read_csv(DATA / name)


def test_columns_and_split_sizes():
    train, test = load("train_data.csv"), load("test_data.csv")
    assert list(train.columns) == ["FINDINGS", "IMPRESSION"]
    assert list(test.columns) == ["FINDINGS", "IMPRESSION"]
    assert len(train) == 2169
    assert len(test) == 543


@pytest.mark.xfail(
    strict=True,
    reason="Known leakage: 32 test reports (29 distinct texts) also appear in training, which inflates test ROUGE. "
    "evaluate_findings.py reports a deduplicated subset. Remove this marker once split_data.py groups duplicates.",
)
def test_no_report_appears_in_both_splits():
    train, test = load("train_data.csv"), load("test_data.csv")
    overlap = set(train["FINDINGS"].str.strip()) & set(test["FINDINGS"].str.strip())
    assert not overlap, f"{len(overlap)} identical FINDINGS texts appear in train and test"


def test_no_empty_text():
    for name in ("train_data.csv", "test_data.csv"):
        df = load(name)
        assert df["FINDINGS"].str.strip().str.len().min() > 0
        assert df["IMPRESSION"].str.strip().str.len().min() > 0
