import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finding_metrics import exact_normal_rate, is_exact_normal, mentions, sensitivity_table


def test_negated_finding_is_not_a_mention():
    assert not mentions("No pneumothorax or pleural effusion.", "effusion")
    assert not mentions("Cardiomegaly with no pneumothorax.", "pneumothorax")


def test_positive_finding_is_a_mention():
    assert mentions("Right lower lobe consolidation.", "consolidation")
    assert mentions("Large pleural effusion. No pneumothorax.", "effusion")
    assert mentions("Pleural effusion. No pneumothorax.", "effusion")


def test_exact_normal_sentence_ignores_case_and_punctuation():
    assert is_exact_normal("No acute cardiopulmonary abnormality.")
    assert is_exact_normal("no acute cardiopulmonary abnormality")
    assert not is_exact_normal("No acute cardiopulmonary abnormality. Mild cardiomegaly.")


def test_sensitivity_counts_only_reports_with_the_finding():
    refs = ["Right lower lobe consolidation.", "No acute abnormality.", "Pneumonia."]
    preds = ["Consolidation in the right lung.", "Normal.", "No acute cardiopulmonary abnormality."]
    row = sensitivity_table(refs, preds)["consolidation"]
    assert (row["n"], row["found"]) == (2, 1)
    assert row["sensitivity"] == 0.5


def test_exact_normal_rate():
    assert exact_normal_rate(["No acute cardiopulmonary abnormality.", "Cardiomegaly."]) == 0.5
