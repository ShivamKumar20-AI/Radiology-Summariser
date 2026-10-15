"""Clinical-finding metrics for generated impressions.

ROUGE rewards word overlap, so a model that always says "no acute abnormality"
scores well on a dataset where most reports are normal. These metrics ask a
more useful question: when the reference impression names a finding, does the
generated impression name it too?

Negation is handled with a simple rule: inside each sentence, everything from
"no", "without", "negative for" or "free of" onwards is ignored, so
"Cardiomegaly with no pneumothorax" counts as cardiomegaly only. This is a
heuristic, not a clinical NLP parser.
"""
import re

FINDINGS = {
    "consolidation": r"consolidat|pneumonia|air-?space|infiltrat|opacit",
    "effusion": r"effusion",
    "pneumothorax": r"pneumothora",
    "cardiomegaly": r"cardiomegaly|enlarged heart|heart (is )?enlarged",
    "edema": r"edema|congestion",
    "nodule": r"nodul|mass\b",
}

NORMAL_SENTENCE = "no acute cardiopulmonary abnormality"
_NEGATION = re.compile(r"\b(no|without|negative for|free of)\b.*", re.IGNORECASE)


def positive_text(impression):
    """Return the parts of an impression that are not negated."""
    sentences = re.split(r"[.;]\s*", str(impression))
    return " ".join(_NEGATION.sub("", s) for s in sentences).lower()


def mentions(impression, finding):
    return re.search(FINDINGS[finding], positive_text(impression)) is not None


def is_exact_normal(impression):
    return re.sub(r"[^a-z ]", "", str(impression).lower()).strip() == NORMAL_SENTENCE


def sensitivity_table(references, predictions):
    """For each finding: reference count, how many predictions also mention it."""
    rows = {}
    for name in FINDINGS:
        positives = [i for i, ref in enumerate(references) if mentions(ref, name)]
        found = sum(mentions(predictions[i], name) for i in positives)
        rows[name] = {
            "n": len(positives),
            "found": found,
            "sensitivity": found / len(positives) if positives else float("nan"),
        }
    return rows


def exact_normal_rate(impressions):
    impressions = list(impressions)
    return sum(is_exact_normal(i) for i in impressions) / len(impressions)
