"""Report finding sensitivity and normal-sentence rate for a predictions CSV.

    python src/evaluate_findings.py data/processed/predictions_epoch2.csv

The CSV needs FINDINGS, IMPRESSION (reference) and PREDICTION columns.
"""
import sys

import pandas as pd
from rouge_score import rouge_scorer

sys.path.insert(0, "src")
from finding_metrics import exact_normal_rate, sensitivity_table, NORMAL_SENTENCE


def main(path):
    df = pd.read_csv(path).fillna("")
    refs, preds = df["IMPRESSION"].tolist(), df["PREDICTION"].tolist()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def rouge(candidates):
        scores = [scorer.score(r, c) for r, c in zip(refs, candidates)]
        return {k: sum(s[k].fmeasure for s in scores) / len(scores) for k in ("rouge1", "rouge2", "rougeL")}

    constant = [NORMAL_SENTENCE.capitalize() + "."] * len(refs)
    print(f"Reports: {len(df)}")
    train = set(pd.read_csv("data/processed/train_data.csv")["FINDINGS"].str.strip())
    unseen = ~df["FINDINGS"].str.strip().isin(train)
    print(f"Test reports whose FINDINGS text also appears in training: {(~unseen).sum()}")
    print(f"ROUGE-1/2/L  model:    {rouge(preds)}")
    clean_refs, clean_preds = [r for r, u in zip(refs, unseen) if u], [p for p, u in zip(preds, unseen) if u]
    scores = [scorer.score(r, c) for r, c in zip(clean_refs, clean_preds)]
    print(f"ROUGE-1/2/L  model, {len(clean_refs)} unseen-findings reports only: "
          f"{ {k: round(sum(s[k].fmeasure for s in scores) / len(scores), 3) for k in ('rouge1', 'rouge2', 'rougeL')} }")
    print(f"ROUGE-1/2/L  constant 'normal' baseline: {rouge(constant)}")
    print(f"Exact normal sentence  model: {exact_normal_rate(preds):.1%}  reference: {exact_normal_rate(refs):.1%}")
    print("\nFinding sensitivity (reference names the finding, does the model?)")
    for name, row in sensitivity_table(refs, preds).items():
        print(f"  {name:<14} {row['found']:>3}/{row['n']:<3} = {row['sensitivity']:.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
