"""Generate impressions for the test set with a chosen model and decoding settings.

    python src/predict_csv.py model/balanced data/processed/predictions_balanced.csv [num_beams] [length_penalty]
"""
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_dir, out_path = sys.argv[1], sys.argv[2]
num_beams = int(sys.argv[3]) if len(sys.argv) > 3 else 4
length_penalty = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

device = "mps" if os.environ.get("USE_MPS") == "1" and torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()

df = pd.read_csv("data/processed/test_data.csv").fillna("")
predictions = []
for i, findings in enumerate(df["FINDINGS"], 1):
    inputs = tokenizer("summarize radiology findings: " + findings, max_length=512,
                       truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=128, num_beams=num_beams, length_penalty=length_penalty)
    predictions.append(tokenizer.decode(out[0], skip_special_tokens=True))
    if i % 100 == 0:
        print("Processed", i, "of", len(df), flush=True)
df["PREDICTION"] = predictions
df.to_csv(out_path, index=False)
print("Saved", out_path)
