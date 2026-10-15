"""Fine-tune T5-small with abnormal reports oversampled.

The training data is mostly normal studies, and the original model learned to
answer "No acute cardiopulmonary abnormality." far too often. Here every report
whose impression names a finding (see finding_metrics.py) is repeated
OVERSAMPLE times, so abnormal examples carry more weight.

    python src/train_balanced.py [oversample] [epochs]    # defaults: 3 and 3

The model is saved to model/balanced; the deployed model/final is untouched.
"""
import os
import sys

import pandas as pd
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

sys.path.insert(0, "src")
from finding_metrics import FINDINGS, mentions

OVERSAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MODEL_NAME = "google-t5/t5-small"
OUT_DIR = "model/balanced"

data = pd.read_csv("data/processed/train_data.csv").fillna("")
abnormal = data["IMPRESSION"].apply(lambda s: any(mentions(s, f) for f in FINDINGS))
balanced = pd.concat([data] + [data[abnormal]] * (OVERSAMPLE - 1), ignore_index=True)
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Reports: {len(data)} -> {len(balanced)} after oversampling "
      f"({abnormal.mean():.1%} -> {abnormal.sum() * OVERSAMPLE / len(balanced):.1%} abnormal)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def prepare(batch):
    inputs = tokenizer(["summarize radiology findings: " + f for f in batch["FINDINGS"]],
                       max_length=512, truncation=True)
    targets = tokenizer(text_target=batch["IMPRESSION"], max_length=128, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs


dataset = Dataset.from_pandas(balanced).map(prepare, batched=True, remove_columns=list(balanced.columns))

args = Seq2SeqTrainingArguments(
    output_dir="model/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", 8)),
    learning_rate=5e-5,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=42,
    use_cpu=os.environ.get("USE_CPU") == "1",  # set USE_CPU=1 on machines short of memory
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer,
                         data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model))
trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("Saved", OUT_DIR)
