import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

model_path="model/final" # location of fine tuning model

tokenizer=AutoTokenizer.from_pretrained(model_path) # loading in Tokenizer
model=AutoModelForSeq2SeqLM.from_pretrained(model_path) # load the model

print("Model and Tokenizer have been loaded")

model.eval() # evaluation mode not testing


sample_findings=[
    "The lungs are clear. The heart size is normal"

    "There is no focal air-space consolidation. Pleural effusion or pneumothorax"

    "Mild cardiomegaly is present. No pulmonary edman is identified"
]

def create_summary(findings):

    text= "summarise radiology findings:" + findings

    inputs=tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output= model.generate(
            **inputs,
            max_length=128,
            num_beams=4
        )

        summary=tokenizer.decode( # used to decode converted text so its readable
            output[0],
            skip_special_tokens=True
        )
    return summary

with open(
    "data/processed/sample_prediction.txt",
    "w"
) as file:

    for number, findings in enumerate(
        sample_findings,
        start=1
    ):

        prediction = create_summary(findings)

        print("Example", number)
        print("FINDINGS", findings)
        print("IMPRESSION:", prediction)
        print()

        file.write("Example " + str(number) + "\n")
        file.write("FINDINGS: " + findings + "\n")
        file.write("IMPRESSION: " + prediction + "\n\n")


print("Summary returned")




import os

print("Sample script model path:")
print(model_path)

print("Model name:")
print("google-t5/t5-small")

print("Tokenizer name:")
print(tokenizer)