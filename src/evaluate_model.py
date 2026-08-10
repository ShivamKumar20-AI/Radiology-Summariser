import pandas as pd
# importing pandas to read and work with the CSV files

import torch
# importing PyTorch to run the model


from transformers import (
    AutoTokenizer,
    # turns text into tokens

    AutoModelForSeq2SeqLM
    # loads the sequence-to-sequence model
)


from rouge_score import rouge_scorer
# importing the tool used to calculate the ROUGE score


data = pd.read_csv("data/processed/test_data.csv")
# reading the test CSV file

print("Test data loaded")
# checking that the test data has loaded

print("Number of test rows:", len(data))
# showing how many test rows were loaded


model_path = "model/final"
# setting the location of the fine-tuned model

model_path = "model/final"
model_name = "google-t5/t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_path)
# loading the tokenizer saved during training

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
# loading the fine-tuned sequence-to-sequence model

print("Model name:", model_name)

model.eval()
# putting the model into evaluation mode
# evaluation mode means the model is being tested, not trained


print("Model loaded")
# confirming that the model loaded successfully


def create_summary(findings):
    # creating a function to generate one impression from one findings text

    text = "summarize radiology findings: " + findings
    # adding the same instruction that was used during training

    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    # converting the findings text into tokens
    # max_length limits the input to 512 tokens
    # truncation shortens text longer than 512 tokens
    # return_tensors converts the tokens into PyTorch tensors

    with torch.no_grad():
        # preventing gradients from being calculated during evaluation

        output = model.generate(
            **inputs,
            max_length=128,
            num_beams=4
        )
        # asking the model to generate an impression
        # max_length limits the output to 128 tokens
        # num_beams makes the model consider four possible outputs

    summary = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
    # converting the model's token output back into readable text

    return summary
    # returning the generated impression


scorer = rouge_scorer.RougeScorer(
    ["rouge1"],
    use_stemmer=True
)
# creating the ROUGE-1 scorer
# use_stemmer helps compare similar word forms


predictions = []
# creating an empty list to store generated impressions

scores = []
# creating an empty list to store ROUGE scores


for number, (findings, real_impression) in enumerate(
    zip(data["FINDINGS"], data["IMPRESSION"]),
    start=1
):
    # going through the findings and real impressions
    # number counts which report is being processed

    prediction = create_summary(findings)
    # generating an impression from the findings

    score = scorer.score(
        real_impression,
        prediction
    )["rouge1"].fmeasure
    # comparing the generated impression with the real impression
    # extracting the ROUGE-1 F1 score

    predictions.append(prediction)
    # adding the generated impression to the predictions list

    scores.append(score)
    # adding the ROUGE score to the scores list

    if number % 10 == 0:
        # checking whether 10 reports have been processed

        print(
            "Processed",
            number,
            "of",
            len(data),
            "reports",
            flush=True
        )
        # showing progress in the terminal


average_score = sum(scores) / len(scores)
# calculating the average ROUGE-1 F1 score


print(
    "Average ROUGE-1 F1:",
    round(average_score, 4)
)
# displaying the average ROUGE-1 F1 score


data["PREDICTION"] = predictions
# adding the generated impressions as a new column


data.to_csv(
    "data/processed/predictions_data.csv",
    index=False
)

print("prediction saved")