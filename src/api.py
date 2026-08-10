import torch
from fastapi import FastAPI # importing in the API
from pydantic import BaseModel
from transformers import(
    AutoTokenizer, # loading tokenizer and fine tuning model
    AutoModelForSeq2SeqLM
)

model_path = "model/final" # path of fine tuning model

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

model.eval() # only evaluate not test

app = FastAPI() # 

class SummaryRequest(BaseModel):
    findings: str

def create_summary(findings): # same code we used in the sample code and when we evaluate the model.

    text= "summarize radiology findings: " + findings

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


#creating API endpoint

@app.get("/")
def home():

    return {
        "message": "Radiology summariser API is running"
    }


@app.post("/Summarize")
def summarise(request:SummaryRequest):
        impression = create_summary(request.findings)

        return {
        "impression" : impression
    }





import os

print("API model path:")
print(os.path.abspath(model_path))

print("Model name:")
print(model.config._name_or_path)

print("Tokenizer name:")
print(tokenizer.name_or_path)