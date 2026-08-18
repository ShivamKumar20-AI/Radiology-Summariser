import torch
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

@app.get("/")
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

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

@app.post("/get_summary")
def summarise(request: SummaryRequest):
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