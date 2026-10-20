import logging
import threading
import time

import torch
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Request  # importing in the API
from pydantic import BaseModel, Field

try:
    from src.guards import (
        MAX_CONCURRENT_SUMMARIES, MAX_FINDINGS_CHARS, MAX_REQUESTS_ALL_PER_MIN,
        MAX_REQUESTS_PER_IP_PER_MIN, PerKeyLimiter, SlidingWindowLimiter, client_address,
    )
except ImportError:  # when run as a script from inside src/
    from guards import (
        MAX_CONCURRENT_SUMMARIES, MAX_FINDINGS_CHARS, MAX_REQUESTS_ALL_PER_MIN,
        MAX_REQUESTS_PER_IP_PER_MIN, PerKeyLimiter, SlidingWindowLimiter, client_address,
    )

logger = logging.getLogger("radiology-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from transformers import(
    AutoTokenizer, # loading tokenizer and fine tuning model
    AutoModelForSeq2SeqLM
)

model_path = "model/final" # path of fine tuning model

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

model.eval() # only evaluate not test

# The interactive docs and schema are switched off on the public service.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

per_ip_limiter = PerKeyLimiter(MAX_REQUESTS_PER_IP_PER_MIN)
global_limiter = SlidingWindowLimiter(MAX_REQUESTS_ALL_PER_MIN)
work_slots = threading.BoundedSemaphore(MAX_CONCURRENT_SUMMARIES)

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
    findings: str = Field(min_length=1, max_length=MAX_FINDINGS_CHARS)

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
def summarise(request: SummaryRequest, http_request: Request):
    address = client_address(http_request.headers, http_request.client.host if http_request.client else "unknown")

    if not per_ip_limiter.allow(address) or not global_limiter.allow():
        logger.warning("rate limited")
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")

    if not work_slots.acquire(blocking=False):
        logger.warning("busy")
        raise HTTPException(status_code=503, detail="The service is busy. Please try again in a moment.")

    started = time.time()
    try:
        impression = create_summary(request.findings)
    except Exception:
        logger.exception("summary failed")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")
    finally:
        work_slots.release()

    # Log only the size and timing, never the report text.
    logger.info("summary ok chars=%d seconds=%.1f", len(request.findings), time.time() - started)
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