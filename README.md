# Radiology Summariser

I built this project to experiment with fine-tuning language models for medical text summarisation. It uses a T5-small model trained to generate radiology impressions from chest X-ray findings.

## What's under the hood

The base model is `google-t5/t5-small`, and I saved the fine-tuned weights to `model/final`. I trained it for 2 epochs. After the first epoch, I was getting a ROUGE-1 F1 of about 0.41, which bumped up to 0.45 after the second pass—not a huge jump, but enough to see the model learning something useful.

## How the repo's laid out

```text
src/
├── api.py
├── train_model.py
└── evaluate.py

model/
└── final/

data/
└── processed/
    └── predictions_epoch2.csv
```

The `src` folder has the API code, training script, and evaluation logic. The trained model lives in `model/final`, and I've dropped some sample predictions in the `data/processed` directory.

## Getting it running

First, grab the dependencies:

```bash
pip install -r requirements.txt
```

Then fire up the API:

```bash
uvicorn src.api:app --reload
```

Once it's running, you can hit the docs at `(https://radiology-summariser.fly.dev/docs)` to see what endpoints are available.

## Trying it out

Hit the summarisation endpoint with a POST request. Here's an example payload:

```json
{
  "findings": "The lungs are clear. No pleural effusion or pneumothorax is present."
}
```

The model should spit back a generated impression based on what you send.

## How it performed

The ROUGE-1 F1 score went from 0.4090 after epoch one to 0.4509 after epoch two. When I tested the API manually, it picked up on some abnormalities reasonably well, but it did miss a few clinically important findings here and there—so there's definitely room for improvement.

## A word of caution

This is very much a research prototype. Don't use it for actual clinical diagnosis or as a substitute for a trained radiologist. It would need a lot more training, proper validation, and clinical evaluation before it could be trusted in any real-world setting.
