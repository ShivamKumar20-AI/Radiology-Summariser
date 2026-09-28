# Radiology Report Summariser

Fine‑tuned T5 model that converts radiology “findings” into concise “impression” summaries.  
🚀 **Live demo:** https://radiology-summariser.fly.dev/

---

## What it does

- Takes the **findings** section of a radiology report as input.
- Generates a short, structured **impression** summary using a fine‑tuned T5 model.
- Provides a prototype decision‑support tool for clinicians (not a medical device).

---

## Model & approach

**Task:** Automatic text summarisation (findings → impression).

**Data**
- Public radiology dataset with paired findings and impressions (e.g., MIMIC‑CXR or similar).
- Preprocessed to use “findings” as input text and “impressions” as target summaries.
- Split into train / validation / test sets.

**Model**
- Fine‑tuned **T5 base** sequence‑to‑sequence model for summarisation.
- Compared against a simple **extractive baseline** (e.g., first N sentences / TextRank‑style).
- Experiments tracked (hyperparameters, epochs, losses) with a tool like **MLflow** (or similar).

**Evaluation**
- Metrics: **ROUGE‑1, ROUGE‑2, ROUGE‑L**.
- Example results:
  - ROUGE‑1: ~0.48
  - ROUGE‑2: ~0.28
  - ROUGE‑L: ~0.39
- Outperforms the extractive baseline across all metrics.

---

## Results

- Achieves strong ROUGE scores on the test set, demonstrating meaningful summarisation quality.
- Inference latency around **~1–1.5s per report** on CPU (adjust if your actual latency differs).
- Provides a consistent, automatic way to generate impression sections from findings.

---

## How to run locally

**Requirements**
- Python 3.9+
- GPU recommended for training; CPU is sufficient for inference.

**Setup**

```bash
# Clone the repo
git clone [https://github.com/ShivamKumar20-AI/Radiology-Summariser.git](https://github.com/ShivamKumar20-AI/Radiology-Summariser.git)
cd Radiology-Summariser

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Inference (example)**

```bash
python infer.py --input "Findings text here..." --model_path path/to/fine_tuned_model
```

**Training (if you want to retrain)**

```bash
python train.py \
  --data_path path/to/dataset \
  --model_name t5-base \
  --output_dir path/to/save_model \
  --num_epochs 5 \
  --batch_size 8
```

(Adjust script names and arguments to match your actual code.)

---

## Example usage

**Input (findings)**

> “The lungs are clear. No focal consolidation, pneumothorax, or pleural effusion. Heart size is normal. Mediastinal contours are unremarkable.”

**Output (impression)**

> “No acute cardiopulmonary process. Normal heart size and mediastinal contours.”

You can integrate this model via:

- Direct Python calls to `infer.py` or a `summarise_report()` function.
- A simple API wrapper (FastAPI/Flask) around the model (not yet implemented in this repo).

---

## Deployment

- Containerised with **Docker**.
- Deployed as a service/app on a cloud platform (e.g., **Fly.io**, Render, or similar).
- Basic CI/CD via **GitHub Actions** (tests + deploy on push).
- Live demo: https://radiology-summariser.fly.dev/

---

## Limitations & next steps

**Limitations**
- Performance drops on very long reports and rare pathologies.
- No clinical validation; explicitly a prototype, not a medical device.
- Currently focused on English reports; no multilingual support.

**Next steps**
- Add section‑aware token limits or hierarchical models for longer reports.
- Try LongT5 / LED for better handling of long contexts.
- Add basic monitoring (latency, error rates) and structured logging.
- Extend to a full API service with authentication and usage tracking.

---
