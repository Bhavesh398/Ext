# 🛡️ PhishGuard ML — DistilBERT Phishing Detector

Fine-tuned DistilBERT model for real-time phishing email detection.
Powers the PhishGuard Chrome Extension backend.

---

## 📁 Structure

```
phishing-ml/
├── data/
│   ├── raw/                    ← Put your datasets here
│   ├── processed/              ← Auto-generated train/val/test splits
│   └── data_loader.py          ← Dataset loading + preprocessing
├── features/
│   ├── structural.py           ← Rule-based feature extraction
│   └── text_preprocessor.py   ← Text cleaning for DistilBERT
├── model/
│   ├── dataset.py              ← PyTorch Dataset class
│   ├── distilbert_classifier.py ← Model architecture
│   ├── trainer.py              ← Training loop
│   └── evaluate.py             ← Metrics + plots
├── artifacts/
│   ├── model/                  ← Saved model weights (auto-generated)
│   └── tokenizer/              ← Saved tokenizer (auto-generated)
├── explainability/
│   └── reasons.py              ← Score combination + reason generation
├── api/
│   ├── main.py                 ← FastAPI app
│   └── predictor.py            ← Inference engine
├── logs/                       ← Training logs + plots (auto-generated)
├── train.py                    ← Main training entry point
├── config.yaml                 ← All configuration
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
# Create conda environment
conda create -n phishguard python=3.10
conda activate phishguard

# Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify GPU
```python
import torch
print(torch.cuda.is_available())        # Should be True
print(torch.cuda.get_device_name(0))    # Your GPU name
```

---

## 📦 Get Training Data

Download this dataset from Kaggle (free, ~82k emails, already labeled):
```
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
```

1. Download the CSV file
2. Place it in `data/raw/` folder
3. Rename it to `phishing_dataset.csv`

**Optional additional datasets:**
- Nazario Phishing Corpus: https://www.monkey.org/~jose/phishing/
- Enron Dataset: https://www.cs.cmu.edu/~enron/

---

## 🚀 Train the Model

```bash
# Full pipeline (data prep + training + evaluation)
python train.py

# Skip data prep if already done
python train.py --skip-data

# Override hyperparameters
python train.py --epochs 5 --batch-size 32 --lr 1e-5

# Only evaluate saved model
python train.py --eval-only
```

**Expected training time on GPU:**
- RTX 3060 (12GB): ~2-3 hours
- RTX 4070 (12GB): ~1.5-2 hours
- RTX 4090 (24GB): ~45-60 minutes

---

## 🌐 Run the API

```bash
# Development
uvicorn api.main:app --port 8000 

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Test it:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "URGENT: Verify your PayPal account",
    "from": "PayPal <noreply@paypa1.tk>",
    "body": "Click here immediately to verify your account",
    "urls": ["http://paypa1.tk/verify"]
  }'
```

---

## 🚢 Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Set environment variable in Railway dashboard:
```
PORT=8000
```

Your API will be live at: `https://your-app.railway.app`

Then update your Chrome extension's `background.js`:
```javascript
const MOCK_API_URL = "https://your-app.railway.app/analyze";
```

---

## 📊 Target Performance

| Metric | Target | 
|---|---|
| F1 Score | > 0.96 |
| Recall | > 0.97 |
| Precision | > 0.95 |
| ROC-AUC | > 0.98 |
| Inference | < 200ms |

---

## 🔌 API Contract

```
POST /analyze
{
  "subject": "string",
  "from": "string", 
  "replyTo": "string",
  "body": "string",
  "urls": ["string"]
}

→ Response:
{
  "score": 87,
  "label": "phishing",
  "reasons": ["..."],
  "flags": ["DOMAIN_MISMATCH"],
  "senderAnalysis": {...},
  "urlAnalysis": {...},
  "confidence": 0.94
}
```
