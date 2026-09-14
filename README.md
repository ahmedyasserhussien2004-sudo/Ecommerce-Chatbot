[README.md](https://github.com/user-attachments/files/32215873/README.md)
# RAG-Based E-commerce Customer Support Chatbot

An end-to-end pipeline: **language detection → sentiment/emotion classification →
intent routing → grounded RAG answer**, deployed as a FastAPI service.

```
Customer message
      │
      ├─► Language Detection   (TF-IDF char n-grams + LinearSVC)
      ├─► Sentiment/Emotion    (fine-tuned DistilBERT)
      └─► Intent Classifier    (TF-IDF word n-grams + LogisticRegression)
                │
                ▼
        route decision
   ┌────────────┼─────────────┬──────────────┐
 smalltalk   complaint   out_of_scope    order_status / order_management /
   │            │             │          billing_and_refunds / account_management
 canned      escalation    "can't help,        │
 reply       message       rephrase?"          ▼
                                         RAG (FAISS/Qdrant + MiniLM
                                         embeddings + Groq gpt-oss)
```

## Project layout

```
ecommerce_support_chatbot/
├── src/                      # the 4 modules + shared utils + orchestration
│   ├── utils.py                  preprocessing, intent→route mapping
│   ├── language_detection.py     Module 1
│   ├── sentiment_classifier.py   Module 2
│   ├── intent_classifier.py      Module 3
│   ├── rag_pipeline.py           Module 4
│   └── pipeline.py               ties all 4 together (used by the API)
├── app/
│   ├── main.py                # FastAPI app (/chat, /health)
│   └── schemas.py             # request/response models
├── scripts/                  # one-shot CLI scripts to train / build / test
│   ├── train_language_detector.py
│   ├── train_sentiment_classifier.py
│   ├── train_intent_classifier.py
│   ├── build_vector_store.py
│   └── test_api.py
├── notebooks/                # the 4 deliverable notebooks (mirror the scripts, with commentary)
├── tests/                    # fast offline tests (no GPU / API key needed)
├── models/                   # trained artifacts land here (gitignored, empty until you train)
├── requirements.txt
└── .env.example
```

## Design decisions (quick reference for the assessment)

| Module | Approach | Why |
|---|---|---|
| Language ID | TF-IDF **character** n-grams (1-4, `char_wb`) + `LinearSVC` (calibrated) | Short messages, typo-robust, no huge vocab needed; standard approach for language ID |
| Sentiment/Emotion | Fine-tuned **DistilBERT** on 6 emotions, collapsed to 3-bucket sentiment at inference | ~20k rows is enough to fine-tune a pretrained head but thin for an RNN from scratch; small/fast enough for sync API calls. A documented BiLSTM fallback is in `sentiment_classifier.py` |
| Intent | TF-IDF word 1-2 grams + `LogisticRegression`, trained on the **gold** `intent` column, mapped to 7 routes post-hoc | Fully supervised (dataset provides labels), no zero-shot needed; mapping to routes is a cheap deterministic lookup so no signal is lost |
| RAG | `all-MiniLM-L6-v2` embeddings → FAISS (default) or Qdrant → Groq `gpt-oss-120b` generation, exact system prompt from the brief | Embed `instruction`, retrieve paired `response` as grounding context; FAISS needs zero external account, Qdrant is a drop-in swap via `VECTOR_BACKEND=qdrant` |
| Complaint routing | Bypasses RAG entirely — sentiment-negative + intent=complaint always gets an apology + human-escalation message | Per the brief's guideline to route complaints distinctly rather than auto-answering them |

---

## 1. Setup

```bash
cd ecommerce_support_chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set:
#   GROQ_API_KEY=...      <- free key from https://console.groq.com/keys
```

You need internet access for this step (to download the 3 HuggingFace
datasets, the DistilBERT checkpoint, and the MiniLM embedding model — all
one-time downloads, cached locally afterwards).

## 2. Train all 4 modules

Run these once, in order. Each one saves its artifacts under `models/`.

```bash
# Module 1 — Language Detection (~1-2 min on CPU)
python scripts/train_language_detector.py

# Module 2 — Sentiment/Emotion (fine-tunes DistilBERT; a few minutes on GPU,
# 20-40 min on CPU depending on hardware — lower --epochs in the script if needed)
python scripts/train_sentiment_classifier.py

# Module 3 — Intent Classifier (~1-2 min on CPU)
python scripts/train_intent_classifier.py

# Module 4 — Build the RAG vector index (embeds ~27k instructions, a few minutes)
python scripts/build_vector_store.py
```

Alternatively, open and run the notebooks in `notebooks/` in order
(`01` → `04`) — they call the exact same `src/` functions and include
extra evaluation/inspection cells and commentary.

After this step you should have:

```
models/
├── language_detection/language_detector.joblib
├── sentiment/final/                (tokenizer + model files)
├── intent/intent_classifier.joblib
└── rag_index/faiss.index, payload.jsonl, config.json
```

## 3. Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On startup you'll see log lines confirming all 4 models loaded — this
takes a few seconds (DistilBERT + MiniLM need to load into memory).

- Interactive docs: **http://localhost:8000/docs**
- Health check: `GET http://localhost:8000/health`
- Chat endpoint: `POST http://localhost:8000/chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order #48213?"}'
```

Response shape:
```json
{
  "response": "...",
  "route": "order_status",
  "language": {"language": "en", "confidence": 0.98},
  "sentiment": {"emotion": "sadness", "sentiment": "neutral", "confidence": 0.7},
  "intent": {"intent": "track_order", "route": "order_status", "confidence": 0.91},
  "sources": [{"instruction": "...", "score": 0.83, "category": "ORDER"}]
}
```

Or run the bundled smoke-test client, which fires a handful of varied
messages (greeting, order status, cancellation, an angry complaint, a
non-English message, and an out-of-scope question) at the running API:

```bash
python scripts/test_api.py
```

## 4. Run the offline tests (optional, fast)

These only touch the sklearn-based modules (no GPU, no API key, no
network) and are meant as a quick regression check while you iterate on
`src/utils.py` / the routing map:

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

## 5. Deploying beyond localhost

The app is a stock FastAPI/uvicorn service, so any of the usual options work:

- **Docker** (simplest for the assessment — see `Dockerfile` below if you
  add one): build once, `docker run -p 8000:8000 --env-file .env <image>`.
- **A single VM/server**: `pip install -r requirements.txt`, put `.env`
  next to the app, run uvicorn behind a process manager
  (`systemd`, `supervisor`, or `pm2`) so it restarts on crash, e.g.
  `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1`
  (keep `--workers 1` unless you also move the loaded models to a shared
  cache — each worker process otherwise reloads DistilBERT + MiniLM
  separately, which is fine for one worker on a laptop but wasteful for many).
- **Managed PaaS (Render/Railway/Fly.io/etc.)**: point it at this repo,
  set the start command to
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, and set `GROQ_API_KEY`
  (and `QDRANT_URL`/`QDRANT_API_KEY` if you're using the Qdrant backend)
  as environment variables/secrets in their dashboard — do **not** commit `.env`.

A minimal `Dockerfile` you can drop in:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note: `models/` (trained artifacts, several hundred MB once DistilBERT +
MiniLM + the FAISS index are in there) needs to either be baked into the
image/deploy bundle or trained as a build step — it's not included by
default since it's generated by step 2 above.

## 6. Troubleshooting

- **`FileNotFoundError` on `models/.../*.joblib` or `faiss.index`** — you
  skipped step 2; the API expects trained artifacts to already exist on disk.
- **Groq call fails with an auth error** — check `GROQ_API_KEY` is set in
  `.env` and that `python-dotenv`'s `load_dotenv()` in `app/main.py` is
  actually finding it (run uvicorn from the project root, not a subfolder).
- **Sentiment predictions look off on support-style messages** — expected
  to some degree given the Twitter→support-chat domain shift noted in the
  brief; see the qualitative spot-check cell in notebook `02` and consider
  hand-labeling a small supplementary set as suggested there.
- **Slow first request after startup** — the *first* Groq call and the
  *first* embedding call sometimes include connection warm-up; subsequent
  requests are much faster. Model loading itself happens once at startup
  (`lifespan` in `app/main.py`), not per-request.
