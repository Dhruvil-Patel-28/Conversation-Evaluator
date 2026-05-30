# Conversation Evaluator

## Overview

A production-ready pipeline that scores any conversation turn across **300 psychological, linguistic, and behavioral facets** using an open-weights LLM with confidence estimation.

Each facet receives:
- A **score** (1–5) indicating strength or presence
- A **confidence** (0.0–1.0) indicating how well the text supports the score
- Non-scoreable facets are flagged with `score=null` and a clear reason

---

## Live Demo

https://conversation-evaluator.streamlit.app/

---

## Architecture

```
┌─────────────────────────┐
│   Conversation Input    │
│   (user-typed text)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Preprocessing          │
│  (preprocess.py)        │
│  • Cleans 300 facets    │
│  • Tags scoreable vs    │
│    non-scoreable        │
│  • Assigns categories   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Facet Batcher          │
│  (batcher.py)           │
│  • Splits scoreable     │
│    facets into batches   │
│    of 20                │
│  • Handles non-scoreable│
│    separately           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  LLM Evaluator          │
│  (evaluator.py)         │
│  • Llama 3.1 8B via Groq│
│  • Score + confidence   │
│    per facet            │
│  • Retry logic for      │
│    failed parses        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Pipeline               │
│  (pipeline.py)          │
│  • Merges scoreable &   │
│    non-scoreable results│
│  • Returns unified      │
│    DataFrame            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Streamlit UI           │
│  (app.py)               │
│  • Input + samples      │
│  • Filters & charts     │
│  • Results table        │
│  • CSV download         │
└─────────────────────────┘
```

---

## Tech Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| LLM Model          | Llama 3.1 8B (Meta, open-weights)   |
| Inference          | Groq API                            |
| UI                 | Streamlit                           |
| Data Processing    | Pandas, NumPy                       |
| Progress Tracking  | tqdm                                |

---

## Hard Constraints — How We Satisfy Them

### 1. No one-shot prompting

Facets are batched into groups of **20**. Each batch is a separate, focused LLM call. Never all 300 facets in one prompt. This ensures:
- More accurate per-facet scoring
- Reliable JSON output from the model
- Manageable token counts per request

### 2. Open-weights model ≤ 16B

Using **Llama 3.1 8B** under the Meta open-weights license. Groq is only the inference provider — the model itself is fully open-weights and under 16B parameters. This satisfies the requirement while providing fast inference speeds.

### 3. Scales to 5,000 facets without redesign

The batch architecture means scaling equals more loop iterations:
- **300 facets** → ~15 API calls
- **5,000 facets** → ~250 API calls
- **Zero code changes required**

The only variable is the number of batches processed. The batcher automatically splits any number of facets into groups of 20.

---

## Facet Preprocessing Decisions

Started with **300 raw facets** from the provided CSV. The preprocessing pipeline:

1. **Removed category headers** — Rows ending with `:` (e.g., "Democratic Leadership:") are section headers, not actual facets
2. **Removed malformed entries** — Rows starting with digits followed by a dot (e.g., "800. Sufi practice...") are numbered sub-items that don't belong in the clean dataset
3. **Flagged non-scoreable facets** — Medical/biological facets (FSH, basophil, hormone, etc.) and spiritual/religious observance facets (I Ching, Kabbalah, Quran khatam, etc.) cannot be scored from conversation text alone
4. **All 300 facets preserved** — Non-scoreable facets appear in the final output with `score=null` and a clear reason, ensuring full transparency

---

## Scoring Scale

| Score | Meaning                              |
|-------|--------------------------------------|
| 1     | Extremely low / completely absent    |
| 2     | Low / slightly present               |
| 3     | Moderate / somewhat present          |
| 4     | High / clearly present               |
| 5     | Extremely high / strongly dominant   |

**Confidence (0.0 – 1.0):** How strongly the text supports the assigned score. Low confidence means the facet is hard to judge from the given text.

---

## How to Run Locally

```bash
# Step 1: Clone the repository
git clone <your-repo-url>
cd conversation-evaluator

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Step 4: Generate the cleaned facets CSV
python src/preprocess.py

# Step 5: Launch the Streamlit app
streamlit run app.py
```

---

## Running with Docker (Dockerized Baseline)

Docker support is fully provided in the codebase via the [Dockerfile](file:///c:/Users/HP/Desktop/Dhruvil/Resume%20Project/Ahoum/AI%20Assignment/Dockerfile) and [docker-compose.yml](file:///c:/Users/HP/Desktop/Dhruvil/Resume%20Project/Ahoum/AI%20Assignment/docker-compose.yml). 

To run using Docker:
```bash
docker compose up --build
```
This automatically preprocesses the facets and exposes the Streamlit dashboard on port `8501`.

> [!IMPORTANT]
> **Docker Verification Note:** While full Docker support has been coded and verified structurally (including a corresponding [.dockerignore](file:///c:/Users/HP/Desktop/Dhruvil/Resume%20Project/Ahoum/AI%20Assignment/.dockerignore)), local container runtime execution verification is currently pending due to a local WSL (Windows Subsystem for Linux) configuration issue on the development host.

---

## How to Deploy on Streamlit Cloud

1. **Push** your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. **Connect** your GitHub repo
4. Add `GROQ_API_KEY` in the **Secrets** settings
5. Click **Deploy**

---

## Project Structure

```
conversation-evaluator/
├── data/
│   ├── Facets_Assignment.csv        # Raw facets (provided)
│   └── facets_cleaned.csv           # Enriched facets (generated)
├── src/
│   ├── __init__.py
│   ├── preprocess.py                # Clean, classify, flag facets
│   ├── batcher.py                   # Split into batches of 20
│   ├── evaluator.py                 # LLM scoring via Groq
│   └── pipeline.py                  # Orchestrate & format results
├── samples/
│   ├── generate_samples.py          # 50 sample conversations
│   ├── generate_samples_fast.py     # Fast high-fidelity scoring script
│   ├── conversations_with_scores.json # 50 conversations & scores
│   ├── scores_summary.csv           # Tabular summaries of scores
│   └── conversations_with_scores.zip # Packaged scores deliverable
├── app.py                           # Streamlit web interface
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── Dockerfile                       # Container definition for app
├── docker-compose.yml               # Multi-container baseline setup
├── .dockerignore                    # Optimization block for image builds
└── README.md                        # This file
```
## Scaling to 5000 Facets

The batch architecture makes scaling trivial. Facets are split 
into batches of 20 regardless of total count. Going from 300 
to 5000 facets means 250 API calls instead of 17 — the same 
code, more iterations.

The only practical consideration at scale is API rate limits,
which is handled by the existing time.sleep() between batches
and can be tuned by adjusting batch size or sleep duration
without any architectural changes.

300 facets  → 17 batches  → ~3 mins
1000 facets → 50 batches  → ~10 mins  
5000 facets → 250 batches → ~45 mins

## Scaling to 5000 Facets

### Code Architecture ✅
Zero redesign needed. The batch loop handles any number
of facets identically.

### Rate Limits ✅  
Free tier Groq handles current load with time.sleep(1).
For 5000 facets in production, two options requiring
NO architectural changes:

Option A: Increase sleep between batches (slower but free)
Option B: Upgrade to Groq paid tier (faster, same code)

### For High-Scale Production
An async implementation using asyncio.Semaphore can process
5 batches concurrently, reducing 5000-facet evaluation from
~45 mins to ~9 mins. This is an optimization, not a redesign.
