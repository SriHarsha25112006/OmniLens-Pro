# OmniLens Pro 🔭
### AI-Powered Shopping Intelligence Platform

> **Version:** 2.0 (April 2026) · **Author:** Sri Harsha · **Architecture:** HieraSpark PEFT

OmniLens Pro is a full-stack, local-first AI shopping agent that takes any natural language query — from a specific product to an abstract lifestyle goal — and autonomously scrapes real-time marketplace data, evaluates every product through a multi-signal ML pipeline, and delivers a ranked, scored results page. It runs entirely on your machine with no external AI API required.

---

## 📋 Table of Contents

1. [Features](#-features)
2. [Tech Stack](#-tech-stack)
3. [System Requirements](#-system-requirements)
4. [Project Structure](#-project-structure)
5. [Setup & Installation](#-setup--installation)
   - [Step 1 — Clone the Repository](#step-1--clone-the-repository)
   - [Step 2 — Backend Setup (ML Engine)](#step-2--backend-setup-ml-engine)
   - [Step 3 — Frontend Setup (Next.js UI)](#step-3--frontend-setup-nextjs-ui)
   - [Step 4 — Run Everything](#step-4--run-everything)
6. [How It Works — The 7-Stage Pipeline](#-how-it-works--the-7-stage-pipeline)
7. [AI Models Used](#-ai-models-used)
8. [HieraSpark Architecture](#-hiraspark-architecture)
9. [Scoring System](#-scoring-system)
10. [Configuration](#-configuration)
11. [Deployment Notes](#-deployment-notes)
12. [Citation](#-citation)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Natural Language Search** | Type anything — product names, lifestyle goals, vague ideas |
| 🧠 **3-Tier Intent Classifier** | Taxonomy → ML → NLI zero-shot fallback chain |
| 🤖 **Query Clarifier** | Auto-corrects typos, slang, abbreviations before searching |
| 🕷️ **Live Marketplace Scraping** | Real-time Playwright scraper for Amazon.in + Flipkart |
| 📊 **6-Signal Composite Scoring** | Sentiment · Semantic Match · Brand Trust · Rating · Volume · Discount |
| 💬 **AI Shopping Assistant** | Conversational chatbot for comparisons, filtering, and recommendations |
| 🔗 **Explore Further (Graph)** | Interactive force-directed product relationship graph |
| 💖 **Wishlist & Cart** | Save products, track budgets, get budget-exceeded alerts |
| 🧾 **Receipts** | History of simulated purchases with downloadable PDF |
| 🔁 **Anti-Ban Scraper** | Rotates browser contexts, user-agents, and falls back to alternate platforms |
| 🧬 **HieraSpark PEFT** | Novel spectral adapter architecture powering the AI assistant |

---

## 🛠️ Tech Stack

### Frontend (`omnilens/`)
| Technology | Version | Role |
|---|---|---|
| **Next.js** | 16.1.6 | React framework + SSR |
| **React** | 19.2.3 | UI library |
| **TypeScript** | 5 | Type safety |
| **Tailwind CSS** | 4 | Utility-first styling |
| **Framer Motion** | 12 | Animations |
| **Zustand** | 5 | Global state management |
| **Lucide React** | — | Icon set |

### Backend (`omnilens-ml/`)
| Technology | Role |
|---|---|
| **FastAPI** | REST API server |
| **Uvicorn** | ASGI server |
| **Playwright** | Headless browser scraping |
| **Transformers (HuggingFace)** | BART, Flan-T5, RoBERTa models |
| **PyTorch** | Model inference |
| **Sentence-Transformers** | Semantic embeddings |
| **scikit-learn** | Logistic Regression intent classifier |
| **BeautifulSoup4** | HTML parsing |
| **aiohttp** | Async HTTP |

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10/11, macOS, Linux | Windows 11 / Ubuntu 22.04 |
| **Python** | 3.10+ | 3.11 |
| **Node.js** | 18+ | 20 LTS |
| **RAM** | 8 GB | 16 GB |
| **Storage** | ~5 GB (models cache) | 10 GB SSD |
| **GPU** | Not required | NVIDIA (for faster inference) |

> ⚠️ **RAM Note:** The ML backend loads BART-Large-MNLI, Flan-T5-Small, and RoBERTa concurrently on first use. Expect ~4–6 GB RAM usage during scraping + inference.

---

## 📁 Project Structure

```
OmniLens Pro/
│
├── run_servers.py             ← Main launcher (starts both servers)
├── open_omnilens.bat          ← Windows double-click launcher
│
├── omnilens/                  ← Next.js Frontend
│   ├── src/
│   │   ├── app/               ← Next.js App Router pages
│   │   └── components/        ← Reusable UI components
│   ├── public/                ← Static assets
│   ├── package.json
│   └── next.config.ts
│
├── omnilens-ml/               ← Python ML Backend
│   ├── ml_engine/
│   │   ├── main.py            ← FastAPI app + all API routes
│   │   ├── models/
│   │   │   ├── intent_parser.py        ← 3-tier intent classifier
│   │   │   ├── intent_taxonomy.py      ← 200+ keyword taxonomy
│   │   │   ├── query_clarifier.py      ← Typo/slang correction
│   │   │   ├── hiraspark_adapter.py    ← HieraSpark PEFT implementation
│   │   │   ├── hiraspark_finetune.py   ← DPO training script
│   │   │   └── finetuner.py            ← RoBERTa + HieraSpark integration
│   │   └── services/
│   │       ├── scraper.py              ← Playwright stealth scraper
│   │       ├── evaluator.py            ← 6-signal scoring engine
│   │       └── session_manager.py      ← Session state management
│   ├── requirements.txt
│   └── run_server.py
│
├── HIRASPARK_ARCHITECTURE.md  ← HieraSpark PEFT full specification
├── HIRASPARK_REPORT.md        ← Research report & comparisons
├── OMNILENS_WORKFLOW.md       ← Full feature workflow documentation
└── System_Architecture.md     ← High-level system diagram
```

---

## 🚀 Setup & Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/SriHarsha25112006/OmniLens-Pro.git
cd "OmniLens-Pro"
```

---

### Step 2 — Backend Setup (ML Engine)

The backend is a **Python FastAPI** server. It must have its own virtual environment.

```bash
# Navigate to the ML backend directory
cd omnilens-ml

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt

# Install Chromium for Playwright (headless browser)
playwright install chromium
```

> 📝 **First-run model download:** On the first search, HuggingFace Transformers will automatically download BART-Large-MNLI (~1.6 GB), Flan-T5-Small (~300 MB), and RoBERTa (~500 MB) to your local cache (`~/.cache/huggingface/`). This only happens once.

---

### Step 3 — Frontend Setup (Next.js UI)

```bash
# Navigate to the frontend directory (from project root)
cd omnilens

# Install Node.js dependencies
npm install
```

---

### Step 4 — Run Everything

You have two options:

#### Option A — Unified Launcher (Recommended)

From the **project root**, run:

```bash
python run_servers.py
```

This script will:
1. Kill any processes already using ports `3000` (frontend) or `8000` (backend)
2. Start the Next.js frontend (`npm run dev`)
3. Start the FastAPI ML backend (`python -m ml_engine.main`)
4. Wait 4 seconds, then auto-open `http://localhost:3000` in your browser

Press **Ctrl+C** in the terminal to gracefully shut down both servers.

#### Option B — Windows Double-Click

Double-click **`open_omnilens.bat`** in the project root. This runs `run_servers.py` inside a terminal window.

#### Option C — Manual (Two Separate Terminals)

**Terminal 1 — Backend:**
```bash
cd omnilens-ml
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
python -m ml_engine.main
```

**Terminal 2 — Frontend:**
```bash
cd omnilens
npm run dev
```

Then open `http://localhost:3000` in your browser.

---

## 🧠 How It Works — The 7-Stage Pipeline

```
User Query
    │
    ▼
[Stage 0]  Query Clarifier
           ├── Corrects typos & slang (difflib + Flan-T5 span masking)
           └── Asks user to confirm severely corrected queries
    │
    ▼ (confirmed)
[Stage 1]  Intent Classification (3-tier)
           ├── Tier 1: Taxonomy keyword match  (<1ms)
           ├── Tier 2A: SentenceTransformer + LogReg  (~50ms)
           └── Tier 2B: BART-Large-MNLI zero-shot  (~500ms fallback)
    │
    ├── SCENARIO ──────────────────────────────────────────────┐
    │                                                          │
    ▼                                                          ▼
[Stage 2A] Variant Generation               [Stage 2B] Category Generation
           Flan-T5 product variants                   Flan-T5 scene checklist
           (up to 10 per query)                        (5–8 categories)
    │                                                          │
    └──────────────────────┬───────────────────────────────────┘
                           ▼
[Stage 3]  Parallel Scraping (Playwright Stealth)
           ├── Amazon.in + Flipkart.com (concurrent nodes)
           ├── Human-mimicry: randomized timing, user-agent rotation
           └── Fallback chain: retry → alternate context → cache
                           │
                           ▼
[Stage 4]  Product Evaluation (6-signal scoring)
           ├── Semantic Match  (all-MiniLM-L6-v2)
           ├── Sentiment       (twitter-roberta-base-sentiment)
           ├── Brand Trust     (curated multiplier table)
           ├── Star Rating     (normalized 0–100)
           ├── Sales Volume    (log-scaled review count)
           └── Price Value     (discount percentage)
                           │
                           ▼
[Stage 5]  Composite Scoring & Ranking
           Final = 20%·Semantic + 25%·Sentiment + 20%·Rating
                 + 15%·Brand + 10%·Volume + 10%·Value
                           │
                           ▼
[Stage 6]  Results Page (Next.js UI)
           Sortable grid with filters, product cards, buy links
                           │
                           ▼
[Stage 7]  Post-Search Features
           ├── Explore Further (product graph)
           ├── AI Shopping Assistant (Chatbot)
           ├── Wishlist · Cart · Receipts
           └── RLHF feedback loop → re-weights scoring signals
```

**End-to-end latency:** typically **10–25 seconds** (dominated by live scraping).

---

## 🤖 AI Models Used

| Model | Source | Role | Size |
|---|---|---|---|
| `facebook/bart-large-mnli` | HuggingFace | Zero-shot intent classification (Tier 2B) | ~1.6 GB |
| `google/flan-t5-small` | HuggingFace | Category & variant list generation | ~300 MB |
| `all-MiniLM-L6-v2` | Sentence-Transformers | Semantic similarity scoring | ~90 MB |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | HuggingFace | Product review sentiment analysis | ~500 MB |
| **SentenceTransformer + scikit-learn LogReg** | Local (pre-trained) | Fast Tier 2A intent classifier | ~5 MB (`.pkl`) |
| **HieraSpark (Qwen2-7B base)** | Local | AI Shopping Assistant backbone | Optional |

> All models are downloaded automatically on first use via HuggingFace Hub. No API keys required.

---

## 🔬 HieraSpark Architecture

HieraSpark is a **novel PEFT (Parameter-Efficient Fine-Tuning)** architecture developed for OmniLens Pro that powers the AI Shopping Assistant. It introduces three original components:

### 1. RotarySpectralGate (RSG)
A dual-path tanh-bounded complex gate operating on the **sequence dimension** via FFT/iFFT. Unlike LoRA (weight-domain) or FDA (hidden-dimension FFT), RSG captures temporal frequency patterns — low-frequency semantic structure and high-frequency syntactic noise — with a guaranteed zero-disruption initialization.

### 2. SpectralKernelBank (SKB)
A threshold-activated sparse bank of `N` learnable frequency-domain kernels. A learnable threshold gate creates input-adaptive sparsity — zero-cost routing for low-energy tokens, spectral modulation for high-energy positions.

### 3. Hierarchical Cross-Layer Distillation (HCLD)
An intra-model, training-only auxiliary loss where deep HieraSpark adapters act as teachers for shallow adapters within the same training run. Zero inference overhead — projection heads are removed at inference.

```
Full HieraSpark vs LoRA baseline:
  GLUE Avg:   85.4 → 87.2  (+1.8%)
  Intent F1:  88.2 → 93.7  (+5.5%)
  Convergence: 1200 steps → 820 steps (32% faster)
```

📄 See [`HIRASPARK_ARCHITECTURE.md`](./HIRASPARK_ARCHITECTURE.md) for the full mathematical specification and implementation.

---

## 📊 Scoring System

Every product is scored on six weighted signals:

| Signal | Weight | How it's computed |
|---|---|---|
| **Semantic Match** | 20% | Cosine similarity between query embedding and product title |
| **Sentiment** | 25% | Mean positive probability across top reviews (RoBERTa) |
| **Star Rating** | 20% | `(stars / 5.0) × 100` |
| **Brand Trust** | 15% | Curated brand tier multiplier (Apple/Sony = 0.95, unknown = 0.50) |
| **Sales Volume** | 10% | `min(1.0, log10(review_count) / 5) × 100` |
| **Price Value** | 10% | Discount percentage bonus over a 50% base |

**Example — Sony WH-1000XM5:**
```
Semantic=87% · Brand=95% · Rating=88% · Sentiment=68% · Volume=82% · Value=73%
Final Score = 0.20×87 + 0.15×95 + 0.20×88 + 0.25×68 + 0.10×82 + 0.10×73 = 81.75%
```

---

## ⚙️ Configuration

The backend API runs on **`http://localhost:8000`** and the frontend on **`http://localhost:3000`**.

To change the backend URL the frontend talks to, edit:

```
omnilens/.env.local
```

Default content:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🌍 Deployment Notes

OmniLens Pro is designed for **local-first operation**. For public deployment:

- **Docker**: Containerize both services and expose ports 3000 and 8000.
- **Cloud platforms** (Render, Railway, Fly.io): Require a plan with **≥ 4 GB RAM** to prevent OOM during model inference.
- **GPU acceleration**: Set `device="cuda"` in the model loading calls inside `ml_engine/main.py` for significantly faster inference.
- **Model caching**: On cloud deployments, mount a persistent disk at `~/.cache/huggingface/` to avoid re-downloading models on each restart.

---

## 📚 Documentation Index

| File | Purpose |
|---|---|
| [`HIRASPARK_ARCHITECTURE.md`](./HIRASPARK_ARCHITECTURE.md) | Full HieraSpark PEFT mathematical specification |
| [`HIRASPARK_REPORT.md`](./HIRASPARK_REPORT.md) | Research report: novelty analysis & comparison vs FDA, LoRA |
| [`OMNILENS_WORKFLOW.md`](./OMNILENS_WORKFLOW.md) | Complete feature workflow with input/output examples |
| [`System_Architecture.md`](./System_Architecture.md) | High-level system architecture diagram |

---

## 📖 Citation

If you use HieraSpark or OmniLens Pro in your research, please cite:

```bibtex
@misc{hiraspark2026,
  title  = {HieraSpark: Hierarchical Spectral Adapters with Cross-Layer Distillation for PEFT},
  author = {Sri Harsha},
  year   = {2026},
  note   = {OmniLens Pro — Local-First AI Shopping Intelligence Platform},
  url    = {https://github.com/SriHarsha25112006/OmniLens-Pro}
}
```

---

<div align="center">
  <sub>OmniLens Pro v2.0 — Built with 🔭 by Sri Harsha</sub>
</div>