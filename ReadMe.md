# OmniLens Pro — Cognitive AI Shopping Architecture

OmniLens Pro is a futuristic, generative AI shopping agent that autonomously scans marketplaces to build, refine, and optimize your perfect shopping list.

## 🚀 The Cognitive Pipeline
The system uses a state-of-the-art ML pipeline to bridge the gap between human intent and marketplace data.

### 1. Intent Matrix (Parsing)
When you enter a prompt, our multi-tier parser identifies the context:
- **BART-Large-MNLI**: Classifies queries into **Scenarios** (e.g., "Skiing trip") or **Products** (e.g., "RTX 4090").
- **Flan-T5-Small**: Dynamically extrapolates missing items. If you mention a scenario, it automatically generates a list of 10 essential physical components.

### 2. Market Extrapolation (Scraping)
The engine utilizes a **Playwright Stealth** cluster to scrape real-time data from platforms like Amazon.
- **SSR Extraction**: Bypasses bot protections by interacting directly with the Server-Side Rendered DOM.
- **Human Mimicry**: Randomized interactions simulate organic browsing patterns.

### 3. Cognitive Scoring Engine
Every product is passed through a multi-dimensional scoring algorithm:
- **Semantic Match**: NLP comparison between query intent and product title.
- **Twitter-RoBERTa-Sentiment**: Real-time sentiment analysis of thousands of user reviews to detect quality issues or community praise.
- **Reliability Index**: A transparent metric based on data density (Brand Trust, Review Volume, etc.).

### 4. Generative Expansion
The Chat Assistant acts as a brain over the data:
- **Session Memory**: Tracks `seen_links` to ensure every "Explore Further" request delivers fresh products.
- **Data Manipulation**: High-level commands like "Remove the cheapest" or "Show me better brands."

---

## 🧠 Machine Learning Models
For deep technical details on the AI components, see:
👉 **[models_documentation.txt](./models_documentation.txt)**

---

## 🔧 Local Setup (System Requirements)
OmniLens Pro is designed for high-performance **Local-First** operation.

### Prerequisites
- **Python 3.10+** (for ML Engine)
- **Node.js 18+** (for Cyber-Glass UI)
- **RAM**: 8GB Minimum (16GB Recommended for running all ML models concurrently)

### Running the System
1. **Initialize Backend**:
   ```bash
   cd omnilens-ml
   pip install -r requirements.txt
   playwright install chromium --with-deps
   python -m ml_engine.main
   ```
2. **Initialize Frontend**:
   ```bash
   cd omnilens
   npm install
   npm run dev
   ```
3. **Access**: Open `http://localhost:3000` in your browser.

---

## 🌍 Public Deployment
While optimized for local use, the architecture supports containerized deployment (Docker) on platforms like Render or Railway for public access. Note that these require a "Pro" level plan with at least 2GB-4GB RAM to prevent OOM (Out-Of-Memory) crashes during AI inference.

---
*OmniLens Pro v8.5 — Local Matrix Active.*