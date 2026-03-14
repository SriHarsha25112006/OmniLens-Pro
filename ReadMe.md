# OmniLens Pro — Cognitive AI Shopping Architecture

OmniLens Pro is a futuristic, generative AI shopping agent that autonomously scans marketplaces to build, refine, and optimize your perfect shopping list.

## 🚀 The Cognitive Pipeline

### 1. Intent Matrix (Parsing)
When a user enters a prompt, the **Intent Parser** (Transformers/NLP) breaks it down into:
- **Core Subject:** (e.g., "4K Monitor")
- **Constraints:** (e.g., "Under ₹30,000", "Focus on Sony")
- **Action Type:** (e.g., "Full search", "Replace node #12", "Find top 10")

### 2. Market Extrapolation (Scraping)
The engine utilizes a **Playwright Stealth** cluster to scrape real-time data from Amazon.in.
- **SSR Extraction:** Bypasses basic bot protections by extracting directly from the Server-Side Rendered DOM.
- **Concurrency Management:** Strict semaphores ensure no more than 2 browser instances run at once, preventing IP bans.
- **Human Mimicry:** Randomized scrolling and delays simulate organic browsing patterns.

### 3. Cognitive Scoring Engine
Every product identified is passed through a multi-dimensional scoring algorithm (0-100):
- **Semantic Match:** NLP comparison between query intent and product title.
- **Brand Authority:** Weighted scores for trusted manufacturers vs. generic brands.
- **Reliability Index:** Real-time sentiment analysis of user reviews and ratings.
- **Value Vector:** Price-to-feature ratio analysis.

### 4. Generative Expansion
The **Chat Assistant** acts as a brain over the data:
- **Session Memory:** Tracks all "Seen Links" to ensure "Load More" / "Explore Further" requests always deliver fresh, relevant products.
- **Data Manipulation:** Understands natural language commands to "Remove the cheapest", "Replace with a better brand", or "Undo last change".
- **RL Preference Tuning:** Automatically parses chat input to adjust the underlying Multi-Dimensional Scoring Engine weights (e.g. prioritizing "rating" or "sentiment" over "price").
- **Predictive Wishlist Upgrades:** Integrates deeply with your Wishlist data to suggest 1 high-fidelity complementary item/upgrade for each node saved, displayed elegantly on the Wishlist page.

### 5. Cyber-Glass Interface
A premium, futuristic UI designed for high-end professional use:
- **Cyber-Glass Aesthetics:** Semi-transparent, backdrop-blurred panels with glowing neon accents and immersive ambient lighting overlays.
- **Data-Stream Animation:** Subtle animated background patterns indicating active AI computation.
- **Holographic Interactivity:** Magnetic hover effects and pulse-glow indicators for top-rated "Matrix Matches."

## 🔧 Internal Workflow Example

1. **User:** "Find me the best 4K TVs but not more than 50k"
2. **AI Node:** Parses "4K TV" as target, "50000" as budget cap.
3. **Scraper:** Initialized. Fetches 20 results from Amazon.
4. **Scoring:** Ranks them. TV 'A' is 20k but low rating (Score: 40). TV 'B' is 48k with 4.8 stars (Score: 92).
5. **Generative UI:** Displays TV 'B' as a **Holographic Card**.
6. **User:** "Show me more"
7. **AI Brain:** Checks `seen_links`, tells Scraper to ignore TV 'B', and fetches next-best alternatives.

## 🌍 Deployment

### Frontend (Next.js) - Vercel
OmniLens Pro's Cyber-Glass interface is designed for static exports or edge deployments:
1. Push the `omnilens` folder to a GitHub repository.
2. Import the project in [Vercel](https://vercel.com/).
3. Framework Preset: **Next.js**.
4. Set the Root Directory to `omnilens/`.
5. Deploy. The UI will instantly go live with edge-cached performance.

### Backend (FastAPI + Playwright) - Render / Railway
The AI engine requires Python and Playwright browser instances:
1. Ensure `omnilens-ml` has a valid `requirements.txt`.
2. Push the code to a GitHub repository.
3. Import the `omnilens-ml` project into **Render** as a Web Service, or **Railway**.
4. Build Command: `pip install -r requirements.txt && playwright install chromium --with-deps`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. *Important*: Update the frontend's API calls (e.g., in `page.tsx` and `cart/page.tsx`) to point to your new deployed backend URL.

---
*OmniLens Pro v8.0 — System Active.*