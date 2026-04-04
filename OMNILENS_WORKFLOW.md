# OmniLens Pro: Complete Platform Workflow and Feature Guide
### Every Feature Explained with Input/Output Examples

> **Platform:** OmniLens Pro — AI-Powered Shopping Intelligence
> **Version:** 2.0 (April 2026)
> **Audience:** End users, developers, and stakeholders

---

## Overview

OmniLens Pro is an AI-powered shopping assistant that takes a natural language query and produces a ranked list of real products with detailed scoring. The system uses a **7-stage pipeline** from intent detection through product evaluation.

```
User Query
    |
    v
[Stage 0] Query Clarifier (Pre-processing & UX Confirmation)
    |-- Detects Vagueness / Spelling Errors
    |-- Restructures Query cleanly
    |-- Asks User YES/NO to confirm format
    |
    v (On 'YES')
[Stage 1] Intent Classification (3-tier pipeline)
    |
    v SCENARIO or PRODUCT
    |
    |-- SCENARIO --> [Stage 2A] Category List Generation (Flan-T5)
    |                    |
    |                    v
    |               [Stage 3] Multi-node Parallel Scraping
    |
    |-- PRODUCT --> [Stage 2B] Variant Generation (Flan-T5)
                        |
                        v
                   [Stage 3] Parallel Scraping (Amazon + Flipkart)
                        |
                        v
                   [Stage 4] Multi-signal Product Evaluation
                        |
                        v
                   [Stage 5] Composite Scoring and Ranking
                        |
                        v
                   [Stage 6] Results Display (UI)
                        |
                        v
                   [Stage 7] Explore Further / Assistant / Cart
```

---

## Feature 1: Main Search (Home Page)

### What it does
The entry point. User types any shopping goal — from specific products to abstract lifestyle goals. The system determines what kind of request it is and routes accordingly.

### Input → Decision flow

#### Case 1: Specific Product Query (PRODUCT route)

```
Input:  "Sony WH-1000XM5 headphones"
        "AirPods Pro 2"
        "Best gaming laptop under 80000"
        "iPhone 15 Pro Max"
```

**What happens:**
1. Tier 1 taxonomy check — recognized as direct product
2. Variant generator expands into model alternatives
3. Scraper dispatches parallel fetches on Amazon + Flipkart
4. Products evaluated, ranked, displayed on results page

```
Output: Grid of 12–20 products, sorted by composite score
        Each card shows: image, title, price, Match score, star rating, buy link
```

---

#### Case 2: Lifestyle / Scenario Goal (SCENARIO route)

```
Input:  "I'm setting up a home theater"
        "Planning a hiking trip to the mountains"
        "Starting a home gym"
        "I'm a new baby parent"
```

**What happens:**
1. Tier 1 taxonomy match OR NLI classifier → identified as SCENARIO
2. Flan-T5 generates category checklist:
   - "home theater" → ["4K Projector", "Surround Sound System", "Streaming Device", "HDMI Cables"]
3. Each category node becomes a separate product search
4. All categories display as expandable sections on results page

```
Output: Sectioned product explorer
        ├── 4K Projector: [BenQ TK850, Epson EF-21, ...]
        ├── Surround Sound: [Sonos Arc, Samsung HW-Q990C, ...]
        └── Streaming Device: [Fire TV 4K, Apple TV 4K, ...]
```

---

#### Case 3: Budget-Constrained Query

```
Input:  "Laptop under 50000"
        "Best phone for 25000 rupees"
        "Camera within my budget of 30k"
```

**What happens:**
1. Budget parser extracts: {"amount": 50000, "currency": "INR"}
2. PRODUCT route with budget filter applied post-scraping
3. Products above budget threshold filtered out before ranking

```
Output: Products ranked by score, all within specified budget
        Budget badge shown on results page header
```

---

#### Case 4: Vague or Nonsensical Query (Fallback)

```
Input:  "aasdffg"
        "stuff"
        "I don't know what I want"
```

**What happens:**
1. Tier 1: No keyword match found
2. Tier 2 NLI: Confidence below threshold — fallback triggered
3. System displays friendly error with example queries

```
Output: "We could not understand your query. Try being more specific."
        Suggestions: "Try: Best laptop under 50000 or I am setting up a home office"
        Search bar refocused for retry
```

---

#### Case 5: Non-Shopping Query (Off-topic Redirect)

```
Input:  "What is the capital of France?"
        "Write me a poem"
```

**What happens:**
1. NLI classifier: Very low score for both PRODUCT and SCENARIO labels
2. System recognizes off-topic query

```
Output: "OmniLens is a shopping assistant. Try asking about products!"
        + Example shopping queries shown
```

---

## Feature 2: Query Clarifier (Pre-processing & Confirmation)

### What it does

Before any expensive ML models or web scrapers run, the *QueryClarifier* cleans up the user's input. It corrects spelling mistakes, resolves slang/abbreviations, detects vague scenarios, and formats them into a high-quality prompt. If the input was severely misspelled or overly vague, it pauses and asks the user to confirm ("Yes, search this" or "No, retype").

### Input → Decision Flow

#### Case 1: Slang and Typos

```
Input:  "wanna get best headfhones for my runing"
```

**What happens:**
1. Slang dictionary replaces "wanna" → "want to"
2. Fuzzy matching via difflib corrects "headfhones" → "headphones", "runing" → "running"
3. Re-formats into a clean product search

```
Output: "Find me the best headphones available in India with good reviews and value for money."
Confidence: Medium (due to multiple corrections)
UI: Pauses and asks user to confirm the corrected query.
```

#### Case 2: Broad/Vague Scenario

```
Input:  "i need stuff for college"
```

**What happens:**
1. No specific product detected. Treated as a vague action.
2. Reformats into a structured request.

```
Output: "Help me shop for: I need stuff for college. Find the most relevant products with good ratings and value for money."
Confidence: Low (very vague)
UI: Shows interpreted intent and requires user confirmation.
```

#### Case 3: Recognized Specific Scenario Template

```
Input:  "home gym"
```

**What happens:**
1. Triggers one of the 15+ hardcoded high-quality scenario templates immediately.

```
Output: "I want to set up a home gym. Help me find all the essential workout equipment and accessories."
Confidence: High 
UI: Pauses if confidence is High but a template expansion occurred, giving the user transparency on what will be searched.
```

---

## Feature 3: Intent Classification Pipeline (Behind the Scenes)

### Tier 1: Taxonomy Keyword Match (Instant, under 1ms)

Fast regex-based lookup against a compiled taxonomy of 200+ product categories and scenario keywords.

| Matched Keyword | Route | Example Categories Generated |
|---|---|---|
| "home theater", "projector setup" | SCENARIO | 4K Projector, Sound System, Streaming Device |
| "home gym", "fitness setup" | SCENARIO | Dumbbells, Yoga Mat, Resistance Bands |
| "gaming", "gaming setup" | SCENARIO | Monitor, Headset, Gaming Chair, Gaming Mouse |
| "iPhone", "AirPods", "MacBook" | PRODUCT | Exact model + close alternatives |
| "laptop", "headphones", "camera" | PRODUCT | Top models in category |

**If matched → instant route, no neural inference needed.**

---

### Tier 2A: ML Intent Classifier (Fast, under 50ms)

SentenceTransformer + Logistic Regression classifier trained on labeled intent pairs.

```
Input:  "I'm going backpacking across Europe"
Model:  SentenceTransformer('all-MiniLM-L6-v2') → embedding → LogisticRegression
Output: SCENARIO (93% confidence)
```

If confidence is below 70% → escalates to Tier 2B.

---

### Tier 2B: Zero-Shot NLI Fallback (Slower, under 500ms)

facebook/bart-large-mnli scores the query against two candidate labels.

```
Input:  "I need something for my new apartment"
Labels: ["a specific product", "a lifestyle goal requiring multiple products"]
Output: SCENARIO (71% for label 2) → route to scenario generator

Input:  "Samsung Galaxy S24 Ultra"
Labels: ["a specific product", "a lifestyle goal requiring multiple products"]
Output: PRODUCT (99% for label 1) → route to product search
```

---

## Feature 4: Category and Variant Generation (Flan-T5)

### 3A: Scenario → Category List

```
Model:  google/flan-t5-small
Input:  "starting a home office"
Output: ["Standing Desk", "Ergonomic Chair", "Monitor", "Webcam", "Keyboard", "Mouse", "Desk Lamp"]

Each category becomes a parallel search node.
Target: 5–8 categories per scenario.
```

### 3B: Product → Variant List

```
Model:  google/flan-t5-small
Input:  "Sony headphones"
Output: ["Sony WH-1000XM5", "Sony WH-1000XM4", "Sony WH-XB910N", "Sony WF-1000XM5"]

Each variant gets a parallel scrape.
Up to 10 variants dispatched concurrently.
```

---

## Feature 5: Product Scraping (Amazon + Flipkart)

### What it does

Launches headless Playwright browser sessions to scrape real-time product data for each search term.

### Input → Output per Query Node

```
Input:   "Sony WH-1000XM5"
         Platform: Amazon.in + Flipkart

Scraped data per product:
  Title:        "Sony WH-1000XM5 Wireless Headphones"
  Price:        26990
  Star rating:  4.4 / 5.0
  Review count: 3247 ratings
  URL:          amazon.in/dp/B09XS7JWHH
  Image URL:    [CDN link]
  Top reviews:  ["Excellent ANC", "Battery life is great", "Slightly tight on head"]

Output: Raw product dict passed to evaluator
```

### Anti-Scraping Resilience

```
If:   Amazon returns 503 / CAPTCHA
Then: New browser context launched with rotated user-agent
      Retry up to 3 times before falling back to Flipkart only
```

---

## Feature 6: Product Evaluation and Composite Scoring

Each scraped product passes through 6 scoring signals weighted into a composite score.

### Signal 1: Semantic Match Score (20% weight)

```
Query:  "noise cancelling headphones"
Title:  "Sony WH-1000XM5 Wireless Noise Cancelling Headphones"
Score:  0.87 (high — "noise cancelling" and "headphones" both matched)
```

### Signal 2: Brand Trust Score (15% weight)

```
Apple, Sony, Samsung, Bose → 0.90–1.0 trust multiplier
Boat, JBL, Skullcandy      → 0.70–0.85 multiplier
Unknown / generic brands   → 0.50 multiplier
```

### Signal 3: Star Rating Score (20% weight)

```
Input:  4.5 stars → Output: 90%
Input:  3.2 stars → Output: 64%
Formula: (rating / 5.0) × 100
```

### Signal 4: Sentiment Score (25% weight — highest)

```
Model:  cardiffnlp/twitter-roberta-base-sentiment-latest
Input Reviews:
  "Battery is amazing"  → Positive: 94%
  "Great ANC"           → Positive: 88%
  "Too tight on ears"   → Negative: 72%
  "Price is high"       → Neutral: 65%

Aggregate: mean positive probability = 68%
```

### Signal 5: Sales Volume Score (10% weight)

```
Input:  12,500 ratings → high confidence → near 100%
Input:  8 ratings      → low confidence → ~20%
Formula: min(1.0, log10(rating_count) / 5) × 100
```

### Signal 6: Price Value Score (10% weight)

```
Input:  Original 35000, Current 26990 → Discount 22.9%
Output: min(1.0, 0.5 + 22.9/100) × 100 = 72.9%

Input:  No discount present
Output: 50%
```

### Final Composite Score Example

```
Sony WH-1000XM5:
  semantic  = 87%
  brand     = 95%
  rating    = 88%
  sentiment = 68%
  volume    = 82%
  value     = 73%

Final = 0.20×87 + 0.15×95 + 0.20×88 + 0.25×68 + 0.10×82 + 0.10×73
      = 17.4 + 14.25 + 17.6 + 17.0 + 8.2 + 7.3
      = 81.75%

UI displays: Match Score: 82%
```

---

## Feature 7: Results Page

### What the user sees

A grid of ranked product cards with match scores, pricing, star ratings, and action buttons.

```
Results for "noise cancelling headphones"
─────────────────────────────────────────

[Product Image]  Sony WH-1000XM5
                 Rs 26,990
                 4.4 stars (3,247 reviews)
                 Match: 82%
                 [Buy on Amazon]  [Save to Wishlist]

[Product Image]  Bose QuietComfort 45
                 Rs 29,000  |  4.3 stars  |  Match: 79%
                 [Buy on Amazon]  [Save to Wishlist]

... (10–20 products total, sorted by match score)
```

### Sorting and Filtering

```
Sort by:  Match Score (default) | Price Low-High | Price High-Low | Rating
Filter:   Price range slider | Brand checkboxes | Minimum rating
```

---

## Feature 8: Explore Further (Product Graph)

### What it does

From any product card, "Explore Further" generates a semantic product graph showing related, complementary, and alternative products as an interactive node graph.

### Input → Output

```
Input:  User clicks "Explore Further" on Sony WH-1000XM5

System generates related node types:
  Alternatives:  ["Bose QC45", "Apple AirPods Max", "Sennheiser HD 660S"]
  Accessories:   ["Sony headphone stand", "Replacement ear pads", "3.5mm cable"]
  Complements:   ["Laptop for home audio", "DAC amplifier", "Bluetooth transmitter"]

Output: Force-directed graph visualization
        Each node is clickable with its own score
        Edge labels indicate relationship type
```

### Fallback: No related products found

```
If:   Scraper returns fewer than 2 related items
Then: "No related products found for this item."
      + Suggestion to search for a broader category
```

---

## Feature 9: Wishlist

### What it does

Save products to a persistent wishlist and revisit them later.

### Input → Output

```
Input:  User clicks "Save to Wishlist" on any product card
Output: Product added to Wishlist page
        Toast notification: "Added to your wishlist"
        Wishlist icon in navbar updates badge count

Input:  User opens Wishlist page
Output: Grid of saved products
        Remove button per product
        Total estimated budget for all saved items
        Share wishlist link (copyable URL)
```

### Edge Case: Duplicate saves

```
If:   Product is already in wishlist
Then: Toast: "Already in your wishlist" (no duplicate added)
```

---

## Feature 10: Cart and Budget Tracking

### What it does

Add products to a virtual cart for side-by-side comparison and budget management.

### Input → Output

```
Input:  User adds Sony WH-1000XM5 (Rs 26,990) + Laptop (Rs 55,000) to cart
Output: Cart page shows itemized list
        Running total: Rs 81,990
        Per-item: name, price, score, remove button
```

### Budget Alert

```
Input:  User set budget Rs 50,000
        Cart total Rs 81,990

Output: Cart exceeds your budget by Rs 31,990
        Alternative suggestion: "Try Sony WH-XB910N (Rs 14,990)"
```

---

## Feature 11: Receipts Page

### What it does

Tracks all past simulated purchases made through OmniLens Pro.

### Input → Output

```
Input:  User has placed 3 simulated purchases over the past month

Output: Receipt timeline (newest first)
        Per receipt: date, products purchased, total, match score at time of purchase
        Downloadable PDF receipt
        Re-buy shortcut: "Search again for similar products"
```

---

## Feature 12: AI Shopping Assistant (Chatbot)

### What it does

A conversational AI assistant embedded in OmniLens Pro that helps users refine searches, get personalized recommendations, compare specific products, and answer shopping-related questions. Powered by Qwen2-7B-Instruct with HieraSpark fine-tuning for shopping intent understanding.

### Conversation Flows

#### Case 1: Product Comparison

```
User:    "Compare the Sony WH-1000XM5 and Bose QC45"

System:  Fetches both products from results cache or triggers fresh scrape
         Runs evaluation on both

Output:
  Sony WH-1000XM5:   Match 82% | Rs 26,990 | ANC: Excellent | Battery: 30h
  Bose QuietComfort 45: Match 79% | Rs 29,000 | ANC: Very Good | Battery: 24h

  Recommendation:
  "For ANC quality and battery life, the Sony WH-1000XM5 is the better pick.
   If you prioritize comfort for long sessions, the Bose QC45 has a softer headband."
```

#### Case 2: Refinement / Follow-up

```
User:    [After seeing noise cancelling results]
         "Show me only ones under 25000"

Output:  Re-filters existing results for price < Rs 25,000
         Shows: Sony WH-XB910N (Rs 14,990), Jabra Evolve2 55 (Rs 22,000), ...
```

#### Case 3: Natural Language Recommendation Request

```
User:    "I work from home and have lots of video calls. What headset should I buy?"

System:  Extracts intent: work-from-home, video calls, headset
         Routes: PRODUCT route with modifiers [microphone quality, comfort, ANC]

Output:  "For video calls from home, I recommend looking at:
          1. Jabra Evolve2 75 — Best-in-class mic quality for calls
          2. Sony WH-1000XM5 — Excellent ANC blocks home noise
          3. Logitech H340 — Budget pick, great USB mic"
         + Triggers product search for all three
```

#### Case 4: Budget Question

```
User:    "What is the best phone I can get for 20000?"

System:  Budget extraction: Rs 20,000
         PRODUCT route: "best smartphone under 20000"

Output:  Ranked smartphones within budget
         Assistant commentary:
         "At Rs 20,000, the Redmi Note 13 Pro offers the best camera-to-price ratio.
          The Samsung Galaxy M34 is a better choice if battery life is your priority."
```

#### Case 5: Off-topic / Non-shopping

```
User:    "What is the weather today?" OR "Tell me a joke"

Output:  "I am specialized in shopping assistance. I can help you find products,
          compare prices, or build a shopping list. What are you looking to buy today?"
```

#### Case 6: Unclear Intent

```
User:    "I need a gift for my friend"

Output:  "Happy to help find a gift! A few quick questions:
          1. What is your budget? (e.g., under Rs 2000, under Rs 5000)
          2. What are your friend's interests? (tech, fashion, fitness, books...)
          3. Any specific product in mind or should I suggest ideas?"

→ Chatbot enters multi-turn clarification flow before triggering search
```

### Chatbot Architecture

```
Input message
     |
     v
Intent Detection (is it a shopping query, comparison, off-topic?)
     |
     v
If shopping:
  → Extract: product name, budget, constraints, comparison targets
  → Trigger OmniLens search pipeline
  → Return formatted product recommendations + commentary

If comparison:
  → Fetch both products from cache or scrape
  → Run scoring → generate comparison table
  → LLM generates natural language recommendation

If off-topic:
  → Polite redirect to shopping assistance
```

---

## Feature 13: Heuristic Fallbacks and Error Recovery

These run silently in the background to ensure robustness.

### Typo Correction

```
Input:  "aple macbook"
System: Matches "aple" to "Apple" via known entities set
Output: Searches for "Apple MacBook" correctly
```

### Scraper Fallback Chain

```
Primary:  Amazon.in scraper
If fail:  Retry with new browser context (up to 3 times)
If fail:  Fall back to Flipkart.com
If fail:  Return cached results (if available within 24h)
If fail:  Return error: "Products temporarily unavailable. Please try again."
```

### Low Confidence Fallback

```
If:   Intent classifier confidence < 50% and NLI score < 60%
Then: Show disambiguation prompt:
      "Are you looking for:
       A) A specific product to buy [PRODUCT]
       B) Products for a lifestyle goal [SCENARIO]
       Please click one or rephrase your search."
```

### Reliability Score Warning

```
If:  Product has < 20 reviews
Then: Yellow badge shown: "Low review count — score may not be reliable"

If:  Product has > 1000 reviews
Then: Green badge: "High confidence score"
```

---

## Full Pipeline Timing Reference

| Stage | Action | Typical Latency |
|---|---|---|
| Stage 1 | Tier 1 taxonomy match | < 1ms |
| Stage 1 | Tier 2A ML classifier | 30–80ms |
| Stage 1 | Tier 2B NLI fallback | 300–700ms |
| Stage 2 | Flan-T5 generation | 200–500ms |
| Stage 3 | Scraping (parallel, 10 nodes) | 8–20 seconds |
| Stage 4 | Sentiment scoring (RoBERTa) | 50–200ms per product |
| Stage 5 | Composite scoring | < 5ms per product |
| Stage 6 | UI render | < 100ms |
| **Total** | **End-to-end** | **10–25 seconds** |

---

## Summary: Decision Tree Quick Reference

```
User types query
       |
       |-- Contains known product name/brand? --> PRODUCT route
       |
       |-- Contains lifestyle/setup keyword?  --> SCENARIO route
       |
       |-- Has budget mention?               --> Extract budget + apply filter
       |
       |-- ML classifier says PRODUCT?       --> PRODUCT route
       |
       |-- ML classifier says SCENARIO?      --> SCENARIO route
       |
       |-- Low confidence (<60%)?            --> Show disambiguation
       |
       |-- Off-topic / nonsense?             --> Friendly error + examples
```
