import sys
import os

# CRITICAL: Windows Proactor Loop must be set BEFORE loop creation for Playwright/Subprocesses
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    _original_set_policy = asyncio.set_event_loop_policy
    def _prevent_selector_policy(policy):
        if not isinstance(policy, asyncio.WindowsProactorEventLoopPolicy):
            return # Ignore uvicorn's attempt to set SelectorEventLoopPolicy
        _original_set_policy(policy)
    asyncio.set_event_loop_policy = _prevent_selector_policy

import asyncio
import json
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playwright.async_api import async_playwright
from ml_engine.models.intent_parser import intent_parser
from ml_engine.services.scraper import scraper_service
from ml_engine.services.evaluator import scoring_engine
from ml_engine.services.session_manager import session_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OmniLens ML Engine")

# Allow Next.js frontend to access SSE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ShopRequest(BaseModel):
    prompt: str
    budgetStr: str

async def process_item(item: dict, context, log_q: asyncio.Queue) -> list[dict]:
    """
    Processes a single search term into multiple product nodes.
    """
    item_id = item['id']
    item_name = item['name']
    search_query = item.get('search_query', item_name)

    async def log_cb(msg: str):
        data_log = json.dumps({'event': 'log', 'data': {'message': msg, 'item_id': item_id, 'name': item_name}})
        await log_q.put(f"data: {data_log}\n\n")

    # Status: Scraping
    await log_q.put(f"data: {json.dumps({'event': 'item_update', 'data': {'id': item_id, 'status': 'scraping', 'statusText': f'Extracting items for {item_name}...', 'progress': 10}})}\n\n")

    scraped_results = await scraper_service.scrape_items(search_query, context=context, log_cb=log_cb)

    if not scraped_results:
        await log_q.put(f"data: {json.dumps({'event': 'item_update', 'data': {'id': item_id, 'status': 'complete', 'statusText': 'No nodes found', 'progress': 100}})}\n\n")
        return []

    processed = []
    # Limit to 8 products per search term to avoid UI overload
    for i, data in enumerate(scraped_results[:8]):
        unique_id = f"{item_id}_{i}"
        await log_q.put(f"data: {json.dumps({'event': 'item_update', 'data': {'id': unique_id, 'status': 'analyzing', 'statusText': f'Scoring {data["title"][:20]}...', 'progress': 50 + (i*5)}})}\n\n")
        
        score_data = scoring_engine.calculate_raw_score(data, float(item['essentiality']))
        
        processed.append({
            'id':           unique_id,
            'name':         data['title'],
            'target_query': item_name,
            'essentiality': float(item['essentiality']),
            'raw_score':    score_data['raw_score'],
            'price_inr':    score_data['price_inr'],
            'platform':     score_data['platform'],
            'image':        data.get('image'),
            'external_link': data.get('link'),
            'tags':         score_data.get('tags', []),
            'sentiment':    score_data.get('sentiment', 0.0),
            'is_bestseller': score_data.get('is_bestseller', False),
            'sales_volume':  score_data.get('sales_volume', 0),
            'discount_pct':  score_data.get('discount_pct', 0),
            'reliability_score': score_data.get('reliability_score', 0.0)
        })

    return processed

@app.post("/api/stream_shop")
async def stream_shop(req: ShopRequest):
    """
    Main SSE endpoint. Extrapolates intent, checks budget, and spawns concurrent scraping tasks.
    """
    # Parse INR budget
    try:
         budget_str = str(req.budgetStr).lower().strip()
         if not budget_str or budget_str in ['null', '0', '0.0', 'none']:
             budget = float('inf')
         else:
             budget = float(re.sub(r'[^0-9.]', '', req.budgetStr))
    except (ValueError, TypeError):
         budget = 50000.0 # Default 50k INR
         
    async def sse_generator() -> AsyncGenerator[str, None]:
        try:
            # ─── Query Specificity & Fast-Path ────────────────────────────────
            prompt_lower = req.prompt.lower().strip()
            
            # Use the robust ML intent classifier
            intent = intent_parser._classify_intent(prompt_lower)
            is_vague = (intent == "SCENARIO")
            logger.info(f"Query parsed. Intent: {intent} => is_vague: {is_vague}")
            
            if not is_vague:
                # Fast-Path: Specific product query (skip full checklist expansion)
                logger.info(f"Specific product query detected: '{req.prompt}'")
                items_to_scrape = [
                    {'id': '1', 'name': req.prompt, 'search_query': req.prompt, 'essentiality': 1.0, 'category': 'Gear'}
                ]
                items_to_show = 1
            else:
                # Scenario-Path: Building a checklist
                items = intent_parser.extrapolate_checklist(req.prompt)
                items_to_scrape = items[:10]
                items_to_show = 10
            
            intent_type = "Indefinite Query" if is_vague else "Definite Query"
            msg_expansion = json.dumps({'event': 'expansion', 'data': {'message': f'NLP Engine recognized intent ({intent}). Initiating search...', 'items': items_to_scrape, 'intent_type': intent_type}})
            yield f"data: {msg_expansion}\n\n"
            
            # Simple fallback check
            if budget < 500.0 and budget != float('inf') and is_vague:
                 msg_pivot = json.dumps({'event': 'pivot', 'data': {'message': f'Budget of ₹{budget} is restrictive. Showing top 2 essentials.', 'suggestedRentals': [i["name"] for i in items_to_scrape[2:]]}})
                 yield f"data: {msg_pivot}\n\n"
                 items_to_scrape = items_to_scrape[:2] # Truncate to top 2 essentials

            # ── Phase 2: Concurrent Scraping & ML Scoring via Playwright ───────
            logger.info("Spawning Playwright headless browser and concurrent tasks...")

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context_args = {
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "viewport": {"width": 1440, "height": 900},
                    "ignore_https_errors": True,
                    "locale": "en-IN",
                    "timezone_id": "Asia/Kolkata",
                }
                context = await browser.new_context(**context_args)

                # Each process_item streams live log/status events via log_q.
                # The item_finish is returned directly as a dict for batch processing.
                log_q: asyncio.Queue[str | None] = asyncio.Queue()

                async def run_all():
                    tasks = [asyncio.create_task(process_item(item, context, log_q)) for item in items_to_scrape]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    await log_q.put(None)  # Sentinel
                    return results

                run_task = asyncio.create_task(run_all())

                # Stream live logs while scraping is underway
                while True:
                    chunk = await log_q.get()
                    if chunk is None:
                        break
                    yield chunk

                nested_results = await run_task
                await context.close()
                await browser.close()

            # ── Flatten & Save Session Data ─────────────────────────────────────
            valid_results = []
            for sublist in nested_results:
                if isinstance(sublist, list):
                    valid_results.extend(sublist)

            if not valid_results:
                yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'No products could be retrieved. Please try a different query.'}})}\n\n"
                return

            # Save to session_products.json
            try:
                session_manager.save_products(valid_results)
            except Exception as e:
                logger.error(f"Failed to write session file: {e}")

            # Normalize scores across the batch
            valid_results = scoring_engine.normalize_scores(valid_results)

            # Sort by score descending so the best bubbles to top
            valid_results.sort(key=lambda r: r.get('raw_score', 0), reverse=True)

            # ── Budget & Category Filtering per Intent Type ──────────────────────
            if not is_vague:
                # DEFINITE intent: only show products at or below budget
                if budget == float('inf') or budget <= 0:
                    filtered = valid_results
                    budget_message = None
                else:
                    in_budget = [r for r in valid_results if (r.get('price_inr') or 0) <= budget]
                    filtered = in_budget[:20]  # Limit UI weight for definite searches
                    if not filtered:
                        cheapest = min(r.get('price_inr') or 0 for r in valid_results)
                        budget_message = (
                            f"Budget matrix breach detected. "
                            f"Cheapest node found at \u20b9{cheapest:,.0f} — "
                            f"exceeds your \u20b9{budget:,.0f} allocation by \u20b9{cheapest - budget:,.0f}. "
                            f"No products exist within this budget envelope. "
                            f"Consider expanding your allocation or narrowing the objective."
                        )
                    else:
                        budget_message = None
                
                # Tags are already assigned by normalize_scores in evaluator.py
                pass
            else:
                # INDEFINITE/SCENARIO intent: Priority is variety (one best per type)
                # We show the top 10 categories regardless of individual item price
                # but track the total for an advisory message.
                dedup = []
                seen_tq = set()
                total_cost = 0.0
                
                # results is already sorted by score
                for r in valid_results:
                    tq = r.get('target_query', 'General')
                    if tq not in seen_tq:
                        # Tag it as Scenario Result
                        r['tags'] = ['Top Search Products']
                        dedup.append(r)
                        seen_tq.add(tq)
                        total_cost += (r.get('price_inr') or 0)
                    if len(dedup) >= 10: break
                
                filtered = dedup
                
                # Advisory budget message
                if budget != float('inf') and budget > 0 and total_cost > budget:
                    budget_message = (
                        f"Target budget: \u20b9{budget:,.0f}. "
                        f"Curated 10-item Gear List totals \u20b9{total_cost:,.0f} — "
                        f"\u20b9{total_cost - budget:,.0f} above allocation. "
                        f"Prioritizing essential quality for your scenario."
                    )
                else:
                    budget_message = None

            # ── Categorization & Final Payloads ────────────────────────────────
            def categorize_results(results_list, vague_intent):
                if not results_list: return {}
                
                if vague_intent:
                    return {
                        "top_search_products": results_list,
                        "top_pick": results_list[0] if results_list else None
                    }
                else:
                    return {
                        "most_reliable": [r for r in results_list if "Most Reliable" in (r.get("tags") or [])][:5],
                        "best_value": [r for r in results_list if "Most Discounted" in (r.get("tags") or [])][:5],
                        "bestsellers": [r for r in results_list if "Best Seller" in (r.get("tags") or [])][:5],
                        "top_rated": [r for r in results_list if "Trending" in (r.get("tags") or [])][:5],
                        "top_pick": results_list[0] if results_list else None
                    }

            # ── Stream Results ─────────────────────────────────────────────────
            for r in filtered:
                finish_data = {
                    'id':           r['id'],
                    'status':       'complete',
                    'progress':     100,
                    'score':        r.get('score', 0.0),
                    'finalPrice':   r.get('price_inr', 0),
                    'platform':     r.get('platform', ''),
                    'image':        r.get('image', ''),
                    'external_link': r.get('external_link', '#'),
                    'tags':         r.get('tags', []),
                    'sentiment':    r.get('sentiment', 0.0),
                    'name':         r.get('name', ''),
                    'isBestseller': r.get('is_bestseller', False),
                    'reliability':  r.get('reliability_score', 0.8)
                }
                yield f"data: {json.dumps({'event': 'item_finish', 'data': finish_data})}\n\n"

            # ── Budget Message (after results) ────────────────────────────────
            if budget_message:
                yield f"data: {json.dumps({'event': 'pivot', 'data': {'message': budget_message, 'suggestedRentals': []}})}\n\n"

            collections = categorize_results(filtered, is_vague)
            yield f"data: {json.dumps({'event': 'done', 'data': {'message': 'Search complete. Categorized nodes ready.', 'collections': collections}})}\n\n"


        
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            msg_error = json.dumps({'event': 'error', 'data': {'message': f"Internal ML Engine Error: {str(e) or 'NotImplementedError (Windows Loop Issue)'}"}})
            yield f"data: {msg_error}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

# In-memory weight store (per server session)
_global_weights = {
    "price": 0.10,
    "rating": 0.25,
    "sentiment": 0.35,
    "bestseller": 0.15,
    "sales": 0.15,
}

class TuneRequest(BaseModel):
    feedback: str = ""
    weights: dict = {}

@app.post("/api/tune_weights")
async def tune_weights(req: TuneRequest):
    """
    RLHF endpoint. Accepts:
    - Natural language feedback string (parsed by simple NLP rules)
    - Explicit weights dict from the Review UI sliders
    Returns the updated weights.
    """
    global _global_weights

    if req.weights:
        # Direct slider update - just save it
        _global_weights.update(req.weights)
        # Normalize to sum=1
        total = sum(_global_weights.values())
        if total > 0:
            _global_weights = {k: round(v / total, 3) for k, v in _global_weights.items()}
        # Update the scoring engine's live weights
        scoring_engine.update_weights(_global_weights)
        return {"weights": _global_weights, "message": "Weights updated."}

    if req.feedback:
        fb = req.feedback.lower()
        w: dict[str, float] = dict(_global_weights)

        # Simple NLP rules for RLHF intent
        if any(x in fb for x in ["don't care about price", "no budget limit", "price doesn't matter", "ignore price"]):
            w["price"] = max(0.0, w["price"] - 0.08)
            w["rating"] += 0.04
            w["sentiment"] += 0.04
        if any(x in fb for x in ["best rated", "highest rating", "top rated", "quality"]):
            w["rating"] += 0.08
            w["price"] = max(0.0, w["price"] - 0.04)
            w["sales"] = max(0.0, w["sales"] - 0.04)
        if any(x in fb for x in ["cheapest", "budget", "affordable", "low price", "cheap"]):
            w["price"] += 0.10
            w["rating"] = max(0.0, w["rating"] - 0.05)
            w["sentiment"] = max(0.0, w["sentiment"] - 0.05)
        if any(x in fb for x in ["bestseller", "popular", "best seller", "trending"]):
            w["bestseller"] += 0.08
            w["sales"] += 0.05
            w["price"] = max(0.0, w["price"] - 0.05)
            w["sentiment"] = max(0.0, w["sentiment"] - 0.08)
        if any(x in fb for x in ["reviews", "review", "sentiment", "customer feedback"]):
            w["sentiment"] += 0.08
            w["sales"] = max(0.0, w["sales"] - 0.04)
            w["price"] = max(0.0, w["price"] - 0.04)

        # Normalize
        total = float(sum(w.values()))
        if total > 0.0:
            w = {k: round(float(v) / total, 3) for k, v in w.items()}
        
        _global_weights = w
        scoring_engine.update_weights(_global_weights)
        return {"weights": _global_weights, "message": f"Feedback applied. Adjusted weights based on your preference."}

    return {"weights": _global_weights, "message": "No changes."}

@app.get("/api/weights")
async def get_weights():
    return {"weights": _global_weights}

@app.post("/api/clear_session")
async def clear_session():
    session_manager.clear()
    return {"message": "Session cleared."}

class RLFeedbackRequest(BaseModel):
    id: str
    name: str = ""
    finalPrice: float = 0
    platform: str = ""
    sentiment: float = 0
    score: float = 0
    tags: list = []

@app.post("/api/rl_feedback")
async def rl_feedback_endpoint(item: RLFeedbackRequest):
    global _global_weights
    w: dict[str, float] = dict(_global_weights)
    
    # Analyze item to adjust weights dynamically based on RL logic
    if item.finalPrice > 0 and item.finalPrice < 2000:
        w["price"] = w.get("price", 0.10) + 0.05
        w["rating"] = max(0.01, w.get("rating", 0.25) - 0.02)
    elif item.finalPrice > 10000:
        w["price"] = max(0.01, w.get("price", 0.10) - 0.05)
        w["bestseller"] = w.get("bestseller", 0.15) + 0.05
        
    if item.sentiment > 80:
        w["sentiment"] = w.get("sentiment", 0.35) + 0.05
    elif item.sentiment < 40 and item.sentiment > 0:
        w["sentiment"] = max(0.01, w.get("sentiment", 0.35) - 0.05)
        
    if "Best Seller" in item.tags or "Most Monthly Sales" in item.tags:
        w["sales"] = w.get("sales", 0.15) + 0.05
        
    if "Most Reliable" in item.tags:
        w["rating"] = w.get("rating", 0.25) + 0.05

    # Normalize weights
    total = float(sum(w.values()))
    if total > 0.0:
        _global_weights = {k: round(float(v) / total, 3) for k, v in w.items()}
    
    scoring_engine.update_weights(_global_weights)
    logger.info(f"RL Updated Weights after interaction with {item.name}: {_global_weights}")
    
    return {"message": "RL Feedback Processed", "weights": _global_weights}

class WishlistRequest(BaseModel):
    wishlist: list[dict]

@app.post("/api/wishlist_suggestions")
async def wishlist_suggestions_endpoint(req: WishlistRequest):
    """Generates 1 complementary or upgrade item mapped to each item in the user's wishlist."""
    if not req.wishlist:
        return {"items": []}
        
    names = [w.get("name", "") for w in req.wishlist if w.get("name")]
    if not names:
        return {"items": []}
        
    scored = []
    seen_links = set()
    for w_name in names:
        logger.info(f"Generating 1 upgrade suggestion for wishlist item: {w_name}")
        new_item_nodes = intent_parser.extrapolate_checklist(f"Upgrade or complementary item for {w_name}", exclude_items=names, num_items=1)
        
        for node in new_item_nodes:
            scraped = await scrape_single_item(node["name"])
            if scraped:
                link = scraped.get('link', '#')
                if link != "#":
                    if link in seen_links:
                        continue
                    seen_links.add(link)
                
                sd = scoring_engine.calculate_raw_score(scraped, node["essentiality"])
                unique_id = f"wish_{abs(hash(link)) % 100000}_{random.randint(1000, 99999)}"
                scored.append({
                    "id": unique_id,
                    "name": scraped.get("title", node["name"]),
                    "score": round(sd["raw_score"] * 100, 1),
                    "finalPrice": sd["price_inr"],
                    "platform": sd["platform"],
                    "link": link,
                    "image": scraped.get("image", ""),
                    "tags": ["Wishlist Suggestion", "Complementary"],
                    "wait_to_buy": sd.get("wait_to_buy", False),
                    "coupon_applied": sd.get("coupon_applied", None),
                    "reddit_sentiment": sd.get("reddit_sentiment", None),
                    "suggested_for": w_name
                })
            
    return {"items": scored}

# ─── Chat / Tuning Assistant ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    items: list = []   # current product cards on client: [{id, name, finalPrice, ...}]
    wishlist: list = [] # user's persistent wishlist items

async def scrape_single_item(item_name: str) -> dict | None:
    """Launches a fresh Playwright context to scrape one replacement item."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                viewport={"width": 1440, "height": 900},
                locale="en-IN",
                timezone_id="Asia/Kolkata",
            )
            result = await scraper_service.scrape_item(item_name, context=context, log_cb=lambda m: None)
            await context.close()
            await browser.close()
            return result
    except Exception as e:
        logger.error(f"scrape_single_item error: {e}")
        return None

def _find_item_to_remove(msg: str, items: list) -> dict | None:
    """Fuzzy-find which item the user is referring to from the message."""
    msg_lower = msg.lower()
    best = None
    best_score = 0
    for item in items:
        name_words = item.get("name", "").lower().split()
        hits = sum(1 for w in name_words if w in msg_lower and len(w) > 3)
        if hits > best_score:
            best_score = hits
            best = item
    return best if best_score > 0 else None

def _parse_add_item(msg: str) -> str | None:
    """Extract what the user wants to add/replace with."""
    patterns = [
        r"(?:add|include|suggest|show me|replace.*?with|instead add|also get)\s+(.+?)(?:\s|$|,|\.)",
        r"(?:get|find|search for)\s+(.+?)(?:\s|$|,|\.)",
    ]
    for pattern in patterns:
        m = re.search(pattern, msg, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # Filter out noise words
            noise = {"me", "a", "an", "the", "some", "good", "best", "one"}
            words = [w for w in candidate.split() if w.lower() not in noise]
            if words:
                return " ".join(words)
    return None

def _apply_weight_nlp(msg: str) -> dict | None:
    """Apply NLP rules to adjust weights. Returns updated weights or None if no match."""
    global _global_weights
    fb = msg.lower()
    w = dict(_global_weights)
    changed = False

    rules = [
        (["don't care about price", "price doesn't matter", "ignore price", "no budget", "unlimited budget",
          "cost doesn't matter", "forget the cost", "price is not important"],
         {"price": -0.10, "rating": +0.05, "sentiment": +0.05}),
        (["best rated", "highest rating", "top rated", "quality matters", "focus on quality", "rating is important"],
         {"rating": +0.10, "price": -0.05, "sales": -0.05}),
        (["cheapest", "budget friendly", "affordable", "low price", "cheap as possible", "save money", "cost effective"],
         {"price": +0.12, "rating": -0.06, "sentiment": -0.06}),
        (["bestseller", "popular", "best seller", "trending", "most popular"],
         {"bestseller": +0.10, "sales": +0.05, "price": -0.05, "sentiment": -0.10}),
        (["reviews", "customer feedback", "sentiment", "people say", "review score", "most reviewed"],
         {"sentiment": +0.10, "sales": -0.05, "price": -0.05}),
        (["sales", "monthly sales", "most sold", "top selling", "high sales", "selling fast"],
         {"sales": +0.10, "sentiment": -0.05, "price": -0.05}),
    ]

    for triggers, deltas in rules:
        if any(t in fb for t in triggers):
            for key, delta in deltas.items():
                w[key] = max(0.0, w.get(key, 0) + delta)
            changed = True

    if not changed:
        return None

    total = sum(w.values())
    if total > 0:
        w = {k: round(v / total, 3) for k, v in w.items()}

    _global_weights = w
    scoring_engine.update_weights(_global_weights)
    return w

_REMOVE_TRIGGERS = ["remove", "already have", "i have", "don't need", "skip", "exclude", "take out", "delete", "drop"]
_REPLACE_TRIGGERS = ["replace", "swap", "instead", "substitute", "change to"]
_ADD_TRIGGERS = ["add", "also include", "include", "show me", "suggest", "find me", "also get"]

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Natural language chatbot endpoint for session-based tuning.
    Handles weight adjustments, item removal, replacement, and addition.
    """
    msg = req.message.strip()
    msg_lower = msg.lower()
    items = req.items

    # ── 1. Weight Tuning ──────────────────────────────────────────────────────
    new_weights = _apply_weight_nlp(msg)
    if new_weights:
        dominant = max(new_weights, key=new_weights.get)
        label_map = {"price": "price efficiency", "rating": "star rating", "sentiment": "customer sentiment",
                     "bestseller": "bestseller status", "sales": "sales volume"}
        dominant_label = label_map.get(dominant, dominant)
        return {
            "action": "weights_updated",
            "weights": new_weights,
            "message": (
                f"⚙️ Got it! I've tuned the ranking weights based on your preference. "
                f"**{dominant_label.title()}** is now the strongest factor in scoring. "
                f"Hit **Initialize Matrix** to re-rank with these new parameters."
            )
        }

    # ── 2. Ranking Intent (Show me top 10...) ────────────────────────────────
    is_ranking = any(t in msg_lower for t in ["top", "rank", "list", "best 10", "show me 10"])
    if is_ranking:
        count_match = re.search(r"top\s+(\d+)", msg_lower)
        count = int(count_match.group(1)) if count_match else 10
        
        # Try to find what they want to rank
        subject = _parse_add_item(msg) or (items[0]["name"] if items else None)
        if subject:
            logger.info(f"🏆 Ranking request for '{subject}' (Top {count})")
            # For ranking, we just trigger a fresh deep scrape for that item
            scraped = await scraper_service.scrape_items(subject)
            if scraped:
                scored = []
                for data in scraped[:count]:
                    sd = scoring_engine.calculate_raw_score(data, 1.0)
                    scored.append({
                        "id": f"rank_{abs(hash(data['link'])) % 100000}",
                        "name": data["title"],
                        "score": round(sd["raw_score"] * 100, 1),
                        "finalPrice": sd["price_inr"],
                        "platform": sd["platform"],
                        "external_link": data["link"],
                        "image": data.get("image", ""),
                        "tags": ["Best Rated"] if random.random() > 0.5 else ["Top Match"]
                    })
                session_manager.save_products(scored)
                ranked_list = "\n".join([f"{i+1}. **{s['name'][:40]}...** — ₹{s['finalPrice']:,.0f}" for i, s in enumerate(scored)])
                return {
                    "action": "add_bulk", # New client action to add multiples
                    "items": scored,
                    "message": f"🏆 Here are the top {len(scored)} rankings for **{subject}**:\n\n{ranked_list}\n\n*Nodes integrated into your dashboard.*"
                }

    # ── 3. Generative Expansion (Load More / Explore Further) ────────────────
    is_more = any(t in msg_lower for t in ["more", "further", "load more", "what else", "extended", "explore beyond"])
    is_system_explore = msg.startswith("[SYSTEM] EXTRAPOLATE_MORE")
    
    if is_more or is_system_explore:
        query_context = None
        current_names = [i.get("name", "").lower() for i in items if i.get("name")]
        
        if is_system_explore:
            # Extract prompt context
            extracted_ctx = msg.replace("[SYSTEM] EXTRAPOLATE_MORE:", "").strip()
            # if user typed something generic, fall back to first item name
            query_context = extracted_ctx if extracted_ctx and extracted_ctx != "[SYSTEM] EXTRAPOLATE_MORE" else (current_names[0] if current_names else "generic items")
        else:
            m = re.search(r"relevant to (.*?)[\.\?] I want to", msg)
            if m: query_context = m.group(1).strip()
            if not query_context:
                query_context = _parse_add_item(msg) or (items[0]["name"] if items else None)

        if not query_context:
            return {"action": "message", "message": "🤔 I'm not sure what to explore for."}
            
        logger.info(f"🧠 Generative Expansion (Explore Further) for: '{query_context}'")
        
        new_item_nodes = intent_parser.extrapolate_checklist(query_context, exclude_items=current_names, num_items=5)
        
        if not new_item_nodes:
            return {"action": "message", "message": f"🏆 You've seen all high-fidelity nodes for **{query_context}**."}

        scored = []
        for node in new_item_nodes:
            scraped = await scrape_single_item(node["name"])
            if scraped:
                # Calculate scores
                sd = scoring_engine.calculate_raw_score(scraped, node["essentiality"])
                scored.append({
                    "id": f"gen_{abs(hash(scraped.get('link', node['name']))) % 100000}",
                    "name": scraped.get("title", node["name"]),
                    "score": round(sd["raw_score"] * 100, 1),
                    "finalPrice": sd["price_inr"],
                    "platform": sd["platform"],
                    "external_link": scraped.get("link", "#"),
                    "image": scraped.get("image", ""),
                    "tags": ["New Discovery"],
                    # Novelty Upgrades
                    "wait_to_buy": sd.get("wait_to_buy", False),
                    "coupon_applied": sd.get("coupon_applied", None),
                    "reddit_sentiment": sd.get("reddit_sentiment", None)
                })
        
        if not scored:
            return {"action": "message", "message": f"⚠️ Extrapolated new nodes but web extraction failed."}

        session_manager.save_products(scored)
        ranked_list = "\n".join([f"✨ **{s['name'][:40]}...** — ₹{s['finalPrice']:,.0f}" for s in scored])
        return {
            "action": "add_bulk",
            "items": scored,
            "message": f"🚀 Extrapolated {len(scored)} new dimensions for **{query_context}**:\n\n{ranked_list}"
        }

    # ── 4. Item Removal ───────────────────────────────────────────────────────
    is_remove = any(t in msg_lower for t in _REMOVE_TRIGGERS)
    is_replace = any(t in msg_lower for t in _REPLACE_TRIGGERS)
    is_add = any(t in msg_lower for t in _ADD_TRIGGERS)

    if is_remove and not is_replace:
        target = _find_item_to_remove(msg, items)
        if target:
            return {
                "action": "remove",
                "remove_id": target["id"],
                "message": f"🗑️ Removed **{target['name']}** from your list."
            }
        return {"action": "message", "message": "🤔 I couldn't figure out which item to remove."}

    # ── 5. Item Replacement ───────────────────────────────────────────────────
    if is_replace or (is_remove and is_add):
        target = _find_item_to_remove(msg, items)
        new_item_name = _parse_add_item(msg)
        if not new_item_name:
            return {"action": "message", "message": "🤔 What should I replace it with?"}

        scrape_result = await scrape_single_item(new_item_name)
        if not scrape_result:
            return {"action": "message", "message": f"⚠️ Couldn't scrape **{new_item_name}**."}

        score_data = scoring_engine.calculate_raw_score(scrape_result, 0.8)
        new_item = {
            "id": f"chat_{abs(hash(new_item_name)) % 90000}",
            "name": new_item_name,
            "status": "complete", "progress": 100,
            "score": round(score_data["raw_score"] * 100, 1),
            "finalPrice": scrape_result.get("price_inr", 0),
            "platform": scrape_result.get("platform", "Amazon India"),
            "image": scrape_result.get("image", ""),
            "external_link": scrape_result.get("link", "#"),
        }
        return {
            "action": "replace",
            "remove_id": target["id"] if target else None,
            "new_item": new_item,
            "message": f"🔄 Replaced with **{new_item_name}**!"
        }

    # ── 6. Item Addition ──────────────────────────────────────────────────────
    if is_add:
        new_item_name = _parse_add_item(msg)
        if not new_item_name:
            return {"action": "message", "message": "🤔 What item would you like to add?"}

        scrape_result = await scrape_single_item(new_item_name)
        if not scrape_result:
            return {"action": "message", "message": f"⚠️ Couldn't find **{new_item_name}**."}

        score_data = scoring_engine.calculate_raw_score(scrape_result, 0.7)
        new_item = {
            "id": f"chat_{abs(hash(new_item_name)) % 90000}",
            "name": new_item_name,
            "status": "complete", "progress": 100,
            "score": round(score_data["raw_score"] * 100, 1),
            "finalPrice": scrape_result.get("price_inr", 0),
            "platform": scrape_result.get("platform", "Amazon India"),
            "image": scrape_result.get("image", ""),
            "external_link": scrape_result.get("link", "#"),
        }
        return {
            "action": "add",
            "new_item": new_item,
            "message": f"✅ Added **{new_item_name}** — ₹{scrape_result.get('price_inr', 0):,.0f}"
        }

    # ── 7. Knowledge Domain Constraint ──────────────────────────────────────────
    # If it reached here, it's a general question. We search memory for known products.
    memory_items = items + req.wishlist
    target_memory = _find_item_to_remove(msg, memory_items)

    if target_memory:
        src = "Wishlist" if any(w.get("id") == target_memory["id"] for w in req.wishlist) else "Session"
        return {
            "action": "message",
            "message": (
                f"🧠 Context ({src}): I see **{target_memory['name']}** in your memory.\n"
                f"It is currently priced at ₹{target_memory.get('finalPrice', 0):,.0f} with an ML score of {target_memory.get('score', 0)}/100.\n"
                "I can manipulate this item if you ask me to replace or remove it!"
            )
        }

    # Default Restricted Fallback
    return {
        "action": "message",
        "message": (
            f"🔒 System Constraint: My domain knowledge is strictly restricted to your current **Session Items** and **Wishlist Memory**.\n\n"
            f"If you'd like to ask a general question, please formulate an operation like *'Add a gaming mouse'* so I can fetch external nodes into my memory."
        )
    }


if __name__ == "__main__":
    import uvicorn
    if sys.platform == 'win32':
        import uvicorn.loops.auto
        import uvicorn.loops.asyncio
        def _proactor_setup():
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        uvicorn.loops.auto.auto_setup = _proactor_setup
        uvicorn.loops.asyncio.asyncio_setup = _proactor_setup

    uvicorn.run("ml_engine.main:app", host="0.0.0.0", port=8000)
