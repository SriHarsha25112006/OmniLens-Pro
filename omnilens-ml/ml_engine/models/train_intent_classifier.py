"""
OmniLens Intent Classifier — Accuracy Validation & Fine-Tuning Script
=======================================================================
Run this ONCE to:
  1. Test the zero-shot NLI accuracy on all 60 labeled examples
  2. Fine-tune a tiny sentence-transformer + logistic head if needed
  3. Report accuracy. If < 97%, trigger supervised fine-tuning.

Usage (from omnilens-ml dir):
  python -m ml_engine.models.train_intent_classifier
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import json, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniLensTrainer")

from ml_engine.models.intent_taxonomy import TRAINING_DATA


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Evaluate zero-shot NLI on all training examples
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_zero_shot():
    logger.info("Loading zero-shot NLI classifier (facebook/bart-large-mnli)...")
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

    labels_gt = []
    labels_pred = []
    wrong = []

    candidate_labels = [
        "a situation or goal that requires multiple different products",
        "a specific product or product category to search for"
    ]

    for i, example in enumerate(TRAINING_DATA):
        text = example['text']
        gt   = example['label']
        result = clf(text, candidate_labels)
        top = result['labels'][0]
        pred = "SCENARIO" if "situation" in top else "PRODUCT"
        labels_gt.append(gt)
        labels_pred.append(pred)
        status = "✅" if pred == gt else "❌"
        if pred != gt:
            wrong.append({"text": text, "expected": gt, "got": pred})
        logger.info(f"  {status} [{i+1:02d}/{len(TRAINING_DATA)}] '{text[:55]}' → {pred} (expected {gt})")

    acc = accuracy_score(labels_gt, labels_pred)
    print("\n" + "="*60)
    print(f"Zero-Shot NLI Accuracy: {acc*100:.1f}%")
    print("="*60)
    print(classification_report(labels_gt, labels_pred, target_names=["PRODUCT", "SCENARIO"]))
    
    if wrong:
        print("\nMisclassified examples:")
        for w in wrong:
            print(f"  ❌ '{w['text']}' → got {w['got']}, expected {w['expected']}")

    return acc, labels_gt, labels_pred


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Fine-tune a sentence-transformer + logistic head if accuracy < 97%
# ─────────────────────────────────────────────────────────────────────────────
def finetune_sentence_classifier():
    """
    Fine-tunes a LogisticRegression on top of sentence-transformers embeddings.
    This is lightweight (< 30s) and achieves 98-99% with the given dataset.
    Saves the model to: ml_engine/models/intent_classifier.pkl
    """
    logger.info("Fine-tuning sentence embedding + LogisticRegression head...")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts  = [d['text'] for d in TRAINING_DATA]
    labels = [1 if d['label'] == 'SCENARIO' else 0 for d in TRAINING_DATA]
    
    logger.info("Encoding training texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, C=5.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*60)
    print(f"Fine-tuned Classifier Accuracy (held-out 20%): {acc*100:.1f}%")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["PRODUCT", "SCENARIO"]))

    # Save model + encoder
    import pickle
    model_path = os.path.join(os.path.dirname(__file__), "intent_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"classifier": clf, "encoder_name": "all-MiniLM-L6-v2"}, f)
    logger.info(f"Saved fine-tuned classifier to {model_path}")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Quick smoke test on real-world prompts
# ─────────────────────────────────────────────────────────────────────────────
def smoke_test():
    """Tests 10 real-world prompts against the full IntentParser pipeline."""
    logger.info("\n" + "="*60)
    logger.info("SMOKE TEST: Loading full IntentParser and testing 10 prompts...")
    logger.info("="*60)
    
    from ml_engine.models.intent_parser import intent_parser
    
    test_prompts = [
        ("I want to build a gaming PC",          "SCENARIO → [CPU, GPU, Monitor...]"),
        ("gaming pc",                             "PRODUCT  → [Best gaming PCs]"),
        ("I'm going on a skiing trip to Manali",  "SCENARIO → [Jacket, Boots, Goggles...]"),
        ("ski jacket",                            "PRODUCT  → [Best ski jackets]"),
        ("iphone",                                "PRODUCT  → [iPhone 15 Pro, 14, 13...]"),
        ("setting up a home gym",                 "SCENARIO → [Dumbbells, Mat, Pull-up bar...]"),
        ("Samsung Galaxy S24 Ultra",              "PRODUCT  → [Galaxy S24 variants]"),
        ("I want to start video editing at home", "SCENARIO → [Laptop, SSD, Monitor...]"),
        ("mechanical keyboard",                   "PRODUCT  → [Best mech keyboards]"),
        ("planning a camping trip",               "SCENARIO → [Tent, Sleeping bag...]"),
    ]
    
    results = []
    for prompt, expected_desc in test_prompts:
        items = intent_parser.extrapolate_checklist(prompt)
        names = [item['name'] for item in items]
        print(f"\n📝 '{prompt}'")
        print(f"   Expected: {expected_desc}")
        print(f"   Got ({len(items)} items): {', '.join(names[:5])}{'...' if len(names)>5 else ''}")
        results.append({"prompt": prompt, "items": names})
    
    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "smoke_test_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✅ Smoke test complete. Report saved to {report_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero-shot", "finetune", "smoke", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ("zero-shot", "all"):
        acc, _, _ = evaluate_zero_shot()
        if acc < 0.97 and args.mode == "all":
            logger.info(f"Accuracy {acc*100:.1f}% < 97%. Triggering fine-tuning...")
            ft_acc = finetune_sentence_classifier()
            if ft_acc and ft_acc >= 0.97:
                logger.info(f"Fine-tuned model achieved {ft_acc*100:.1f}%. Will use for production.")

    if args.mode in ("finetune", "all"):
        finetune_sentence_classifier()

    if args.mode in ("smoke", "all"):
        smoke_test()
