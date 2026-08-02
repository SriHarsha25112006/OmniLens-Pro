import json
import os

# Create a small dataset of labeled queries
# Relevance (0 to 3)
# 3: Exact match, highly trusted, great rating, excellent product
# 2: Good match, minor flaw (e.g. fewer reviews, or slightly off-brand but good specs)
# 1: Barely relevant, or poor rating, or overpriced
# 0: Completely irrelevant or scam product

dataset = [
    {
        "query": "gaming mouse",
        "items": [
            {
                "title": "Logitech G502 Hero High Performance Wired Gaming Mouse",
                "price": 3500, "rating": 4.8, "review_count": 45000, "discount": 20, "is_bestseller": True,
                "sentiment_label": "positive", "relevance": 3
            },
            {
                "title": "Razer DeathAdder Essential Gaming Mouse",
                "price": 1500, "rating": 4.6, "review_count": 25000, "discount": 30, "is_bestseller": True,
                "sentiment_label": "positive", "relevance": 3
            },
            {
                "title": "Zebronics Zeb-Transformer-M Optical USB Gaming Mouse",
                "price": 350, "rating": 4.0, "review_count": 8000, "discount": 10, "is_bestseller": False,
                "sentiment_label": "mixed", "relevance": 1
            },
            {
                "title": "Generic Wireless Mouse 2.4Ghz",
                "price": 200, "rating": 3.2, "review_count": 50, "discount": 0, "is_bestseller": False,
                "sentiment_label": "negative", "relevance": 0
            }
        ]
    },
    {
        "query": "noise cancelling headphones",
        "items": [
            {
                "title": "Sony WH-1000XM5 Wireless Active Noise Cancelling Headphones",
                "price": 29990, "rating": 4.8, "review_count": 12000, "discount": 15, "is_bestseller": True,
                "sentiment_label": "positive", "relevance": 3
            },
            {
                "title": "Bose QuietComfort 45 Bluetooth Wireless Noise Cancelling Headphones",
                "price": 27900, "rating": 4.7, "review_count": 9500, "discount": 10, "is_bestseller": False,
                "sentiment_label": "positive", "relevance": 3
            },
            {
                "title": "JBL Tune 760NC, Over Ear Active Noise Cancelling Headphones",
                "price": 5499, "rating": 4.3, "review_count": 4200, "discount": 30, "is_bestseller": True,
                "sentiment_label": "positive", "relevance": 2
            },
            {
                "title": "boAt Rockerz 450 Bluetooth On Ear Headphones (No ANC)",
                "price": 1499, "rating": 4.1, "review_count": 125000, "discount": 50, "is_bestseller": True,
                "sentiment_label": "positive", "relevance": 0 # Not ANC, so irrelevant despite popularity
            },
            {
                "title": "Unknown Brand Super Bass Noise Canceling Headset",
                "price": 899, "rating": 3.5, "review_count": 25, "discount": 5, "is_bestseller": False,
                "sentiment_label": "negative", "relevance": 0
            }
        ]
    },
    {
        "query": "4k monitor for macbook",
        "items": [
            {
                "title": "LG 27 inch 4K-UHD (3840 x 2160) HDR 10 Monitor with USB Type-C",
                "price": 28500, "rating": 4.6, "review_count": 3500, "discount": 25, "is_bestseller": True,
                "sentiment_label": "positive", "relevance": 3
            },
            {
                "title": "Dell U2723QE 27 inch 4K Hub Monitor USB-C",
                "price": 42000, "rating": 4.7, "review_count": 1200, "discount": 15, "is_bestseller": False,
                "sentiment_label": "positive", "relevance": 3
            },
            {
                "title": "Samsung 27-Inch Odyssey G5 2K QHD Gaming Monitor",
                "price": 24000, "rating": 4.5, "review_count": 2800, "discount": 18, "is_bestseller": False,
                "sentiment_label": "positive", "relevance": 1 # Not 4K
            },
            {
                "title": "HP 24mh FHD Monitor - 1080p",
                "price": 11000, "rating": 4.4, "review_count": 15000, "discount": 10, "is_bestseller": True,
                "sentiment_label": "positive", "relevance": 0 # Not 4K at all
            }
        ]
    },
    {
        "query": "protein powder whey isolate",
        "items": [
            {
                "title": "Optimum Nutrition (ON) Gold Standard 100% Whey Protein Isolate",
                "price": 3200, "rating": 4.5, "review_count": 35000, "discount": 10, "is_bestseller": True,
                "sentiment_label": "positive", "relevance": 3
            },
            {
                "title": "MuscleBlaze Biozyme Performance Whey Protein",
                "price": 2300, "rating": 4.4, "review_count": 28000, "discount": 15, "is_bestseller": True,
                "sentiment_label": "positive", "relevance": 2 # Concentrate/Blend, not pure Isolate
            },
            {
                "title": "Isopure Low Carb, 100% Whey Protein Isolate Powder",
                "price": 4500, "rating": 4.6, "review_count": 18000, "discount": 5, "is_bestseller": False,
                "sentiment_label": "positive", "relevance": 3
            },
            {
                "title": "Fake Brand Muscle Mass Gainer",
                "price": 999, "rating": 2.8, "review_count": 150, "discount": 50, "is_bestseller": False,
                "sentiment_label": "negative", "relevance": 0
            }
        ]
    }
]

# Duplicate the dataset multiple times to simulate a larger training set 
# (LightGBM needs a decent number of samples, we'll perturb values slightly)
import random
random.seed(42)

expanded_dataset = []
for i in range(100):
    for group in dataset:
        new_group = {"query": group["query"], "items": []}
        for item in group["items"]:
            # Perturb slightly to avoid identical rows
            new_item = dict(item)
            new_item["price"] = int(item["price"] * random.uniform(0.9, 1.1))
            new_item["rating"] = min(5.0, round(item["rating"] + random.uniform(-0.2, 0.2), 1))
            new_item["review_count"] = int(item["review_count"] * random.uniform(0.8, 1.2))
            new_item["discount"] = min(100, max(0, int(item["discount"] + random.uniform(-5, 5))))
            new_group["items"].append(new_item)
        expanded_dataset.append(new_group)

os.makedirs('c:/Projects/OmniLens Pro/omnilens-ml/ml_engine/models', exist_ok=True)
with open('c:/Projects/OmniLens Pro/omnilens-ml/ml_engine/models/ltr_dataset.json', 'w') as f:
    json.dump(expanded_dataset, f, indent=2)

print("LTR Dataset generated successfully.")
