"""
OmniLens Intent Taxonomy
========================
A curated knowledge base mapping scenario/product intents to their
canonical shopping component lists.

Two types of intent:
  SCENARIO  - User is describing a situation/goal. We suggest what they need (components).
  PRODUCT   - User is asking for a specific item. We find the best options.

The taxonomy also serves as high-confidence training data for the intent classifier.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  SCENARIO TAXONOMY
#  key: trigger keywords (any of these found → map to this scenario)
#  value: list of products to search for
# ──────────────────────────────────────────────────────────────────────────────
SCENARIO_TAXONOMY = {
    # ── TECH / COMPUTING ─────────────────────────────────────────────────────
    "gaming pc|build a pc|gaming computer|desktop gaming|gaming rig|gaming setup": [
        "Gaming PC Cabinet", "Intel Core i7 processor", "NVIDIA RTX 4070 GPU",
        "32GB DDR5 RAM", "1TB NVMe SSD", "Gaming Monitor 144Hz",
        "Mechanical Gaming Keyboard", "Gaming Mouse", "WiFi Router Gaming",
        "Gaming Headset", "UPS power backup", "Thermal Paste",
    ],
    "home office|work from home|wfh setup|home studio|remote work": [
        "Ergonomic Office Chair", "Standing Desk", "4K Monitor",
        "Webcam 1080p", "USB Microphone", "Noise Cancelling Headphones",
        "Laptop Stand", "Mechanical Keyboard", "Wireless Mouse",
        "Ring Light", "Cable Management", "External Hard Drive",
    ],
    "video editing|content creator|youtube setup|streaming setup|video production": [
        "Video Editing Laptop", "External SSD 2TB", "4K Camera",
        "Gimbal Stabilizer", "Video Editing Software", "Color Calibration Monitor",
        "Studio Microphone", "RGB Key Light", "Green Screen",
        "SD Card 256GB", "Lens Cleaning Kit", "Tripod Professional",
    ],
    "music production|home studio recording|dj setup|podcast setup": [
        "Audio Interface", "Studio Monitor Speakers", "MIDI Keyboard",
        "Condenser Microphone", "Headphones Studio", "Pop Filter",
        "Acoustic Foam Panels", "DAW Software", "MIDI Controller",
        "Microphone Stand", "XLR Cable", "Audio Mixer",
    ],
    "home theatre|home theater|theater setup|theatre setup|cinema setup": [
        "4K Projector", "120-inch Projector Screen", "7.1 Home Theater System",
        "AV Receiver 8K", "Universal Remote Control", "Acoustic Wall Panels",
        "Home Theater Seating", "HDMI 2.1 Cable 10m", "Popcorn Machine",
        "Smart Lighting Kit",
    ],
    "coding setup|developer setup|programmer setup|programming": [
        "Mechanical Keyboard Programmer", "4K Monitor 27 inch",
        "Ergonomic Mouse", "Laptop Stand", "USB-C Hub",
        "External SSD", "Noise Cancelling Headphones",
        "Ergonomic Chair", "Webcam", "Drawing Tablet",
    ],

    # ── SPORTS / FITNESS ─────────────────────────────────────────────────────
    "home gym|workout at home|fitness setup|gym equipment|exercise": [
        "Adjustable Dumbbell Set", "Resistance Bands", "Yoga Mat",
        "Pull-up Bar", "Kettlebell", "Jump Rope",
        "Gym Gloves", "Protein Shaker", "Foam Roller",
        "Fitness Band Tracker",
    ],
    "outdoor gym|outdoor workout|calisthenics": [
        "Outdoor Pull-up Bar", "Resistance Bands Heavy", "Workout Gloves",
        "Jump Rope Speed", "Knee Sleeves", "Sports Shoes Running",
        "Gym Bag", "Fitness Tracker", "Bluetooth Speaker Waterproof",
        "Sunscreen SPF 50",
    ],

    # ── TRAVEL / ADVENTURE ───────────────────────────────────────────────────
    "skiing|ski trip|snowboarding|ski resort": [
        "Ski Jacket", "Ski Pants", "Ski Boots",
        "Ski Helmet", "Ski Goggles", "Thermal Base Layer",
        "Ski Gloves", "Ski Socks", "Hand Warmers",
        "Ski Poles", "Ski Bag", "Sunscreen SPF 50 Snow",
    ],
    "camping|trekking trip|hiking trip|backpacking": [
        "Trekking Backpack 60L", "Ultralight Tent", "Sleeping Bag",
        "Trekking Poles", "Hiking Boots", "Headlamp",
        "Water Purifier Portable", "First Aid Kit", "Thermal Jacket",
        "Multi-tool Knife", "Hydration Bladder", "Trail Gaiters",
    ],
    "beach vacation|beach trip|sea holiday": [
        "Reef-safe Sunscreen", "Rash Guard", "Beach Sandals",
        "Underwater Camera", "Beach Bag Waterproof", "Snorkeling Kit",
        "Quick-dry Towel", "Portable Bluetooth Speaker", "Water Shoes",
        "Sunglasses UV400",
    ],
    "international travel|trip abroad|overseas travel|flight vacation": [
        "Travel Backpack Cabin", "Universal Travel Adapter", "Neck Pillow",
        "Noise Cancelling Headphones Travel", "Packing Cubes", "Travel Wallet RFID",
        "Portable Charger 20000mAh", "TSA Lock", "Compression Socks Flight",
        "Travel Umbrella Compact",
    ],

    # ── HOME / KITCHEN ───────────────────────────────────────────────────────
    "kitchen setup|new kitchen|cooking setup|baking setup": [
        "Induction Cooktop", "Non-stick Cookware Set", "Chef Knife Set",
        "Cutting Board", "Mixing Bowls", "Kitchen Scale Digital",
        "Instant Pot Pressure Cooker", "Stand Mixer", "Blender",
        "Storage Containers", "Microwave Oven", "Air Fryer",
    ],
    "smart home|home automation|iot home": [
        "Smart Speaker Amazon Echo", "Smart Bulb Set", "Smart Plug",
        "Smart Security Camera", "Robot Vacuum", "Smart Thermostat",
        "Smart Door Lock", "Wi-Fi Mesh Router", "Smart TV",
        "Smart Switch", "Video Doorbell", "Air Quality Monitor",
    ],
    "nursery|baby room|newborn|baby setup": [
        "Baby Crib", "Baby Monitor Video", "Baby Diaper Bag",
        "Baby Swing", "Breast Pump Electric", "Baby Carrier",
        "Baby Thermometer", "White Noise Machine", "Baby Wipe Warmer",
        "Nursing Pillow", "Baby Bottle Sterilizer", "Baby Gate",
    ],

    # ── PHOTOGRAPHY ──────────────────────────────────────────────────────────
    "photography|photo setup|camera kit|photoshoot": [
        "DSLR Camera", "Versatile Zoom Lens", "Camera Tripod",
        "External Flash Speedlight", "Camera Bag", "SD Card Fast",
        "Remote Shutter Release", "ND Filter Kit", "Reflector 5-in-1",
        "Lens Cleaning Kit",
    ],

    # ── STUDY / SCHOOL ───────────────────────────────────────────────────────
    "marriage|wedding|reception|engagement|marriage function": [
        "Designer Wedding Suit", "Bridal Lehanga", "Men's Formal Shoes",
        "Perfume Luxury Set", "Wedding Jewelry Set", "Clutch Bag for Women",
        "Saree Designer Wear", "Sherwani for Men", "Makeup Kit Professional",
        "Wedding Gift hamper", "Wrist Watch Premium",
    ],
    "home decor|renovation|living room setup|interior design": [
        "Canvas Wall Art", "Decorative Floor Vase", "Velvet Accent Chair",
        "Indoor Large Plants", "Smart RGB Floor Lamp", "Area Rug Patterned",
        "Decorative Throw Pillows", "Glass Coffee Table", "Floating Wall Shelves",
        "Curtains Blackout", "Aromatic Diffuser",
    ],
    "gardening|backyard setup|lawn care|garden": [
        "Gardening Tool Set", "Automatic Plant Waterer", "Lawn Mower Electric",
        "Garden Lights Solar", "Outdoor Bench", "Potting Soil 10kg",
        "Seed Starter Kit", "Watering Can", "Pruning Shears",
        "Bird Feeder", "Garden Kneeler",
    ],
    "fitness|workout|gym setup|fat loss|training": [
        "Adjustable Dumbbells", "Yoga Mat Non-slip", "Resistance Bands Set",
        "Pull Up Bar", "Kettlebell", "Jump Rope",
        "Gym Gloves", "Protein Shaker", "Fitness Tracker",
        "Workout Bench", "Foam Roller",
    ],
    "study setup|student setup|college|university|online classes": [
        "Lightweight Laptop", "Noise Cancelling Headphones", "Desk Lamp LED",
        "Ergonomic Chair Student", "Notebook Hardcover", "Tablet for Notes",
        "Stylus Pen", "Portable Hard Drive", "Webcam HD",
        "Power Bank",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
#  TRAINING DATASET  (intent_type: SCENARIO or PRODUCT)
#  Used to fine-tune / zero-shot a sentence classifier
# ──────────────────────────────────────────────────────────────────────────────
TRAINING_DATA = [
    # SCENARIO examples
    {"text": "I want to build a gaming PC", "label": "SCENARIO"},
    {"text": "I want to set up a gaming rig", "label": "SCENARIO"},
    {"text": "help me set up a home gym", "label": "SCENARIO"},
    {"text": "I'm planning a skiing trip to Manali", "label": "SCENARIO"},
    {"text": "setting up my home studio for music production", "label": "SCENARIO"},
    {"text": "I need things for my baby nursery", "label": "SCENARIO"},
    {"text": "going camping next month, what do I need?", "label": "SCENARIO"},
    {"text": "setting up a YouTube content creation setup", "label": "SCENARIO"},
    {"text": "I need a work from home setup", "label": "SCENARIO"},
    {"text": "planning a beach vacation", "label": "SCENARIO"},
    {"text": "I want to start cooking at home more", "label": "SCENARIO"},
    {"text": "setting up smart home devices", "label": "SCENARIO"},
    {"text": "I'm going to a ski resort and need gear", "label": "SCENARIO"},
    {"text": "preparing for a trekking trip in the Himalayas", "label": "SCENARIO"},
    {"text": "building a podcast setup at home", "label": "SCENARIO"},
    {"text": "setting up a programming workstation", "label": "SCENARIO"},
    {"text": "I want to start doing outdoor workouts", "label": "SCENARIO"},
    {"text": "preparing for international travel", "label": "SCENARIO"},
    {"text": "setting up a photography studio", "label": "SCENARIO"},
    {"text": "need items for my college dorm room", "label": "SCENARIO"},
    {"text": "I want to upgrade my home office", "label": "SCENARIO"},
    {"text": "what do I need for video editing", "label": "SCENARIO"},
    {"text": "setting up a streaming setup for gaming", "label": "SCENARIO"},
    {"text": "going on a backpacking trip", "label": "SCENARIO"},
    {"text": "I want to set up a kitchen for baking", "label": "SCENARIO"},
    {"text": "need gear for snowboarding", "label": "SCENARIO"},
    {"text": "help me put together a fitness corner at home", "label": "SCENARIO"},
    {"text": "I want to automate my entire house", "label": "SCENARIO"},
    {"text": "planning a road trip, what should I buy?", "label": "SCENARIO"},
    {"text": "building a DJ setup for events", "label": "SCENARIO"},

    # PRODUCT examples
    {"text": "iPhone 15 Pro Max", "label": "PRODUCT"},
    {"text": "Samsung Galaxy S24 Ultra", "label": "PRODUCT"},
    {"text": "Best gaming laptop under 80000", "label": "PRODUCT"},
    {"text": "noise cancelling headphones", "label": "PRODUCT"},
    {"text": "gaming mouse", "label": "PRODUCT"},
    {"text": "RTX 4090 GPU", "label": "PRODUCT"},
    {"text": "mechanical keyboard", "label": "PRODUCT"},
    {"text": "OLED television 65 inch", "label": "PRODUCT"},
    {"text": "air fryer", "label": "PRODUCT"},
    {"text": "wireless earbuds", "label": "PRODUCT"},
    {"text": "standing desk electric", "label": "PRODUCT"},
    {"text": "protein powder whey", "label": "PRODUCT"},
    {"text": "4K camera for photography", "label": "PRODUCT"},
    {"text": "gaming chair", "label": "PRODUCT"},
    {"text": "ultrawide monitor 34 inch", "label": "PRODUCT"},
    {"text": "running shoes Nike", "label": "PRODUCT"},
    {"text": "smart watch Apple Watch Series 9", "label": "PRODUCT"},
    {"text": "trekking backpack 60L", "label": "PRODUCT"},
    {"text": "wireless router WiFi 6", "label": "PRODUCT"},
    {"text": "Dyson vacuum cleaner", "label": "PRODUCT"},
    {"text": "cookware set non-stick", "label": "PRODUCT"},
    {"text": "adjustable dumbbell set", "label": "PRODUCT"},
    {"text": "ergonomic chair Herman Miller", "label": "PRODUCT"},
    {"text": "Sony WH-1000XM5", "label": "PRODUCT"},
    {"text": "iPad Pro 12.9", "label": "PRODUCT"},
    {"text": "Kindle Oasis", "label": "PRODUCT"},
    {"text": "graphics card", "label": "PRODUCT"},
    {"text": "SSD 1TB NVMe", "label": "PRODUCT"},
    {"text": "ski jacket", "label": "PRODUCT"},
    {"text": "yoga mat thick", "label": "PRODUCT"},
]
