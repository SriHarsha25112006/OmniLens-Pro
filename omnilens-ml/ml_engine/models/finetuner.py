import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ==============================================================================
# OmniLens ML Fine-Tuning Pipeline (Demo)
# 
# This script demonstrates how an AI architect would fine-tune the BERT/RoBERTa
# sentiment encoder using explicit/implicit user feedback from the OmniLens UI.
# ==============================================================================

def prepare_dataset():
    """
    In production, this data would be pulled from the ClickHouse telemetry DB
    where: text = product review/description, label = User clicked/bought (1) or ignored (0).
    """
    mock_telemetry = {
         "text": [
             "This ski jacket is incredibly warm and waterproof. Best purchase ever.",
             "The zipper broke on day two. Terrible quality.",
             "Good budget option, fits well enough for a weekend trip.",
             "Way overpriced for what you get, the material feels cheap."
         ],
         "label": [1, 0, 1, 0] # 1: Positive/Bought, 0: Negative/Ignored
    }
    return Dataset.from_dict(mock_telemetry)

def finetune_sentiment_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", output_dir="./models/omnilens-sentiment-v1"):
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We load it with num_labels=2 for binary classification (Good for me vs. Bad for me)
    # ignoring the original 3 (Negative, Neutral, Positive) since we are specializing it for purchasing.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

    # Freeze base layers for faster transfer learning (only train the classification head)
    for param in model.base_model.parameters():
        param.requires_grad = False

    dataset = prepare_dataset()
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Split (mock split for demo)
    train_dataset = tokenized_dataset.shuffle(seed=42).select(range(3))
    eval_dataset = tokenized_dataset.select(range(3, 4))

    # Configure Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        # logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting Fine-Tuning process based on user telemetry...")
    # trainer.train() # Uncomment in production with real data
    
    print(f"Model saved to {output_dir}")
    # trainer.save_model(output_dir)

if __name__ == "__main__":
    finetune_sentiment_model()
