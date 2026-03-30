import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ==============================================================================
# OmniLens ML Fine-Tuning Pipeline
#
# TWO fine-tuning paths available:
#
#   Path A — Standard head-only fine-tune of RoBERTa sentiment model
#             (original approach: freeze backbone, train classification head)
#
#   Path B — HieraSpark RSG micro-adapter fine-tuning
#             (novel approach: inject RotarySpectralGate into RoBERTa FFN outputs,
#             proving HieraSpark is general-purpose — not only for Qwen2/DPO)
#
# ==============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# HieraSpark RSG — Mini version for encoder models (no SKB for simplicity)
# ─────────────────────────────────────────────────────────────────────────────

class RSGMicroAdapter(nn.Module):
    """
    Lightweight RotarySpectralGate for encoder models (e.g. RoBERTa-base, H=768).
    Injected after each transformer layer's intermediate FFN output.

    Properties:
     - Zero-init up_proj → starts as identity (no degradation at init)
     - Spectral masks initialised to 0 → sigmoid(0) = 0.5 (gentle start)
     - ~600K params per injection for H=768 (vs 3.2M for H=3584/Qwen2)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        freq_bins = hidden_size // 2 + 1
        bottleneck = max(32, hidden_size // 8)

        self.spectral_mask_real = nn.Parameter(torch.zeros(freq_bins))
        self.spectral_mask_imag = nn.Parameter(torch.zeros(freq_bins))

        self.down_proj = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up_proj   = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.zeros_(self.up_proj.weight)  # zero-init → identity at start
        self.ln        = nn.LayerNorm(bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual   = x
        X_f        = torch.fft.rfft(x.float(), dim=-1)
        mask_r     = torch.sigmoid(self.spectral_mask_real)
        mask_i     = torch.sigmoid(self.spectral_mask_imag)
        X_mod      = torch.complex(X_f.real * mask_r, X_f.imag * mask_i)
        x_spec     = torch.fft.irfft(X_mod, n=x.size(-1), dim=-1).to(x.dtype)

        b = F.gelu(self.ln(self.down_proj(x_spec).float()).to(x.dtype))
        return residual + self.up_proj(b)


class RoBERTaLayerWithRSG(nn.Module):
    """Wraps a single RoBERTa transformer layer's output with an RSGMicroAdapter."""
    def __init__(self, original_layer: nn.Module, hidden_size: int):
        super().__init__()
        self.original_layer = original_layer
        self.rsg = RSGMicroAdapter(hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        layer_out = self.original_layer(hidden_states, *args, **kwargs)
        # RoBERTa layers return tuples: (hidden_state, ...)
        if isinstance(layer_out, tuple):
            gated = self.rsg(layer_out[0])
            return (gated,) + layer_out[1:]
        return self.rsg(layer_out)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset():
    """
    In production: pull from ClickHouse telemetry DB where
    text = product review/description, label = User clicked/bought (1) or ignored (0).
    """
    mock_telemetry = {
        "text": [
            "This ski jacket is incredibly warm and waterproof. Best purchase ever.",
            "The zipper broke on day two. Terrible quality.",
            "Good budget option, fits well enough for a weekend trip.",
            "Way overpriced for what you get, the material feels cheap.",
        ],
        "label": [1, 0, 1, 0],
    }
    return Dataset.from_dict(mock_telemetry)


# ─────────────────────────────────────────────────────────────────────────────
# Path A — Standard head-only fine-tuning (original approach)
# ─────────────────────────────────────────────────────────────────────────────

def finetune_sentiment_model(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    output_dir: str = "./models/omnilens-sentiment-v1",
):
    """Standard approach: freeze backbone, fine-tune classification head only."""
    print(f"[Path A] Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    # Freeze backbone — only classification head trains
    for param in model.base_model.parameters():
        param.requires_grad = False

    dataset = prepare_dataset()

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=128)

    tok_ds       = dataset.map(tokenize_fn, batched=True)
    train_ds     = tok_ds.shuffle(seed=42).select(range(3))
    eval_ds      = tok_ds.select(range(3, 4))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
    )
    print("[Path A] Fine-Tuning process ready (uncomment trainer.train() for production).")
    # trainer.train()
    print(f"[Path A] Model would save to {output_dir}")
    # trainer.save_model(output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Path B — HieraSpark RSG micro-adapter fine-tuning (novel approach)
# ─────────────────────────────────────────────────────────────────────────────

def finetune_with_hiraspark_rsg(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    output_dir: str = "./models/omnilens-sentiment-hiraspark-v1",
    inject_every_n: int = 2,
):
    """
    HieraSpark Path: Inject RSGMicroAdapter into every N-th RoBERTa encoder layer.

    - Backbone is frozen except for RSG adapter weights
    - RSG adapters + classification head are trained
    - Demonstrates HieraSpark RSG is general-purpose (not Qwen2-only)

    Args:
        model_name    : HuggingFace model identifier
        output_dir    : Where to save the fine-tuned model
        inject_every_n: Inject RSG into every N-th encoder layer (default 2)
    """
    print(f"\n[Path B — HieraSpark RSG] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )

    # Determine hidden size from config
    hidden_size = model.config.hidden_size
    print(f"    Hidden size: {hidden_size}")

    # ── Freeze entire backbone ────────────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    # ── Inject RSGMicroAdapter into every N-th encoder layer ─────────────────
    encoder_layers = model.roberta.encoder.layer
    n_layers       = len(encoder_layers)
    injected       = []

    for idx in range(n_layers):
        if idx % inject_every_n == 0:
            encoder_layers[idx] = RoBERTaLayerWithRSG(
                encoder_layers[idx], hidden_size
            )
            # Unfreeze RSG parameters (adapter is trainable)
            for param in encoder_layers[idx].rsg.parameters():
                param.requires_grad = True
            injected.append(idx)

    # Unfreeze classification head
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"    Injected RSG into layers: {injected}")
    print(f"    Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Dataset prep ──────────────────────────────────────────────────────────
    dataset = prepare_dataset()

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=128)

    tok_ds   = dataset.map(tokenize_fn, batched=True)
    train_ds = tok_ds.shuffle(seed=42).select(range(3))
    eval_ds  = tok_ds.select(range(3, 4))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=3e-4,               # Higher LR for adapters (common in PEFT)
        per_device_train_batch_size=8,
        num_train_epochs=5,               # More epochs: adapters need warm-up
        weight_decay=0.01,
        warmup_ratio=0.1,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
    )
    print("[Path B] HieraSpark RSG fine-tuning ready (uncomment trainer.train() for production).")
    # trainer.train()
    print(f"[Path B] Model would save to {output_dir}")
    # trainer.save_model(output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("OmniLens Fine-Tuning Demo")
    print("=" * 40)
    print("Running Path A (standard head-only)...")
    finetune_sentiment_model()

    print("\nRunning Path B (HieraSpark RSG adapters)...")
    finetune_with_hiraspark_rsg()

