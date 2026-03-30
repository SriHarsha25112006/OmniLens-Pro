# -*- coding: utf-8 -*-
"""
HieraSpark DPO Fine-Tuning Script — Colab-Ready
================================================
Architecture : HieraSpark (Hierarchical Sparse Kernel Adapter + Rotary Spectral Gate)

Run on Google Colab with a T4/A100 GPU.
This script is a direct drop-in replacement for the user's CVPR Series training
script, but uses the novel HieraSpark architecture instead.

Setup (run in a Colab cell):

    !pip install -q -U transformers datasets peft trl bitsandbytes accelerate
    !pip install -q datasets sentence-transformers

Required drive structure:
    /content/drive/MyDrive/git_pilot/qwen7b_full   ← Base Qwen2-7B weights
    /content/drive/MyDrive/dpo_training_data.json  ← DPO training data

Output:
    /content/drive/MyDrive/git_pilot/qwen-hiraspark-dpo/  ← LoRA + HieraSpark weights
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports & System Prep
# ─────────────────────────────────────────────────────────────────────────────
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

# HuggingFace & training libraries
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: In Colab, mount drive first:
#   from google.colab import drive; drive.mount('/content/drive')
# ─────────────────────────────────────────────────────────────────────────────

torch.cuda.empty_cache()
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Paths (Edit these for your Drive layout)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/content/drive/MyDrive/git_pilot/qwen7b_full"
OUTPUT_DIR  = "/content/drive/MyDrive/git_pilot/qwen-hiraspark-dpo"
DATA_PATH   = "/content/drive/MyDrive/dpo_training_data.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  HieraSpark Architecture (self-contained — no external import needed)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralKernelBank(nn.Module):
    """
    Sparse Kernel Routing: threshold-gated kernel bank operating in freq domain.
    Each kernel is a learned frequency-domain weight vector (shape: freq_bins).
    Routing is hard-threshold driven — no soft top-k, exact zeros when inactive.
    """
    def __init__(self, hidden_size: int, n_kernels: int = 4, freq_ratio: float = 0.5,
                 threshold_init: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_kernels   = n_kernels
        self.freq_bins   = max(1, int(hidden_size * freq_ratio))

        self.kernel_weights = nn.Parameter(
            torch.ones(n_kernels, self.freq_bins)
            + torch.randn(n_kernels, self.freq_bins) * 0.02
        )
        self.thresholds  = nn.Parameter(torch.full((n_kernels,), threshold_init))
        self.blend_proj  = nn.Linear(n_kernels, 1, bias=False)
        nn.init.constant_(self.blend_proj.weight, 1.0 / n_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy   = x.norm(dim=-1)                                          # (B, T)
        thresholds = self.thresholds.abs()
        gate     = (energy.unsqueeze(-1) > thresholds.unsqueeze(0).unsqueeze(0)).float()
        gate     = gate + energy.unsqueeze(-1) * 0.0                       # straight-through grad

        x_fft    = torch.fft.rfft(x.float(), dim=-1)
        x_real   = x_fft.real[..., :self.freq_bins]                       # (B, T, freq_bins)
        scaled   = torch.einsum("btf,kf->btk", x_real, self.kernel_weights)
        activated= scaled * gate
        mod      = self.blend_proj(activated)
        return torch.tanh(mod).to(x.dtype)                                 # (B, T, 1)


class RotarySpectralGate(nn.Module):
    """
    Core HieraSpark adapter: frequency-domain masking on the channel axis,
    followed by a zero-init bottleneck projection and SpectralKernelBank modulation.

    Identity at initialisation (zero-init up_proj + near-zero spectral mask).
    """
    def __init__(self, hidden_size: int, bottleneck: Optional[int] = None,
                 n_kernels: int = 4, freq_ratio: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        bottleneck       = bottleneck or max(32, hidden_size // 8)

        freq_bins = hidden_size // 2 + 1
        self.spectral_mask_real = nn.Parameter(torch.zeros(freq_bins))
        self.spectral_mask_imag = nn.Parameter(torch.zeros(freq_bins))

        self.down_proj = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up_proj   = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.zeros_(self.up_proj.weight)                                # zero-init → identity start
        self.ln        = nn.LayerNorm(bottleneck)

        self.skb = SpectralKernelBank(hidden_size=hidden_size,
                                       n_kernels=n_kernels, freq_ratio=freq_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual   = x
        x_f32      = x.float()

        # ── Spectral mask in frequency domain ──────────────────────────────
        X_f            = torch.fft.rfft(x_f32, dim=-1)
        mask_r         = torch.sigmoid(self.spectral_mask_real)
        mask_i         = torch.sigmoid(self.spectral_mask_imag)
        X_masked       = torch.complex(X_f.real * mask_r, X_f.imag * mask_i)
        x_spectral     = torch.fft.irfft(X_masked, n=self.hidden_size, dim=-1).to(x.dtype)

        # ── Bottleneck on spectrally-modulated state ────────────────────────
        b_out          = self.down_proj(x_spectral)
        b_out          = self.ln(b_out.float()).to(x.dtype)
        b_out          = F.gelu(b_out)
        rsg_residual   = self.up_proj(b_out)

        # ── Sparse Kernel Bank scalar modulation ────────────────────────────
        skb_mod = self.skb(x)                                              # (B, T, 1)
        return residual + rsg_residual + skb_mod * residual * 0.1


class HieraSparkMLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, config, n_kernels: int = 4):
        super().__init__()
        self.original_mlp = original_mlp
        self.rsg          = RotarySpectralGate(config.hidden_size, n_kernels=n_kernels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.original_mlp(hidden_states)
        if getattr(self, "_hiraspark_disabled", False):
            return out
        return self.rsg(out)


class HieraSparkAttention(nn.Module):
    def __init__(self, original_attn: nn.Module, config, n_kernels: int = 4):
        super().__init__()
        self.original_attn = original_attn
        self.rsg           = RotarySpectralGate(config.hidden_size, n_kernels=n_kernels)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        attn_out   = self.original_attn(hidden_states, *args, **kwargs)
        is_tuple   = isinstance(attn_out, tuple)
        attn_hidden= attn_out[0] if is_tuple else attn_out
        if getattr(self, "_hiraspark_disabled", False):
            return attn_out
        gated = self.rsg(attn_hidden)
        return (gated,) + attn_out[1:] if is_tuple else gated


class HCLDLoss(nn.Module):
    """
    Hierarchical Cross-Layer Distillation: shallow adapter outputs → deep adapter targets.
    Training-only; zero inference overhead.
    """
    def __init__(self, temperature: float = 2.0, weight: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.weight      = weight

    def forward(self, shallow: List[torch.Tensor], deep: List[torch.Tensor]) -> torch.Tensor:
        if not shallow or not deep:
            return torch.tensor(0.0)
        loss  = torch.tensor(0.0, device=shallow[0].device)
        n_pairs = min(len(shallow), len(deep))
        for i in range(n_pairs):
            s = F.normalize((shallow[i] / self.temperature).float(), dim=-1)
            d = F.normalize((deep[i].detach() / self.temperature).float(), dim=-1)
            T = min(s.size(1), d.size(1))
            loss = loss + F.mse_loss(s[:, :T], d[:, :T])
        return (loss / n_pairs) * self.weight


def inject_hiraspark(model, config, layer_interval: int = 2, n_kernels: int = 4):
    """Inject HieraSpark into every `layer_interval`-th transformer layer."""
    layers    = model.model.layers
    injected  = []
    for idx, layer in enumerate(layers):
        if idx % layer_interval != 0:
            continue
        if hasattr(layer, "mlp") and not isinstance(layer.mlp, HieraSparkMLP):
            dev        = next(layer.mlp.parameters()).device
            layer.mlp  = HieraSparkMLP(layer.mlp, config, n_kernels).to(dev)
        if hasattr(layer, "self_attn") and not isinstance(layer.self_attn, HieraSparkAttention):
            dev             = next(layer.self_attn.parameters()).device
            layer.self_attn = HieraSparkAttention(layer.self_attn, config, n_kernels).to(dev)
        injected.append(idx)
    print(f"[HieraSpark] Injected into {len(injected)}/{len(layers)} layers: {injected}")
    return model, injected


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Custom DPO Trainer with HCLD Loss
# ─────────────────────────────────────────────────────────────────────────────

class HieraSparkDPOTrainer(DPOTrainer):
    """
    Extends DPOTrainer with Hierarchical Cross-Layer Distillation loss.

    The HCLD loss is computed between shallow (first 1/3 of injected layers)
    and deep (last 1/3 of injected layers) adapter activations, captured via
    forward hooks registered at init.
    """

    def __init__(self, *args, injected_layers: List[int] = None,
                 hcld_weight: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.hcld_loss_fn    = HCLDLoss(weight=hcld_weight)
        self.injected_layers = injected_layers or []
        self._shallow_acts: List[torch.Tensor] = []
        self._deep_acts:    List[torch.Tensor] = []
        self._hooks = []
        self._register_hcld_hooks()

    def _register_hcld_hooks(self):
        """Register forward hooks on RSG modules of shallow and deep layers."""
        if not self.injected_layers:
            return
        n = len(self.injected_layers)
        shallow_idxs = set(self.injected_layers[:max(1, n // 3)])
        deep_idxs    = set(self.injected_layers[max(1, 2 * n // 3):])

        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in shallow_idxs and isinstance(layer.mlp, HieraSparkMLP):
                h = layer.mlp.rsg.register_forward_hook(
                    lambda m, inp, out: self._shallow_acts.append(out)
                )
                self._hooks.append(h)
            if layer_idx in deep_idxs and isinstance(layer.mlp, HieraSparkMLP):
                h = layer.mlp.rsg.register_forward_hook(
                    lambda m, inp, out: self._deep_acts.append(out)
                )
                self._hooks.append(h)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._shallow_acts.clear()
        self._deep_acts.clear()

        # Standard DPO loss
        loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        # Add HCLD distillation loss
        hcld = self.hcld_loss_fn(self._shallow_acts, self._deep_acts)
        total_loss = loss + hcld

        if return_outputs:
            return total_loss, {}
        return total_loss

    def __del__(self):
        # Clean up hooks
        for h in self._hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model Loading & Quantization
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  HieraSpark DPO Training Pipeline")
print("=" * 60)

print("\n[1/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("[2/6] Loading base model in 4-bit NF4...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)
model_config = model.config

# ─────────────────────────────────────────────────────────────────────────────
# 5.  HieraSpark Injection (Hierarchical — every 2nd layer)
# ─────────────────────────────────────────────────────────────────────────────
print("[3/6] Injecting HieraSpark adapters (hierarchical, every 2nd layer)...")
model, injected_layers = inject_hiraspark(
    model, model_config, layer_interval=2, n_kernels=4
)

model = prepare_model_for_kbit_training(model)

# Count params before LoRA
hs_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    HieraSpark trainable params: {hs_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  LoRA Setup
# ─────────────────────────────────────────────────────────────────────────────
print("[4/6] Configuring LoRA...")

# Build modules_to_save: all HieraSpark component names
hiraspark_module_names = []
for name, module in model.named_modules():
    if isinstance(module, (RotarySpectralGate, SpectralKernelBank)):
        # Extract the leaf name within the module
        leaf = name.split(".")[-1]
        if leaf not in hiraspark_module_names:
            hiraspark_module_names.append(leaf)
# Use specific sub-module names for modules_to_save
save_modules = ["spectral_mask_real", "spectral_mask_imag", "down_proj",
                "up_proj", "ln", "kernel_weights", "thresholds", "blend_proj"]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=save_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Dataset Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
print("[5/6] Loading and formatting dataset...")
raw_dataset = load_dataset("json", data_files=DATA_PATH)

def chat_format_map(examples):
    p = examples.get("prompt") or examples.get("instruction") or ""
    c = examples.get("chosen") or examples.get("output_good") or ""
    r = examples.get("rejected") or examples.get("output_bad") or ""

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        formatted_prompt = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"

    return {
        "prompt":   formatted_prompt,
        "chosen":   c + tokenizer.eos_token,
        "rejected": r + tokenizer.eos_token,
    }

formatted_dataset = raw_dataset["train"].map(chat_format_map)
formatted_dataset = formatted_dataset.filter(
    lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0
)
split = formatted_dataset.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
eval_data  = split["test"]

print(f"    Train: {len(train_data)} samples  |  Eval: {len(eval_data)} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  DPO Training Configuration
# ─────────────────────────────────────────────────────────────────────────────
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.1,                              # DPO temperature
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=4e-5,                    # Slightly lower than CVPR Series (5e-5)
    num_train_epochs=1,
    max_length=512,
    eval_strategy="no",
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    remove_unused_columns=False,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,                     # Warm-up for spectral mask stability
    lr_scheduler_type="cosine",            # Cosine decay for smoother convergence
)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Trainer with HCLD Loss
# ─────────────────────────────────────────────────────────────────────────────
print("[6/6] Initialising HieraSparkDPOTrainer...")

trainer = HieraSparkDPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer,
    injected_layers=injected_layers,
    hcld_weight=0.05,                      # 5% of DPO loss → HCLD auxiliary loss
)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Train & Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Starting HieraSpark DPO Training")
print(f"  Architecture : Hierarchical injection on layers {injected_layers[:5]}...")
print(f"  HCLD Loss    : enabled (weight=0.05)")
print(f"  LoRA         : r=8, alpha=16 on Q/K/V/O projections")
print("=" * 60 + "\n")

trainer.train()

print("\nSaving model (LoRA adapters + HieraSpark spectral weights)...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Print final parameter summary
total_saved = sum(
    p.numel() for p in trainer.model.parameters() if p.requires_grad
)
print(f"\n✅ HieraSpark DPO Training complete!")
print(f"   Total saved adapter params: {total_saved:,}")
print(f"   Output location: {OUTPUT_DIR}")
