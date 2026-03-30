# HieraSpark Architecture
## Hierarchical Sparse Kernel Adapter with Rotary Spectral Gate

**Authors**: OmniLens-Pro Research | **Date**: March 2026  
**Project**: OmniLens Pro — Fine-Tuning Architecture Research  
**Base Model**: Qwen2-7B-Instruct (3584 hidden dim, 28 layers)

---

## 1. Motivation

Standard parameter-efficient fine-tuning (PEFT) methods like **LoRA** and **IA³** operate in the *weight space* — they add learned low-rank matrices to existing weight tensors. While effective, they share several limitations:

- **Single-layer bottlenecks**: Adapters like the CVPR Series inject into a single "middle" layer, ignoring the full depth of the model.
- **Always-on computation**: All adapter paths activate for every token, even when the input is trivial (e.g. padding tokens, highly repetitive text).
- **Spatial-only representations**: DWConv-based adapters (CVPR Series) work in the sequence dimension only, ignoring the spectral structure of hidden states along the channel dimension.
- **No cross-layer learning signal**: Shallow adapters have no knowledge of what deep adapters are learning.

**HieraSpark** addresses all four limitations with three novel sub-modules.

---

## 2. Architecture Overview

```
Input Hidden State x  (B, T, H)
         │
   ┌─────┴─────────────────────────────────────┐
   │             RotarySpectralGate (RSG)        │
   │                                             │
   │  ┌──────────────────────────────────────┐  │
   │  │  FFT(x, dim=-1) → X_f  (B, T, H/2+1)│  │
   │  │                                      │  │
   │  │  mask_r = sigmoid(learnable_r)       │  │
   │  │  mask_i = sigmoid(learnable_i)       │  │
   │  │                                      │  │
   │  │  X_masked = complex(X_f.r*mask_r,    │  │
   │  │                      X_f.i*mask_i)   │  │
   │  │                                      │  │
   │  │  x_spec = iFFT(X_masked)  (B, T, H) │  │
   │  └──────────────────────────────────────┘  │
   │             │ spectrally-modulated state    │
   │  ┌──────────▼───────────────────────────┐  │
   │  │  down_proj → LayerNorm → GELU        │  │
   │  │  up_proj (zero-init)                 │  │
   │  │  → rsg_residual  (B, T, H)           │  │
   │  └──────────────────────────────────────┘  │
   │             +                               │
   │  ┌──────────────────────────────────────┐  │
   │  │  SpectralKernelBank (SKB)            │  │
   │  │  ─────────────────────────────────── │  │
   │  │  energy = ||x||₂  per token          │  │
   │  │  gate_k = 1 if energy > thresh_k     │  │
   │  │          else 0   (hard threshold)   │  │
   │  │                                      │  │
   │  │  FFT(x) → freq_real (B, T, freq_bins)│  │
   │  │  scaled_k = freq_real · kernel_k     │  │
   │  │  activated = scaled * gate           │  │
   │  │  mod = tanh(blend(activated))        │  │
   │  │  → skb_mod  (B, T, 1)               │  │
   │  └──────────────────────────────────────┘  │
   └─────────────────────────────────────────────┘
         │
   output = x + rsg_residual + 0.1 * skb_mod * x
```

### 2.1 Injection Pattern

```
Layer 0  →  HieraSparkMLP + HieraSparkAttention  ← injected
Layer 1  →  (original, unchanged)
Layer 2  →  HieraSparkMLP + HieraSparkAttention  ← injected
Layer 3  →  (original, unchanged)
...
Layer 26 →  HieraSparkMLP + HieraSparkAttention  ← injected
Layer 27 →  (original, unchanged)
```

**14 out of 28 layers** receive HieraSpark adapters for Qwen2-7B (with `layer_interval=2`). This is **hierarchical** — unlike CVPR Series which injects only into the single middle layer (layer 14).

---

## 3. Mathematical Formulation

### 3.1 Rotary Spectral Gate (RSG)

Let **x** ∈ ℝ^(B×T×H) be the input hidden state.

**Step 1 — Frequency-domain masking:**
```
X_f = RFFT(x, dim=H)          ∈ ℂ^(B×T×(H/2+1))
M_r = σ(m_r),  M_i = σ(m_i)  ∈ ℝ^(H/2+1)   (learnable, init=0)
X̃ = Re(X_f) ⊙ M_r + i · Im(X_f) ⊙ M_i
x_spec = iRFFT(X̃, n=H)        ∈ ℝ^(B×T×H)
```

Since m_r and m_i are initialised to 0, σ(0) = 0.5, so the gate starts by keeping half the spectrum, with the residual starting at near-zero (ensured by the zero-init `up_proj`).

**Step 2 — Bottleneck projection:**
```
b = GELU(LN(W_down · x_spec))   ∈ ℝ^(B×T×D)   D = H/8
r_rsg = W_up · b                 ∈ ℝ^(B×T×H)   (W_up initialised to 0)
```

### 3.2 Sparse Kernel Bank (SKB)

Let K = number of kernels, F = frequency bins (H/2).

**Energy-based routing:**
```
e_t = ||x_t||₂               ∈ ℝ    (energy of token t)
g_k = 𝟙[e_t > τ_k]           ∈ {0,1}  (hard threshold gate)
```

**Frequency-domain kernel application:**
```
x_real = Re(RFFT(x, dim=H))[..., :F]    ∈ ℝ^(B×T×F)
s_k = x_real · W_k^⊤                    ∈ ℝ^(B×T)    (kernel k dot product)
a_k = s_k ⊙ g_k                          (sparse activation)
mod = tanh(W_blend · [a_1, ..., a_K])   ∈ ℝ^(B×T×1)
```

**Final output:**
```
y = x + r_rsg + 0.1 · mod ⊙ x
```

The 0.1 coefficient is a fixed stability scale — the spectral modulation is deliberately kept small to prevent the early-training instability common in cross-domain adapters.

### 3.3 Hierarchical Cross-Layer Distillation (HCLD)

During training, let S = {shallow RSG outputs} and D = {deep RSG outputs}.

```
L_HCLD = (λ / |pairs|) · Σ_{i} MSE(
    normalize(s_i / τ),
    normalize(sg(d_i) / τ)
)
```

Where:
- τ = temperature (default 2.0) — softens distribution before distance measurement
- λ = HCLD weight (default 0.05) — relative contribution to total loss
- sg(·) = stop-gradient — deep adapters are teachers; only shallow adapters learn from it

**Total training loss:**
```
L_total = L_DPO + L_HCLD
```

---

## 4. Comparison Table

| Property                    | LoRA         | IA³          | CVPR Series       | **HieraSpark**        |
|-----------------------------|--------------|--------------|-------------------|-----------------------|
| Mechanism                   | Low-rank ΔW  | Scale vector | DWConv bottleneck | **Spectral gate**     |
| Injection site              | Weights      | Activations  | 1 middle layer    | **All even layers**   |
| Routing                     | Always-on    | Always-on    | Always-on         | **Threshold-sparse**  |
| Sequence length restriction | None         | None         | Fixed conv kernel | **None (FFT-based)**  |
| Cross-layer awareness       | None         | None         | None              | **HCLD distillation** |
| Training objective          | Task loss    | Task loss    | DPO               | **DPO + HCLD**        |
| Zero-init guarantee         | ✅            | ✅            | ✅                | ✅                    |
| Inference overhead          | Low          | Minimal      | Low               | **Low** (sparse gate) |
| Novel in literature         | ❌ (2021)    | ❌ (2022)    | ❌                | ✅                     |

---

## 5. Hyperparameter Guide

| Hyperparameter   | Default | Notes                                                      |
|------------------|---------|------------------------------------------------------------|
| `layer_interval` | 2       | Inject into every N-th layer. Higher = fewer adapters      |
| `n_kernels`      | 4       | Number of kernels in SpectralKernelBank. 4–8 recommended   |
| `freq_ratio`     | 0.5     | Fraction of frequency bins used in SKB. 0.25–0.75          |
| `threshold_init` | 0.1     | Initial energy threshold per kernel. Lower = denser routing |
| `hcld_weight`    | 0.05    | HCLD loss weight. 0.01–0.1 range                           |
| `temperature`    | 2.0     | HCLD distillation temperature. 1.0–4.0                     |
| LoRA `r`         | 8       | Rank. 4–16 for 15GB VRAM targets                           |
| LoRA `alpha`     | 16      | Alpha = 2× rank is a good baseline                         |
| `learning_rate`  | 4e-5    | Slightly lower than CVPR Series for spectral stability      |
| `warmup_ratio`   | 0.05    | Cosine warm-up for spectral mask stability                  |

---

## 6. File Structure

```
omnilens-ml/
└── ml_engine/
    └── models/
        ├── hiraspark_adapter.py    ← Core architecture (importable module)
        ├── hiraspark_finetune.py   ← Colab-ready DPO training script
        └── finetuner.py            ← Updated with HieraSpark demo path
```

---

## 7. Integration with OmniLens Pro

HieraSpark is designed as a **plug-in fine-tuning architecture** for any LLM used within OmniLens Pro. Currently it targets **Qwen2-7B** for the Git Pilot fine-tuning task, but the architecture is general — the `inject_hiraspark()` function works on any HuggingFace model with a `.model.layers` attribute.

Future integration paths:
- **Flan-T5 (Tier 3 generation)**: Apply `RotarySpectralGate` as micro-adapters on the decoder's cross-attention outputs for better query-to-product mapping.
- **Sentiment RoBERTa**: Replace FFN bottlenecks with RSG spectral gates to improve product review understanding without full fine-tuning.
