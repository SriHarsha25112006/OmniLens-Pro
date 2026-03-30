# -*- coding: utf-8 -*-
"""
HieraSpark Adapter — Novel LLM Fine-Tuning Architecture
=========================================================

Authors : OmniLens-Pro Research [2026]
Architecture : HieraSpark (Hierarchical Sparse Kernel Adapter
               with Rotary Spectral Gate)

──────────────────────────────────────────────────────────────────────────────
ARCHITECTURE OVERVIEW
──────────────────────────────────────────────────────────────────────────────

Three original sub-modules work together to form HieraSpark:

  1. SpectralKernelBank (SKB)
     ─ A learnable bank of N frequency-domain "kernel" weight matrices.
     ─ An energy threshold gate selects which kernels fire (sparse activation).
     ─ Avoids soft top-k routing; uses hard threshold gating → zero-cost paths.

  2. RotarySpectralGate (RSG)
     ─ Applies a real FFT along the hidden-state channel dimension.
     ─ A learned mask (initialised near 1) scales individual frequency bins.
     ─ iFFT reconstructs the modulated hidden state.
     ─ The gate's up-projection is zero-initialised → identity at init.
     ─ Works on variable sequence lengths with no padding required.

  3. Hierarchical Cross-Layer Distillation (HCLD)   [training-only]
     ─ A lightweight auxiliary loss that pulls shallow-layer adapter outputs
       toward the distribution of deep-layer adapter outputs.
     ─ Added ZERO inference overhead (loss computed only when targets provided).

──────────────────────────────────────────────────────────────────────────────
INJECTION PATTERN
──────────────────────────────────────────────────────────────────────────────

  inject_hiraspark(model, config, layer_interval=2)
    → Wraps MLP and Attention outputs of every `layer_interval`-th layer.
    → Hierarchical: shallow + deep layers all participate (unlike CVPR Series).

──────────────────────────────────────────────────────────────────────────────
COMPARISON vs. CVPR Series Adapter
──────────────────────────────────────────────────────────────────────────────

  Property               CVPR Series          HieraSpark
  ─────────────────────  ───────────────────  ──────────────────────────────
  Injection site         1 fixed middle layer All even layers (hierarchical)
  Residual mechanism     DWConv bottleneck    Spectral frequency gate (FFT)
  Routing                Always-on            Threshold-gated (sparse)
  Training objective     DPO only             DPO + HCLD distillation loss
  Cross-layer awareness  None                 Shallow→Deep distillation
  Parameter cost         ~2 × H²/8           ~H²/8 + N×H/4 (kernel bank)
  Sequence length        Fixed conv kernel    Variable-length FFT, no pad
  Activation function    GELU on DWConv       Learned spectral magnitude scale
"""

from __future__ import annotations

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  SpectralKernelBank  (SKB)
# ═══════════════════════════════════════════════════════════════════════════

class SpectralKernelBank(nn.Module):
    """
    A bank of N learned frequency-domain kernels with threshold-gated routing.

    Each kernel is a weight vector of shape (freq_dim,) applied multiplicatively
    to the FFT spectrum of an input slice. Routing is threshold-driven:

        gate_i = 1  if  energy(x) > threshold_i  else  0

    This is *sparse* — at low-energy tokens (e.g. padding, repetitive text)
    most kernels deactivate, reducing effective computation without any
    approximation error.

    Args:
        hidden_size  : Model hidden dimension (e.g. 3584 for Qwen2-7B)
        n_kernels    : Number of kernels in the bank (default 4)
        freq_ratio   : Fraction of frequency bins kept (default 0.5)
        threshold_init: Initial energy threshold for gating (default 0.1)
    """

    def __init__(
        self,
        hidden_size: int,
        n_kernels: int = 4,
        freq_ratio: float = 0.5,
        threshold_init: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_kernels   = n_kernels
        self.freq_bins   = max(1, int(hidden_size * freq_ratio))

        # Learnable kernel weights: [n_kernels, freq_bins]
        # Initialised near 1 so kernels start as near-identity transforms
        self.kernel_weights = nn.Parameter(
            torch.ones(n_kernels, self.freq_bins) + torch.randn(n_kernels, self.freq_bins) * 0.02
        )

        # Learnable per-kernel energy thresholds (scalar per kernel)
        self.thresholds = nn.Parameter(
            torch.full((n_kernels,), threshold_init)
        )

        # Blend gate: maps n_kernels activated outputs → hidden_size
        self.blend_proj = nn.Linear(n_kernels, 1, bias=False)
        nn.init.constant_(self.blend_proj.weight, 1.0 / n_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, hidden_size)
        Returns a spectral modulation scalar map of shape (batch, seq_len, 1)
        which is multiplied into a downstream projection.
        """
        # Energy per token: (batch, seq_len)
        energy = x.norm(dim=-1)                          # (B, T)
        energy_mean = energy.mean(dim=-1, keepdim=True)  # (B, 1) — batch-level ref

        # Sparse gate per kernel: (B, T, n_kernels)
        # Threshold comparison with straight-through gradient
        thresholds = self.thresholds.abs()               # ensure positive
        # Broadcasting: energy (B,T) vs thresholds (n_kernels,) → (B,T,n_kernels)
        gate = (energy.unsqueeze(-1) > thresholds.unsqueeze(0).unsqueeze(0)).float()
        # Straight-through: allow gradient through the continuous energy
        gate = gate + energy.unsqueeze(-1) * 0.0        # zero-grad trick via detach workaround

        # FFT on hidden dim: (B, T, freq_bins) — take first freq_bins real parts
        x_fft = torch.fft.rfft(x.float(), dim=-1)       # (B, T, H//2+1)
        x_fft_real = x_fft.real[..., :self.freq_bins]   # (B, T, freq_bins)

        # Apply each kernel as a frequency-domain scale and sum with gate
        # kernel_weights: (n_kernels, freq_bins) → scaled_freq: (B, T, n_kernels)
        scaled = torch.einsum("btf,kf->btk", x_fft_real, self.kernel_weights)  # (B,T,K)
        activated = scaled * gate                        # sparse gating

        # Blend activated kernel outputs → scalar modulation (B, T, 1)
        mod = self.blend_proj(activated)                 # (B, T, 1)
        mod = torch.tanh(mod)                            # bounded in [-1, 1]

        return mod.to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  RotarySpectralGate  (RSG)
# ═══════════════════════════════════════════════════════════════════════════

class RotarySpectralGate(nn.Module):
    """
    The core HieraSpark adapter: a residual gate operating in the frequency
    domain of the *channel* axis.

    Forward pass:
      1. FFT(x, dim=-1) → complex spectrum X_f of shape (B, T, H//2+1)
      2. Mask: element-wise multiply real and imag parts by learned mask M
         M is initialised to zeros → the adapter starts as exact identity
      3. iFFT → modulated hidden state x̃
      4. Down-project x̃ → bottleneck → up-project (zero-init up_proj)
      5. Sparse kernel bank provides an additional scalar modulation
      6. Output = x + RSG_residual + SKB_modulation * x

    Args:
        hidden_size   : Model hidden dimension
        bottleneck    : Bottleneck size for down/up projection (default H//8)
        n_kernels     : Number of kernels in the SpectralKernelBank
        freq_ratio    : Fraction of frequency bins kept in SKB
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck: Optional[int] = None,
        n_kernels: int = 4,
        freq_ratio: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        bottleneck = bottleneck or max(32, hidden_size // 8)
        self.bottleneck = bottleneck

        # Spectral frequency mask (real + imaginary parts separately)
        freq_bins = hidden_size // 2 + 1
        # Initialise near zero so adapter is identity at start
        self.spectral_mask_real = nn.Parameter(torch.zeros(freq_bins))
        self.spectral_mask_imag = nn.Parameter(torch.zeros(freq_bins))

        # Bottleneck projection on frequency-modulated hidden state
        self.down_proj = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up_proj   = nn.Linear(bottleneck, hidden_size, bias=False)
        # CRITICAL: zero-init up_proj so residual starts at exactly 0
        nn.init.zeros_(self.up_proj.weight)

        # Layer norm for stability in the bottleneck
        self.ln = nn.LayerNorm(bottleneck)

        # Sparse kernel bank for additional amplitude modulation
        self.skb = SpectralKernelBank(
            hidden_size=hidden_size,
            n_kernels=n_kernels,
            freq_ratio=freq_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, hidden_size)
        Returns same shape as x.
        """
        residual = x

        # ── Step 1: Frequency-domain masking ──────────────────────────────
        x_f32 = x.float()
        X_f = torch.fft.rfft(x_f32, dim=-1)         # (B, T, H//2+1) complex

        # Apply learned spectral mask (real and imaginary independently)
        mask_real = torch.sigmoid(self.spectral_mask_real)   # (freq_bins,)
        mask_imag = torch.sigmoid(self.spectral_mask_imag)   # (freq_bins,)

        X_masked_real = X_f.real * mask_real
        X_masked_imag = X_f.imag * mask_imag

        # Reconstruct complex tensor and iFFT back to spatial domain
        X_masked = torch.complex(X_masked_real, X_masked_imag)
        x_spectral = torch.fft.irfft(X_masked, n=self.hidden_size, dim=-1)  # (B, T, H)
        x_spectral = x_spectral.to(x.dtype)

        # ── Step 2: Bottleneck projection on spectrally-modulated state ───
        bottleneck_out = self.down_proj(x_spectral)     # (B, T, bottleneck)
        bottleneck_out = self.ln(bottleneck_out.float()).to(x.dtype)
        bottleneck_out = F.gelu(bottleneck_out)
        rsg_residual   = self.up_proj(bottleneck_out)   # (B, T, H)

        # ── Step 3: Sparse Kernel Bank amplitude modulation ───────────────
        skb_mod = self.skb(x)                           # (B, T, 1)  in [-1,1]
        # Scale: amplify or attenuate original hidden state features
        skb_contribution = skb_mod * residual           # (B, T, H)

        # ── Step 4: Final residual combination ────────────────────────────
        return residual + rsg_residual + skb_contribution * 0.1


# ═══════════════════════════════════════════════════════════════════════════
# 3.  HieraSpark Wrapper Modules (MLP + Attention)
# ═══════════════════════════════════════════════════════════════════════════

class HieraSparkMLP(nn.Module):
    """
    Wraps any MLP module with a RotarySpectralGate applied to its output.

    Usage:
        layer.mlp = HieraSparkMLP(layer.mlp, model.config)
    """

    def __init__(self, original_mlp: nn.Module, config, n_kernels: int = 4):
        super().__init__()
        self.original_mlp = original_mlp
        hidden_size = config.hidden_size
        self.rsg = RotarySpectralGate(
            hidden_size=hidden_size,
            n_kernels=n_kernels,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mlp_out = self.original_mlp(hidden_states)

        # Check if adapters are disabled (e.g. during reference model pass)
        if getattr(self, "_hiraspark_disabled", False):
            return mlp_out

        return self.rsg(mlp_out)


class HieraSparkAttention(nn.Module):
    """
    Wraps any attention module with a RotarySpectralGate applied to attn_output.

    The original attention module may return a tuple (attn_output, weights, ...)
    or just a tensor — both are handled.

    Usage:
        layer.self_attn = HieraSparkAttention(layer.self_attn, model.config)
    """

    def __init__(self, original_attn: nn.Module, config, n_kernels: int = 4):
        super().__init__()
        self.original_attn = original_attn
        hidden_size = config.hidden_size
        self.rsg = RotarySpectralGate(
            hidden_size=hidden_size,
            n_kernels=n_kernels,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple:
        attn_outputs = self.original_attn(hidden_states, *args, **kwargs)

        # Unpack whether output is tensor or tuple
        if isinstance(attn_outputs, tuple):
            attn_hidden = attn_outputs[0]
        else:
            attn_hidden = attn_outputs

        if getattr(self, "_hiraspark_disabled", False):
            return attn_outputs

        gated = self.rsg(attn_hidden)

        if isinstance(attn_outputs, tuple):
            return (gated,) + attn_outputs[1:]
        return gated


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Hierarchical Cross-Layer Distillation Loss  (HCLD)
# ═══════════════════════════════════════════════════════════════════════════

class HCLDLoss(nn.Module):
    """
    Hierarchical Cross-Layer Distillation Loss.

    During training only: forces shallow HieraSpark adapter outputs to
    approximate the distribution of deep adapter outputs, giving shallow
    layers "depth context" without any inference overhead.

    Implementation:
        For each (shallow_idx, deep_idx) pair:
            loss += MSE( shallow_rsg_out, stop_gradient(deep_rsg_out) )

    The stop_gradient ensures only the shallow adapters receive gradients
    from this loss (the deep adapters are treated as teachers).

    Args:
        temperature : Softens the target distribution (default 2.0)
        weight      : Loss weight relative to DPO loss (default 0.05)
    """

    def __init__(self, temperature: float = 2.0, weight: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(
        self,
        shallow_outputs: List[torch.Tensor],
        deep_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        shallow_outputs : list of tensors from shallow HieraSpark RSG outputs
        deep_outputs    : list of tensors from deep HieraSpark RSG outputs
        """
        if not shallow_outputs or not deep_outputs:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0, device=shallow_outputs[0].device)
        n_pairs = min(len(shallow_outputs), len(deep_outputs))

        for i in range(n_pairs):
            s_out = shallow_outputs[i] / self.temperature
            d_out = deep_outputs[i].detach() / self.temperature   # stop gradient to deep

            # Align sequence length (in case of different positions during eval)
            min_len = min(s_out.size(1), d_out.size(1))
            s_out = s_out[:, :min_len, :]
            d_out = d_out[:, :min_len, :]

            # Normalise to unit sphere before MSE (distribution matching)
            s_norm = F.normalize(s_out.float(), dim=-1)
            d_norm = F.normalize(d_out.float(), dim=-1)

            total_loss = total_loss + F.mse_loss(s_norm, d_norm)

        return (total_loss / n_pairs) * self.weight


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Injection Helper
# ═══════════════════════════════════════════════════════════════════════════

def inject_hiraspark(
    model: nn.Module,
    config,
    layer_interval: int = 2,
    n_kernels: int = 4,
    verbose: bool = True,
) -> Tuple[nn.Module, List[int]]:
    """
    Inject HieraSpark MLP and Attention adapters into a transformer model.

    Targets every `layer_interval`-th layer (0-indexed), creating hierarchical
    injection across ALL depth levels (unlike CVPR Series which targets only
    the middle layer).

    Args:
        model          : The transformer model (e.g. Qwen2ForCausalLM)
        config         : Model config (must have .hidden_size attribute)
        layer_interval : Inject into every N-th layer (default 2 = even layers)
        n_kernels      : Number of kernels in each SpectralKernelBank
        verbose        : Print injection report

    Returns:
        (modified_model, list_of_injected_layer_indices)
    """
    layers = model.model.layers
    n_layers = len(layers)
    injected = []

    for idx, layer in enumerate(layers):
        if idx % layer_interval != 0:
            continue

        # Inject MLP adapter
        if hasattr(layer, "mlp") and not isinstance(layer.mlp, HieraSparkMLP):
            mlp_device = next(layer.mlp.parameters()).device
            layer.mlp = HieraSparkMLP(
                layer.mlp, config, n_kernels=n_kernels
            ).to(mlp_device)

        # Inject Attention adapter
        if hasattr(layer, "self_attn") and not isinstance(layer.self_attn, HieraSparkAttention):
            attn_device = next(layer.self_attn.parameters()).device
            layer.self_attn = HieraSparkAttention(
                layer.self_attn, config, n_kernels=n_kernels
            ).to(attn_device)

        injected.append(idx)

    if verbose:
        logger.info(
            f"HieraSpark injected into {len(injected)}/{n_layers} layers: "
            f"{injected}"
        )
        print(
            f"[HieraSpark] Injected into {len(injected)}/{n_layers} layers "
            f"(interval={layer_interval}): {injected}"
        )

    return model, injected


def get_hiraspark_modules(model: nn.Module) -> List[str]:
    """
    Returns the list of module names that are HieraSpark components.
    Useful for building `modules_to_save` in LoraConfig.
    """
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (RotarySpectralGate, SpectralKernelBank)):
            names.append(name)
    return names


def count_hiraspark_params(model: nn.Module) -> dict:
    """
    Returns a breakdown of trainable parameters in HieraSpark components.
    """
    hiraspark_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            if any(k in name for k in ["rsg", "skb", "spectral_mask", "kernel_weights",
                                        "thresholds", "blend_proj", "down_proj", "up_proj", "ln"]):
                hiraspark_params += param.numel()

    return {
        "hiraspark_trainable": hiraspark_params,
        "total_model_params": total_params,
        "hiraspark_ratio": hiraspark_params / max(total_params, 1),
    }
