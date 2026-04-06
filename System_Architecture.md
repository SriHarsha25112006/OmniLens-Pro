# OmniLens Pro v2: System Architecture

OmniLens Pro v2 is powered by **HieraSpark**, a novel Parameter-Efficient Fine-Tuning (PEFT) architecture designed for high-precision shopping intelligence. This document provides a structural breakdown of the platform's core components and their relationships.

## Hierarchical System Overview

The system is organized into three primary layers, moving from the low-level model foundation to the high-level user interface.

### 1. ML Engine: HieraSpark PEFT
The heart of OmniLens Pro is the **HieraSpark** engine, which adaptively modulates a pre-trained LLM (e.g., Qwen2-7B) using three key innovations:

*   **RotarySpectralGate (RSG)**:
    *   Applies a Real Fast Fourier Transform (RFFT) along the hidden-state channel dimension.
    *   Learned spectral masks (real and imaginary) modulate frequency components.
    *   Initializes as an identity transform, ensuring zero disruption to the pre-trained model at the start of fine-tuning.
*   **SpectralKernelBank (SKB)**:
    *   A sparse bank of learnable frequency-domain kernels.
    *   Uses **Threshold-Gated Routing**: kernels only activate for tokens above an energy threshold (e.g., high-entropy keywords vs. padding).
    *   Provides dynamic, input-adaptive modulation of the hidden states.
*   **Hierarchical Cross-Layer Distillation (HCLD)**:
    *   A training-only auxiliary loss that matches the output distribution of shallow adapters to deep adapters.
    *   Enforces "depth context" in shallow layers without adding inference overhead.

### 2. Application Logic Layer
This layer handles the specialized signals required for intelligent product discovery and ranking.

*   **Contextual Query Clarifier**: A Flan-T5 based module that corrects spelling errors and disambiguates vague user queries before they reach the search pipeline.
*   **Multi-Signal Scoring Engine**: Aggregates four distinct signals for each product:
    *   **Match**: Semantic relevance to the query.
    *   **Sentiment**: Analysis of user reviews and feedback.
    *   **Reliability**: Brand and seller trustworthiness.
    *   **Discount**: Value-for-money and current deal status.
*   **Explore Engine**: A stateless discovery mechanism that uses RLHF-tuned weights (`_global_weights`) to suggest new products based on user interactions.

### 3. Frontend Layer: Next.js
*   **Reactive UI**: High-performance, modern interface built with Next.js and TailwindCSS.
*   **Product Discovery Loop**: Captures RLHF feedback (clicks, likes, adds to cart) and communicates it back to the scoring engine to refine the user's personal recommendation profile.

---

## Architecture Visualization

The following diagram illustrates the structural hierarchy and component relationships described above.

![OmniLens Pro Architecture](/C:/Users/sriha/.gemini/antigravity/brain/e971ba08-1cd9-4cad-b243-6c1e1769b3d5/omnilens_pro_architecture_v2_1775472575197.png)

> [!NOTE]
> This architecture ensures that OmniLens Pro remains parameter-efficient (training <1% of total model params) while achieving state-of-the-art performance in intent classification and product extraction.

---

## File Structure Reference

| Component | Path |
| :--- | :--- |
| **HieraSpark Core** | [`hiraspark_adapter.py`](file:///c:/Projects/OmniLens%20Pro/omnilens-ml/ml_engine/models/hiraspark_adapter.py) |
| **Scoring Engine** | [`main.py`](file:///c:/Projects/OmniLens%20Pro/omnilens-ml/ml_engine/main.py) |
| **Frontend Root** | [`omnilens/`](file:///c:/Projects/OmniLens%20Pro/omnilens) |
| **Architecture Specs** | [`HIRASPARK_ARCHITECTURE.md`](file:///c:/Projects/OmniLens%20Pro/HIRASPARK_ARCHITECTURE.md) |
