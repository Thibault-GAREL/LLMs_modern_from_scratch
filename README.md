# 🧬 modern-transformer

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-2.13-e92063.svg)
![pytest](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="assets/banner.svg" alt="modern-transformer, from the 2017 Transformer to the current open-weights defaults" width="820">
</p>

## 📝 Project Description

Nobody trains the Transformer of **Vaswani et al. (2017)** anymore. Every open-weights model released since **LLaMA** replaced its normalization, its positions, its attention and its feed-forward, one paper at a time. This library rebuilds those deviations in **PyTorch**, each one behind a config flag, so they can be switched on and off and compared.

It is an **ablation library, not a model**. Every component ships with a **naive reference implementation** (readable, slow) plus the fast path, and a numerical equivalence test between the two. That equivalence test is the whole point, it is what separates a reference implementation from a plausible one.

The goal is to answer a question that papers rarely answer directly: **which of these techniques actually earn their complexity, and at what scale**.

🚨 **Work in progress.** Milestone M0 (config schema, profiles, test harness) is done and green. The components themselves land in M1 to M7, see the roadmap below.

---

## ⚙️ Features

  🧩 **One config flag per deviation**, so a single field separates two runs that are otherwise identical

  🔬 **Naive path and fast path for every component**, with an equivalence test between them

  📐 **Five coherent profiles** shipped in `configs/`, from the vanilla 2017 block to a Gemma-style alternated model

  🚫 **No config activates everything at once**, because a model that stacks every brick is the counter-example this repo teaches against

  📄 **29 reference papers downloaded and indexed**, each module docstring citing its authors, year and arXiv id

  🎯 **Validation that fails loudly**, a typo in a YAML file raises instead of being silently ignored

  🧪 **Whole test suite runs on CPU in under 4 seconds**, so ablations stay cheap to check

---

## 🗂️ The deviations, and what each one replaces

| Component | Paper | Config flag | What it replaces in the 2017 block |
|---|---|---|---|
| RMSNorm | Zhang and Sennrich, 2019 ([1910.07467](https://arxiv.org/abs/1910.07467)) | `norm.kind` | LayerNorm, drops the mean subtraction and the bias |
| Pre-norm, sandwich norm | Xiong et al. 2020, Gemma 2 | `norm.placement` | post-norm, which needs a warmup to stay stable |
| QK-Norm | Henry et al., 2020 ([2010.04245](https://arxiv.org/abs/2010.04245)) | `attention.qk_norm` | raw q and k, which let attention logits drift |
| RoPE | Su et al., 2021 ([2104.09864](https://arxiv.org/abs/2104.09864)) | `position.kind` | sinusoidal absolute embeddings added to the input |
| YaRN, NTK, Position Interpolation | Peng et al. 2023, Chen et al. 2023 | `position.scaling` | nothing, the 2017 model had no context extension |
| ALiBi, NoPE | Press et al. 2021, Kazemnejad et al. 2023 | `position.kind` | positional embeddings entirely |
| MQA, GQA | Shazeer 2019, Ainslie et al. 2023 | `attention.kind` | one KV head per query head, the cache bottleneck |
| MLA | DeepSeek-V2, 2024 ([2405.04434](https://arxiv.org/abs/2405.04434)) | `attention.kind` | the KV cache itself, replaced by a latent vector |
| Sliding window, attention sinks | Mistral 7B, StreamingLLM | `attention.sliding_window` | full quadratic attention on every layer |
| Logit softcapping | Gemma 2, 2024 | `attention.logit_softcap` | unbounded attention logits |
| SwiGLU, GeGLU, ReGLU | Shazeer, 2020 ([2002.05202](https://arxiv.org/abs/2002.05202)) | `ffn.kind` | the ReLU feed-forward of width 4·d |
| Fine-grained MoE, shared experts | DeepSeekMoE, 2024 ([2401.06066](https://arxiv.org/abs/2401.06066)) | `moe.n_shared_experts` | the dense feed-forward |
| Aux-loss-free balancing | DeepSeek-V3, 2024 ([2408.15664](https://arxiv.org/abs/2408.15664)) | `moe.balance` | the auxiliary load-balancing loss, which fights the main loss |
| Multi-Token Prediction | Gloeckle et al., 2024 ([2404.19737](https://arxiv.org/abs/2404.19737)) | `model.mtp_depth` | single next-token prediction |
| muP | Yang et al., 2022 ([2203.03466](https://arxiv.org/abs/2203.03466)) | `mup.enabled` | retuning the learning rate at every width |
| WSD schedule | MiniCPM, 2024 ([2404.06395](https://arxiv.org/abs/2404.06395)) | `train.schedule` | cosine decay, which locks the token budget upfront |
| Speculative decoding | Leviathan et al., 2022 ([2211.17192](https://arxiv.org/abs/2211.17192)) | `generate.py` | one forward pass per generated token |

The full index, with one PDF per line and the milestone that implements it, lives in [papers/_INDEX.md](papers/_INDEX.md).

📚 **[docs/taxonomy.md](docs/taxonomy.md) sorts all of these by slot and by function**, meaning which part of the 2017 block they replace (tokenizer, positions, attention heads, score computation, add and norm, feed-forward, head, optimizer, inference) and which constraint they were invented for (memory, compute, quality, stability, long context, hyperparameter transfer). It ends with a "pick by constraint, not by slot" table, which is the one to read when choosing what goes into a model.

---

## Example Outputs

The M0 test suite, which validates the config schema and every shipped profile:

```
tests\test_config.py .................                                   [ 68%]
tests\test_configs_load.py ......                                        [ 92%]
tests\test_seed.py ..                                                    [100%]

============================= 25 passed in 3.24s ==============================
```

The five profiles, each one a coherent selection rather than a pile of features:

| Profile | Attention | Positions | Norm | FFN |
|---|---|---|---|---|
| `base.yaml` | MHA | sinusoidal | LayerNorm, post | ReLU, 4·d |
| `llama_style_150m.yaml` | GQA 16 / 4 | RoPE, theta 500k | RMSNorm, pre | SwiGLU |
| `moe_1b_a200m.yaml` | GQA 16 / 4 | RoPE | RMSNorm, pre | MoE, 64 experts, top 6 |
| `mla_long_ctx.yaml` | MLA, rank 512 | RoPE + YaRN ×4 | RMSNorm, pre | SwiGLU |
| `gemma_style.yaml` | GQA + sliding window | RoPE | RMSNorm, sandwich | GeGLU |

### 📝 Notes and observations

  🧮 **MLA cache math.** At 12 layers in fp16, GQA with 4 KV heads of 64 dims costs 12288 bytes per token. MLA with a rank of 512 and a 64 dim rotary key costs 13824 bytes, so MLA only wins once the number of KV heads grows.

  🔎 **`F.rms_norm` cannot be used as a fast path.** In torch 2.5.1 it computes in the input dtype, and in fp16 it matches an all-fp16 computation bit for bit rather than the fp32 one. A test locks this in and will fail the day torch fixes it, see `test_torch_rms_norm_does_not_upcast`.

  🧭 **What context extension actually buys, measured.** Training at 2048 and serving at 8192, the slowest RoPE band reaches 1.0924 rad instead of the 0.2731 rad it ever saw, so it is extrapolating into unseen angles. Position Interpolation and YaRN both bring it back to exactly 0.2731. The difference is the cost: PI squashes all 32 frequency bands, YaRN leaves 9 of them untouched because they already completed enough periods during training. Its attention temperature comes out at 1.1386, which is `0.1 · ln(4) + 1`.

  📉 **muP is width invariant to 1.31x, standard init to 31.24x.** Measured by the coordinate check over `d_model` in 128 to 1024, see [docs/mup_coord_check.md](docs/mup_coord_check.md). The spread at init alone is 1.02x, so everything above that comes from the training dynamics, not the initialization.

  ⚠️ **bf16 is not free on every GPU.** A GTX 1660 Ti reports `is_bf16_supported() == True`, but that is emulation, the compute capability is 7.5. The local profiles therefore use fp16, and bf16 is kept for the pod-sized MoE profile.

  🎛️ **The defaults of the schema are the modern baseline** (GQA, RoPE, RMSNorm pre-norm, SwiGLU, no bias). The 2017 model is not a default that got overridden, it lives entirely inside `base.yaml`.

---

## ⚙️ How it works

  🧾 **Everything starts from a config.** Nested Pydantic models describe the architecture, and `extra="forbid"` turns a YAML typo into an immediate error instead of a silently ignored key.

  ✅ **Validation runs across fields, not just types.** `n_heads` must be a multiple of `n_kv_heads`, MLA refuses an explicit `n_kv_heads` because its latent cache replaces KV heads, a sliding window is required before layers can alternate, and MTP needs tied embeddings because it shares the LM head.

  🧱 **One `Attention` class, not one class per variant.** MHA, MQA, GQA and MLA are dispatched internally, which is what makes an ablation a single field change.

  🔍 **Each component is written twice.** The naive version reads like the paper, the fast version uses `F.scaled_dot_product_attention`, and a test asserts they agree numerically.

  🌡️ **fp32 where it is not negotiable.** Normalization, the RoPE cos and sin tables, and the softmax and logits are computed in fp32 then cast back, which is the number one source of bf16 divergence.

  📊 **Metrics that catch silent failures.** Router entropy, per-expert token share and drop rate are logged for MoE, because a router collapse is invisible in the loss curve alone.

---

## 🗺️ Architecture Diagram

The decoder block, with every switchable component and the config flag that controls it:

![Architecture Diagram](assets/architecture.svg)

**Default socle** (what you get without asking for anything):

- RoPE, `theta = 10000` (500000 for long context)
- GQA with `n_kv_heads = n_heads / 4`
- RMSNorm, pre-norm, computed in fp32
- SwiGLU with `d_ff = round(8/3 · d, 256)`
- No bias on the linears or the norms, tied embeddings
- Output projections divided by `sqrt(2 · n_layers)` at init
- AdamW `betas = (0.9, 0.95)`, weight decay 0.1 excluded from norms, biases and embeddings

---

## 🛣️ Roadmap

| Milestone | Content | Status |
|---|---|---|
| **M0** | Config schema, five profiles, determinism, test harness | ✅ done, 25 tests green |
| **M1** | LayerNorm, RMSNorm, QK-Norm, DyT, scaled residual init, muP | ✅ done, 54 tests green |
| **M2** | RoPE and its two conventions, YaRN, NTK, ALiBi, NoPE | ✅ done, 96 tests green |
| **M3** | MHA, MQA, GQA, MLA with weight absorption, masks, KV caches | ⏳ |
| **M4** | SwiGLU family, fine-grained MoE, balancing, MTP heads | ⏳ |
| **M5** | Model assembly, optimizer groups, WSD schedule, training loop | ⏳ |
| **M6** | Sampling, incremental decoding, speculative decoding | ⏳ |
| **M7** | Benchmarks (KV memory, throughput) and the ablation table | ⏳ |

---

## 📂 Repository structure

```bash
├── assets/                      # For the README
│   ├── banner.svg
│   └── architecture.svg
│
├── configs/                     # One coherent profile per file, never everything at once
│   ├── base.yaml                # vanilla 2017, the zero-ablation reference
│   ├── llama_style_150m.yaml    # RoPE + GQA + RMSNorm + SwiGLU
│   ├── moe_1b_a200m.yaml        # fine-grained MoE + shared experts + aux-loss-free
│   ├── mla_long_ctx.yaml        # MLA + YaRN + attention sinks
│   └── gemma_style.yaml         # alternated local and global attention + QK-norm + softcap
│
├── src/mt/
│   ├── config.py                # Pydantic schema, all the cross-field validation
│   ├── model.py                 # Transformer, Block, forward and losses          (M5)
│   ├── layers/
│   │   ├── norm.py              # LayerNorm, RMSNorm, QK-Norm, DyT                (M1)
│   │   ├── pos.py               # RoPE, PI, NTK, YaRN, ALiBi, NoPE                (M2)
│   │   ├── attention.py         # MHA / MQA / GQA / MLA, SWA, sinks, softcap      (M3)
│   │   ├── ffn.py               # MLP, SwiGLU, GeGLU, ReGLU                       (M4)
│   │   ├── moe.py               # router, experts, shared experts, balancing      (M4)
│   │   └── heads.py             # LM head, tied embeddings, MTP heads             (M4)
│   ├── cache.py                 # dense KV cache, SWA ring buffer, MLA latent     (M3)
│   ├── init.py                  # standard init, scaled residual init, muP        (M1)
│   ├── optim.py                 # AdamW, param groups, WSD and cosine, z-loss     (M5)
│   ├── train.py                 # minimal training loop, grad accum, checkpoints  (M5)
│   ├── generate.py              # sampling, KV cache, speculative decoding        (M6)
│   └── utils/
│       └── seed.py              # set_determinism
│
├── tests/                       # Equivalence tests, naive path against fast path
│   ├── test_config.py
│   ├── test_configs_load.py
│   └── test_seed.py
│
├── papers/                      # 29 reference PDFs, gitignored
│   ├── _INDEX.md                # paper, component, config flag, milestone
│   └── download.sh              # restores the PDFs after a clone
│
├── bench/                       # KV memory, throughput, ablations                (M7)
├── docs/                        # ablations.md, mup_coord_check.md                (M7)
│
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 💻 Run it on Your PC

Clone the repository and install dependencies:

```bash
git clone https://github.com/Thibault-GAREL/modern-transformer.git
cd modern-transformer

python -m venv .venv # if you don't have a virtual environment
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -e ".[dev]"
```

⚠️ The default install pulls the CPU build of PyTorch. For a **CUDA-compatible GPU**, install torch first:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

### Run the tests

```bash
pytest
```

### Load a profile

```python
from mt.config import Config

cfg = Config.from_yaml("configs/llama_style_150m.yaml")
print(cfg.model.attention.kind, cfg.model.head_dim)   # gqa 64
```

### Get the reference papers

The PDFs are gitignored because they are heavy, this restores all 29 of them:

```bash
bash papers/download.sh
```

---

## 📖 Inspiration / Sources

This project is based on the papers listed in [papers/_INDEX.md](papers/_INDEX.md), and most directly on:

- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), the baseline every flag deviates from
- 📄 [RoFormer, Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
- 📄 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020)
- 📄 [DeepSeek-V2](https://arxiv.org/abs/2405.04434) and [DeepSeek-V3](https://arxiv.org/abs/2412.19437), for MLA, fine-grained MoE and MTP
- 📄 [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Grattafiori et al., 2024)

Related work of mine: [Language Models from Scratch](https://github.com/Thibault-GAREL/Language_Models), where I built a bigram model and a 2017-style Transformer from scratch. This repo picks up exactly where that one stops.

Code created by me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)
