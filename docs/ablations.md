# Ablations: which of these techniques earn their complexity

This is the document the repository exists for. Every other file lets you turn a
component on. This one reports what happened when we did.

Everything below is measured on the hardware named with the result, on a GTX
1660 Ti with 6 GB. That is a small machine, and it bounds what can honestly be
concluded: results at this scale say something about *whether a mechanism is
implemented correctly and what it costs*, and very little about how it behaves
at a billion parameters. Where a published result does not reproduce here, the
disagreement is reported rather than smoothed over.

---

## Method

`bench/ablation.py` trains each variant on the same corpus, for the same number
of tokens, with the same seed, and scores it on held-out loss. Each variant
differs from its stated baseline by **exactly one config field**. Parameter
counts are reported alongside, because a variant that wins by being bigger has
not won.

The corpus is this repository's own source and documentation read as raw bytes.
It is real text with real structure, reproducible from a clone, and byte level
means no tokenizer choice contaminates the comparison.

```bash
python bench/ablation.py                    # the full sweep
python bench/kv_memory.py                   # cache cost per token
python bench/throughput.py --context 2048   # prefill and decode speed
python bench/coord_check.py                 # muP width invariance
```

---

## 1. Memory: what a KV cache actually costs

From `bench/kv_memory.py`, a 32 layer model with `d_model` 4096 and 32 heads,
serving 32k of context at batch 8 in fp16.

| variant | bytes/token | total cache | vs MHA |
|---|---|---|---|
| MHA | 512.0 KB | 128.0 GB | 1.0x |
| GQA g=4 | 128.0 KB | 32.0 GB | 4.0x |
| GQA g=8 | 64.0 KB | 16.0 GB | 8.0x |
| MLA | 36.0 KB | 9.0 GB | 14.2x |
| MQA | 16.0 KB | 4.0 GB | 32.0x |
| SWA w=4096 (GQA g=4) | 128.0 KB | 4.0 GB | bounded |

**Conclusion.** MHA at this context does not fit on any single machine, which is
the entire reason this slot changed. Two things the table makes visible that the
papers tend not to:

The sliding window row has an **unchanged cost per token**. Its saving comes
from the cache no longer growing past the window, not from the mask. A mask
alone saves compute and nothing else.

**MLA is not the cheapest option.** MQA is smaller. MLA beats a dense cache only
when `kv_lora_rank + qk_rope_head_dim < 2 * n_kv_heads * head_dim`, which at
this head dim means below roughly 2.2 KV heads. MLA is chosen over MQA for
quality, not for memory.

---

## 2. Speed: where the memory saving turns into latency

From `bench/throughput.py`, 8 layers, `d_model` 512, 8 heads, batch 8, fp16,
64 decode steps, on the local GPU.

**Short context (256 tokens)**

| variant | prefill tok/s | decode tok/s | peak MiB | vs MHA |
|---|---|---|---|---|
| MHA | 8,345 | 441 | 282 | 1.00x |
| GQA g=4 | 8,973 | 345 | 271 | 0.78x |
| MQA | 9,060 | 370 | 262 | 0.84x |

**Long context (2048 tokens)**

| variant | prefill tok/s | decode tok/s | peak MiB | vs MHA |
|---|---|---|---|---|
| MHA | 6,827 | 216 | 761 | 1.00x |
| GQA g=4 | 6,903 | 194 | 550 | 0.90x |
| MQA | 6,456 | 252 | 515 | 1.16x |
| MLA (naive) | 5,029 | 35 | 574 | 0.16x |
| MLA (absorbed) | 5,019 | 143 | 594 | 0.66x |
| SWA w=256 (GQA g=4) | 6,667 | 227 | 501 | 1.05x |

**Conclusion.** At 256 tokens **GQA is slower than MHA**. The cache is small
enough that decode is dominated by kernel launches and by the extra work of
expanding the KV heads, not by reading the cache. The crossover only appears as
the context grows, and even at 2048 on this small model GQA has barely reached
parity while MQA is ahead.

This is the single most useful result in this document for someone choosing an
architecture. **GQA is not free speed, it is a memory optimization that becomes
a speed optimization only once the cache dominates.** Below that point it costs.

**The MLA absorption is worth its code.** The naive path runs at 35 tok/s, the
absorbed path at 143, a **4.1x** speedup from folding `W_UK` into the queries and
`W_UV` into the output. Without the absorption, MLA is not a serving
optimization at all. It still trails MHA at this scale, because MLA trades
arithmetic for cache and this model's cache is not the bottleneck.

### A concrete trap found while measuring this

`F.scaled_dot_product_attention(..., enable_gqa=True)` is supposed to be
strictly better than expanding the KV heads, since it skips the expansion
entirely. Measured on torch 2.5.1, B=8, H=8, KV=2, T=2048, fp16:

| path | extra memory | time |
|---|---|---|
| `enable_gqa=True` | **+3496 MiB** | **100.0 ms** |
| `repeat_kv` then SDPA | +48 MiB | 36.3 ms |

`enable_gqa` falls back to the math backend and materializes the whole
`(B, H, T, T)` score matrix, costing **73x the memory and 2.8x the time**. The
library therefore expands the heads, behind `USE_SDPA_ENABLE_GQA = False`, and
`test_enable_gqa_still_regresses` fails the day a torch release fixes this.

---

## 3. Length generalization: what context extension does and does not do

Measured on a model trained to copy a repeated block at 32 tokens and evaluated
at 128, in `tests/test_long_context.py`.

| | perplexity |
|---|---|
| at the training length, 32 tokens | **1.00** |
| at 4x the training length, 128 tokens | **1.7 million** |

**Conclusion, part one.** RoPE does not degrade gracefully past its training
length, it collapses. A model that copies perfectly at 32 tokens is worthless at
128. This is the problem every scheme in the positional section exists to solve,
and the size of the collapse is worth internalizing.

**Conclusion, part two, and it is a negative result.** Applying any scaling
scheme to the frozen weights rescues nothing:

| scheme applied zero-shot | ppl at 32 | ppl at 128 |
|---|---|---|
| plain RoPE | 1.00 | 1.7M |
| Position Interpolation x4 | 415,857 | 1.7M |
| NTK-aware x4 | 30.06 | 2.0M |
| YaRN x4 | 3,329,750 | 1.7M |

Every scheme stays in the collapsed regime at 128 **and** destroys the short
context it used to handle. After 60 fine-tuning steps at the target length, the
ordering is Position Interpolation 29.7, NTK 39.6, plain RoPE 41.0, YaRN 48.4,
which does **not** reproduce the published ranking.

**This document therefore refuses to rank the extension schemes.** Four reasons
this setup cannot support that claim, none of them a defect in the
implementations, which are verified against their formulas in `tests/test_pos.py`:

- the task demands an *exact* positional lookup at a fixed offset, where natural
  language attention is soft and statistical
- 60 fine-tuning steps is nowhere near the budget the YaRN paper uses
- 2 layers and 64 dimensions is far too small for a per-frequency-band argument
  to have room to operate
- YaRN's band split is tuned for `theta = 10000` over contexts of thousands, not
  for 32 to 128

Ranking these schemes needs a real model at a real context. What this repository
can honestly claim is that the frequencies are computed as published, the
temperature factor is applied, and the collapse they address is real.

---

## 4. muP: the one result that does transfer

From `bench/coord_check.py`, on the real `Transformer`, `d_model` from 128 to
1024, 30 AdamW steps.

| | worst width spread |
|---|---|
| standard parametrization | 31.63x |
| **muP** | **1.03x** |

**Conclusion.** muP works, and it is the cleanest result in this document
because coordinate invariance is a property of the parametrization rather than
of the data. Two things found while getting there, both documented in
[mup_coord_check.md](mup_coord_check.md): the output layer needs a different
exponent from the hidden ones (`1/mult` against `1/sqrt(mult)`), and measuring
on a stand-in MLP instead of the real model reported 1.31x rather than 1.03x.

A coordinate check measures the model you actually run.

---

## 5. Quality at a fixed token budget

From `bench/ablation.py`: 6 layers, `d_model` 256, byte level, **2,457,600 tokens
per variant**, identical seed, held-out loss on a 10% split.

### Deviating from the 2017 block, one field at a time

| variant | val loss | vs baseline | params | what changed |
|---|---|---|---|---|
| `vanilla-2017` | 3.9888 | reference | 4.80M | the 2017 decoder |
| `pre-norm` | 3.4028 | **-0.586** | 4.80M | `norm.placement` post to pre |
| `rmsnorm` | 3.3795 | -0.023 | 4.80M | `norm.kind` layernorm to rmsnorm (on pre-norm) |
| `rope` | 2.0345 | **-1.954** | 4.80M | `position` sinusoidal to rope |
| `swiglu` | 3.9878 | -0.001 | 4.80M | `ffn` mlp 4d to swiglu 8/3 d |
| `scaled-init` | 3.9889 | +0.000 | 4.80M | `init.scaled_residual` off to on |
| `modern-socle` | **1.9591** | **-2.030** | 4.78M | all four together |

### Deviating from the modern socle

| variant | val loss | vs socle | params | what changed |
|---|---|---|---|---|
| `gqa` | **1.9450** | **-0.014** | **4.19M** | attention MHA to GQA g=4 |
| `mtp` | **1.9193** | **-0.040** | 5.70M | `mtp_depth` 0 to 1 |
| `moe` | 1.9508 | -0.008 | 8.26M (4.20M active) | dense FFN to 8 experts, top 2, plus 1 shared |
| `qk-norm` | 1.9695 | **+0.010** | 4.78M | `attention.qk_norm` off to on |
| `sliding-window` | 2.0057 | **+0.047** | 4.78M | window 128, one global layer in three |

### What these numbers say

**Positions dominate everything else.** RoPE alone accounts for `-1.954` of the
`-2.030` total gain. Nothing else in the sweep is within an order of magnitude
of it. If only one thing were changed from the 2017 block, it should be this.

**Pre-norm is the second real win**, at `-0.586`, and it is a placement change
costing nothing. RMSNorm on top adds only `-0.023`: it is adopted for being
cheaper and simpler, not for being more accurate, and this measurement agrees.

**SwiGLU and scaled residual init did nothing here**, `-0.001` and `+0.000`.
Both were measured against the post-norm 2017 baseline, whose trainability
problem plausibly masks any FFN gain, and at 6 layers a depth correction has
almost no depth to correct. This is a limit of the setup, not evidence that
either is useless, and it is exactly the kind of null result a small-scale
ablation should report rather than bury.

**GQA is free, and slightly better than free.** It reached a *lower* loss than
MHA with **0.6M fewer parameters**. At this size the quality cost that motivates
"GQA is a memory trade" simply is not visible, while the parameter saving is.

**QK-norm and sliding window both cost quality**, `+0.010` and `+0.047`. Both
behave exactly as the taxonomy predicts: QK-norm treats an instability that a
5M parameter model does not have, and a sliding window removes information the
model was using. Neither is a defect, they are the price of solutions to
problems this scale does not have.

**MoE bought almost nothing for 1.7x the parameters**, `-0.008` for 8.26M total
against 4.78M. Active parameters were 4.20M, so the compute was comparable and
the extra capacity went unused. A model this small on this corpus is not
capacity limited, which is precisely when MoE has nothing to offer.

**MTP was the best single addition to the socle**, `-0.040`. The denser training
signal helps even here, which is the one result in this table that beat its
reputation for needing scale.

---

## What this all adds up to

Reading the four sections together, at this scale and on this corpus:

| Technique | Verdict here | Take it? |
|---|---|---|
| **RoPE** | -1.954 loss, by far the largest effect measured | **Always** |
| **Pre-norm** | -0.586 loss, free | **Always** |
| **RMSNorm** | -0.023 loss, cheaper and simpler | **Always**, for the simplicity |
| **GQA** | lower loss with fewer parameters, and 4x less cache | **Always**, but not for speed below long context |
| **MTP** | -0.040 loss, the best addition to the socle | Worth trying earlier than its reputation suggests |
| **SwiGLU** | no measurable effect at 6 layers | Keep, on the published evidence at scale, not on this |
| **Scaled residual init** | no effect at 6 layers | Keep for depth, it has none to fix here |
| **muP** | 1.03x width invariance against 31.63x | **Yes** if you tune hyperparameters at all |
| **MLA** | 4.1x faster absorbed than naive, still behind MHA here | Only when the cache is genuinely the bottleneck |
| **QK-norm** | +0.010 loss, treats an absent instability | No, below a billion parameters |
| **Sliding window** | +0.047 loss, removes usable information | No, unless the context forces it |
| **MoE** | -0.008 loss for 1.7x the parameters | No, this scale is not capacity limited |
| **Context extension** | no scheme rescued a collapsed model zero-shot | Cannot be judged at this scale, see section 3 |

The pattern is the one the repository set out to demonstrate. **The techniques
that help unconditionally are the cheap ones** (RoPE, pre-norm, RMSNorm, and
GQA). **The expensive ones solve problems a small model does not have**, and
adding them costs quality rather than buying it. A 5M parameter model with MLA,
MoE, sliding window and QK-norm would be slower, larger, harder to debug, and
*worse* than the 4.78M socle.

That model is the counter-example this repository exists to teach against, and
these numbers are what it looks like when someone builds it.

### How far to trust this

Every number above comes from one seed, one corpus, one model size, and roughly
2.5M tokens. That is enough to detect a 2.0 loss gap, plainly not enough to
resolve a 0.01 one. Treat the large effects as real, the small ones as noise,
and the null results as statements about this scale rather than about the
techniques. The published results at 7B and beyond are not contradicted by any
of this, and a repository that trains on a laptop GPU is not in a position to
contradict them.
