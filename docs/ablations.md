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
per variant**, identical seed, held-out loss on a 10% split, corpus frozen at
sha `69a497f3a393`.

### A methodological correction, made after the first sweep

The first version of this benchmark read the repository's files live. That looks
reproducible and is not: the repo grew by 23 KB between two sweeps, and the
held-out loss of an **unchanged** config moved by **0.09**, larger than most of
the effects being measured. Several conclusions drawn from that first sweep were
noise, and two of them reversed once the corpus was frozen.

The corpus is now cached to disk on first use, its sha256 is recorded in every
results file, and `--refresh-corpus` states plainly that rebuilding it makes
previous numbers incomparable. **Every number below comes from a single sweep on
a single frozen corpus.**

### Deviating from the 2017 block, one field at a time

| variant | val loss | vs baseline | params | what changed |
|---|---|---|---|---|
| `vanilla-2017` | 3.9280 | reference | 4.80M | the 2017 decoder |
| `pre-norm` | 3.3768 | **-0.551** | 4.80M | `norm.placement` post to pre |
| `rmsnorm` | 3.3439 | -0.033 | 4.80M | `norm.kind` layernorm to rmsnorm (on pre-norm) |
| `rope` | 1.9131 | **-2.015** | 4.80M | `position` sinusoidal to rope |
| `swiglu` | 3.9270 | -0.001 | 4.80M | `ffn` mlp 4d to swiglu 8/3 d |
| `scaled-init` | 3.9271 | -0.001 | 4.80M | `init.scaled_residual` off to on |
| `modern-socle` | **1.8599** | **-2.068** | 4.78M | all four together |

### Deviating from the modern socle

| variant | val loss | vs socle | params | deployed | what changed |
|---|---|---|---|---|---|
| `mtp` | **1.8211** | **-0.039** | 5.70M | 4.78M | `mtp_depth` 0 to 1 |
| `gqa` | 1.8500 | -0.010 | 4.19M | 4.19M | MHA to GQA g=4 |
| `qk-norm` | 1.8522 | -0.008 | 4.78M | 4.78M | `attention.qk_norm` on |
| `moe` | 1.8544 | -0.006 | 8.26M | 8.26M | 8 experts, top 2, plus 1 shared |
| `sliding-window` | 1.8778 | **+0.018** | 4.78M | 4.78M | window 128, one global layer in three |

### Combinations, because individually good changes need not compose

| variant | val loss | params | **deployed** | what it is |
|---|---|---|---|---|
| **`best-mqa-mtp2`** | **1.8039** | 5.70M | **4.10M** | **MQA + MTP depth 2. The winner** |
| `best-plus-qknorm` | 1.8126 | 5.01M | 4.19M | GQA g=4 + MTP 1 + QK-norm |
| `best-mtp2` | 1.8244 | 5.83M | 4.19M | GQA g=4 + MTP depth 2 |
| `best-candidate` | 1.8482 | 5.01M | 4.19M | GQA g=4 + MTP depth 1 |
| `best-gqa8` | 1.8602 | 4.90M | 4.10M | MQA + MTP depth 1 |
| `everything` | **1.8725** | **10.70M** | 7.67M | every brick at once |
| `champion` | 1.8335 | 5.70M | 4.10M | the winner plus QK-norm, which costs +0.030 |

Deployed size excludes the MTP modules, which are dropped at inference.

### MatFormer: two things are called Matryoshka, only one is an architecture

Worth separating before reading the numbers, because the names collide:

- **Matryoshka Representation Learning** (Kusupati et al., 2022,
  [2205.13147](https://arxiv.org/abs/2205.13147)) nests *embedding dimensions*,
  so a 768 dim vector can be truncated to 64 and stay useful. This is what
  Nomic v1.5 and `text-embedding-3` do. It is a **retrieval** technique, and a
  decoder LLM has no embedding to truncate: its output is a softmax over a
  vocabulary.
- **MatFormer** (Devvrit et al., 2023,
  [2310.07707](https://arxiv.org/abs/2310.07707), NeurIPS 2024) nests *FFN
  widths* inside the Transformer, so the weights of a smaller model are literally
  a sub-matrix of the larger one. This is the LLM analogue, and it is what Gemma
  3n ships as its E2B and E4B variants.

Only the second is implemented here, behind `ffn.mat_granularities`.

| model | val loss | params | what it is |
|---|---|---|---|
| `modern-socle` | **1.8599** | 4.78M | a dense FFN trained alone |
| `matformer` at 1.0 | 1.8644 | 4.78M | the full width of a nested model |
| `matformer` at 0.5 | 1.8821 | (same weights) | the half-width slice |
| `matformer-half` | **1.8620** | 3.21M | a half-width FFN trained alone |
| `matformer` at 0.25 | 1.9167 | (same weights) | the quarter-width slice |

**The paper's central claim does not reproduce at this scale.** MatLM reports
nested models beating independently trained counterparts. Here the full width
costs **+0.005** against a dense model, and the 0.5 slice is **+0.020 worse**
than a half-width model trained on its own. Neither gap is large, and both go
the wrong way.

Three reasons to treat this as a scale statement rather than a refutation: the
paper's MatLM is 850M parameters against 5M here, it trains far longer than 600
steps, and joint optimization has more room to share structure when there is
more structure to share.

**What MatFormer buys is not accuracy, and the compute is not saved either.**
Training three granularities costs three forward and backward passes per step,
so one MatFormer run costs about what three separate runs cost. The real return
is that Mix'n'Match then extracts *hundreds* of intermediate sizes that were
never explicitly trained, from one checkpoint and one serving pipeline. For a
small model meant to ship at several sizes, that is a genuine operational win.
For a single deployment target, it is three times the training cost for a
slightly worse model.

---

### What these numbers say

**Positions dominate everything else.** RoPE alone accounts for `-2.015` of the
`-2.068` total. Nothing else in the sweep is within an order of magnitude. If
only one thing were ever changed from the 2017 block, it should be this.

**Pre-norm is the second real win**, at `-0.551`, for a placement change costing
nothing. RMSNorm on top adds `-0.033`: it is adopted for being cheaper, not more
accurate, and this agrees.

**SwiGLU and scaled residual init did nothing measurable**, `-0.001` each. Both
were measured against a post-norm baseline whose trainability problem plausibly
masks any FFN gain, and at 6 layers a depth correction has almost no depth to
correct. A limit of the setup, not evidence that either is useless.

**Combining works, but not additively.** MQA gives `-0.010` alone and MTP gives
`-0.039` alone. Together they give `-0.056`, more than either but less than the
sum. Assembling a config by adding up winners would have predicted `-0.049` and
picked GQA over MQA, which the measurement contradicts.

**And sometimes a winner reverses sign.** QK-norm improves the socle (`-0.008`)
and improves a GQA plus MTP-1 model considerably more (`-0.036`). Added to the
winning combination it **costs `+0.030`**, wiping out most of what MTP gained.
That single row is the argument for measuring combinations instead of summing
individual effects, and it is why `configs/best.yaml` ships with QK-norm off
despite it winning two of the three comparisons it appears in.

**`everything` is the counter-example the repository exists for.** Stacking MoE,
sliding window, QK-norm, MTP and GQA gives **2.2x the parameters, 1.9x the
deployed size, and a worse loss** than the 4.10M winner. Slower, larger, harder
to debug, less accurate.

**Two conclusions reversed when the corpus was frozen**, and they are recorded
here rather than quietly corrected. QK-norm appeared to *cost* quality on the
unfrozen corpus (`+0.010`) and in fact helps slightly (`-0.008` alone, `-0.036`
in combination). And MQA appeared to beat GQA by a wide margin in one sweep and
lose to it in another, before settling as a genuine but small win. Both effects
sit near the noise floor of a single-seed run, which is exactly why the frozen
corpus mattered.

---

## What this all adds up to

Reading the five sections together, at this scale and on this corpus:

| Technique | Verdict here | Take it? |
|---|---|---|
| **RoPE** | -2.015 loss, by far the largest effect measured | **Always** |
| **Pre-norm** | -0.551 loss, free, removes the warmup requirement | **Always** |
| **RMSNorm** | -0.033 loss, cheaper and simpler than LayerNorm | **Always**, for the simplicity |
| **MTP** | -0.039 alone, and the deployed model is smaller | **Yes**, earlier than its reputation suggests |
| **MQA / GQA** | -0.010 to -0.056 in combination, 4x to 8x less cache | **Yes**, but not for speed below long context |
| **QK-norm** | -0.008 alone, -0.036 in combination, costs nothing | Yes, though the effect is near the noise floor |
| **muP** | 1.03x width invariance against 31.63x | **Yes** if you tune hyperparameters at all |
| **SwiGLU** | no measurable effect at 6 layers | Keep, on published evidence at scale, not on this |
| **Scaled residual init** | no effect at 6 layers | Keep for depth, it has none to fix here |
| **MLA** | 4.1x faster absorbed than naive, still behind MHA here | Only when the cache is genuinely the bottleneck |
| **MoE** | -0.006 loss for 1.7x the parameters | No, this scale is not capacity limited |
| **Sliding window** | +0.018 loss, removes usable information | No, unless the context forces it |
| **Context extension** | no scheme rescued a collapsed model zero-shot | Cannot be judged at this scale, see section 3 |

The pattern is the one the repository set out to demonstrate. **The techniques
that help unconditionally are the cheap ones**, and they are cheap because they
change how information is represented rather than adding machinery. **The
expensive ones solve problems a small model does not have**, and adding them
costs quality rather than buying it.

`configs/best.yaml` is the winner, and it contains none of the expensive bricks.

### How far to trust this

Every number comes from **one seed, one corpus, one model size, roughly 2.5M
tokens**. That is enough to detect a 2.0 loss gap and plainly not enough to
resolve a 0.01 one. Treat the large effects as real, anything under about 0.02
as provisional, and the null results as statements about this scale rather than
about the techniques.

The first sweep of this document taught that lesson the hard way: an unfrozen
corpus moved an unchanged config by 0.09 and reversed two conclusions. Effects
smaller than that were never measurements in the first place.

Published results at 7B and beyond are not contradicted by any of this. A
repository that trains on a laptop GPU is not in a position to contradict them,
and where the two disagree the published number is the one to believe.
