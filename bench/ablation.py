"""One change at a time, same token budget, measured.

This is the benchmark the repo exists for. Every other file lets you switch a
component on. This one answers whether doing so was worth it.

Method: each variant differs from its stated baseline by exactly one config
field, trains on the same corpus for the same number of tokens with the same
seed, and is scored by held-out loss. Parameter counts are reported alongside,
because a variant that wins by being larger has not won.

The corpus is this repository's own source and documentation, read as raw
bytes. It is real text with real structure, it is reproducible from a clone,
and byte level means no tokenizer choice contaminates the comparison.

    python bench/ablation.py                       # full sweep
    python bench/ablation.py --steps 200 --only rope,swiglu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from mt.config import (
    AttentionConfig,
    FFNConfig,
    InitConfig,
    ModelConfig,
    MoEConfig,
    NormConfig,
    PositionConfig,
    TrainConfig,
)
from mt.model import Transformer
from mt.optim import build_optimizer, build_scheduler
from mt.utils.numerics import autocast_dtype, pick_device, resolve_precision
from mt.utils.seed import set_determinism

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_corpus(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The repository's own text, split 90/10 into train and validation."""
    parts = []
    for pattern in ("src/mt/**/*.py", "tests/*.py", "bench/*.py", "docs/*.md", "*.md"):
        for path in sorted(REPO.glob(pattern)):
            parts.append(path.read_bytes())
    blob = b"\n".join(parts)
    if len(blob) < seq_len * 64:  # keep the split meaningful on a thin checkout
        blob = blob * (seq_len * 64 // max(len(blob), 1) + 1)
    data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).long()
    split = int(0.9 * len(data))
    return data[:split], data[split:]


def batches(data: torch.Tensor, batch: int, seq_len: int, device, generator):
    hi = len(data) - seq_len - 1
    starts = torch.randint(0, hi, (batch,), generator=generator)
    x = torch.stack([data[s : s + seq_len] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + 1 + seq_len] for s in starts]).to(device)
    return x, y


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def base_model(**overrides) -> ModelConfig:
    """The 2017 decoder, which every other variant deviates from."""
    cfg = {
        "d_model": 256,
        "n_layers": 6,
        "vocab_size": 256,
        "max_seq_len": 512,
        "bias": True,
        "tie_embeddings": True,
        "attention": AttentionConfig(kind="mha", n_heads=8, n_kv_heads=None),
        "position": PositionConfig(kind="sinusoidal"),
        "ffn": FFNConfig(kind="mlp", activation="relu", mult=4.0, multiple_of=1),
        "norm": NormConfig(kind="layernorm", placement="post"),
        "init": InitConfig(scheme="fixed", std=0.02, scaled_residual=False),
    }
    cfg.update(overrides)
    return ModelConfig(**cfg)


def modern_model(**overrides) -> ModelConfig:
    """The default socle: RoPE, RMSNorm pre-norm, SwiGLU, no bias."""
    cfg = {
        "bias": False,
        "attention": AttentionConfig(kind="mha", n_heads=8, n_kv_heads=None),
        "position": PositionConfig(kind="rope", rope_theta=10_000.0),
        "ffn": FFNConfig(kind="swiglu", multiple_of=1),
        "norm": NormConfig(kind="rmsnorm", placement="pre"),
        "init": InitConfig(scheme="fixed", std=0.02, scaled_residual=True),
    }
    cfg.update(overrides)
    return base_model(**cfg)


def variants() -> dict[str, tuple[str, str, ModelConfig]]:
    """``name -> (baseline, what changed, config)``."""
    v: dict[str, tuple[str, str, ModelConfig]] = {}

    v["vanilla-2017"] = ("", "the reference", base_model())
    v["pre-norm"] = (
        "vanilla-2017", "norm.placement post to pre",
        base_model(norm=NormConfig(kind="layernorm", placement="pre")),
    )
    v["rmsnorm"] = (
        "pre-norm", "norm.kind layernorm to rmsnorm",
        base_model(norm=NormConfig(kind="rmsnorm", placement="pre")),
    )
    v["rope"] = (
        "vanilla-2017", "position sinusoidal to rope",
        base_model(position=PositionConfig(kind="rope")),
    )
    v["swiglu"] = (
        "vanilla-2017", "ffn mlp 4d to swiglu 8/3 d",
        base_model(ffn=FFNConfig(kind="swiglu", multiple_of=1)),
    )
    v["scaled-init"] = (
        "vanilla-2017", "init.scaled_residual off to on",
        base_model(init=InitConfig(scheme="fixed", std=0.02, scaled_residual=True)),
    )

    v["modern-socle"] = ("vanilla-2017", "all four of the above together", modern_model())
    v["gqa"] = (
        "modern-socle", "attention mha to gqa g=4",
        modern_model(attention=AttentionConfig(kind="gqa", n_heads=8, n_kv_heads=2)),
    )
    v["qk-norm"] = (
        "modern-socle", "attention.qk_norm off to on",
        modern_model(
            attention=AttentionConfig(kind="mha", n_heads=8, n_kv_heads=None, qk_norm=True)
        ),
    )
    v["sliding-window"] = (
        "modern-socle", "sliding_window 128 with a global layer every 3",
        modern_model(
            attention=AttentionConfig(
                kind="mha", n_heads=8, n_kv_heads=None,
                sliding_window=128, global_every=3,
            )
        ),
    )
    v["moe"] = (
        "modern-socle", "ffn dense to MoE, 8 experts top 2 plus 1 shared",
        modern_model(
            moe=MoEConfig(
                enabled=True, n_experts=8, top_k=2, n_shared_experts=1,
                d_ff_expert=176, first_k_dense=1, balance="aux_loss_free",
            )
        ),
    )
    v["mtp"] = (
        "modern-socle", "mtp_depth 0 to 1",
        modern_model(mtp_depth=1, tie_embeddings=True),
    )
    return v


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, data, args, device, amp_dtype, seed: int = 1234) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        for _ in range(args.eval_batches):
            x, y = batches(data, args.batch, args.seq_len, device, gen)
            _, _, aux = model(x, y)
            losses.append(float(aux.ce))
    model.train()
    return sum(losses) / len(losses)


def run_variant(name: str, model_cfg: ModelConfig, train_data, val_data, args, device):
    set_determinism(args.seed)
    precision = resolve_precision("bf16", device)
    amp_dtype = autocast_dtype(precision)

    train_cfg = TrainConfig(
        lr=args.lr,
        max_steps=args.steps,
        warmup_steps=max(args.steps // 20, 1),
        micro_batch_size=args.batch,
        seq_len=args.seq_len,
        precision=precision,
    )
    model = Transformer(model_cfg, z_loss_coef=1e-4).to(device)
    optimizer = build_optimizer(model, model_cfg, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))
    gen = torch.Generator().manual_seed(args.seed)

    t0 = time.perf_counter()
    model.train()
    for _ in tqdm(range(args.steps), desc=f"{name:<16}", unit="step", leave=False):
        x, y = batches(train_data, args.batch, args.seq_len, device, gen)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model.balance_step()
    train_s = time.perf_counter() - t0

    val = evaluate(model, val_data, args, device, amp_dtype)
    result = {
        "val_loss": val,
        "val_ppl": float(torch.tensor(val).exp()),
        "params": model.n_params(),
        "active_params": model.n_active_params(),
        "train_s": train_s,
        "tokens": args.steps * args.batch * args.seq_len,
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--only", type=str, default=None, help="comma separated variant names")
    p.add_argument("--out", type=Path, default=REPO / "docs" / "ablation_results.json")
    args = p.parse_args()

    device = pick_device()
    train_data, val_data = load_corpus(args.seq_len)
    all_variants = variants()
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        all_variants = {k: v for k, v in all_variants.items() if k in wanted}

    print(
        f"corpus  : {len(train_data):,} train bytes, {len(val_data):,} validation\n"
        f"budget  : {args.steps * args.batch * args.seq_len:,} tokens per variant\n"
        f"device  : {device}\n"
    )

    results = {}
    for name, (baseline, changed, cfg) in all_variants.items():
        r = run_variant(name, cfg, train_data, val_data, args, device)
        r["baseline"] = baseline
        r["changed"] = changed
        results[name] = r
        print(
            f"{name:<16} val loss {r['val_loss']:.4f}  ppl {r['val_ppl']:7.2f}  "
            f"{r['params'] / 1e6:5.2f}M params  {r['train_s']:5.0f}s"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args": vars(args), "results": results},
                                   indent=2, default=str), encoding="utf-8")

    print(f"\n{'variant':<16} {'val loss':>9} {'vs baseline':>12} {'params':>9} {'change'}")
    print("-" * 78)
    for name, r in results.items():
        if r["baseline"]:
            delta = r["val_loss"] - results[r["baseline"]]["val_loss"]
            verdict = f"{delta:+.4f}"
        else:
            verdict = "reference"
        print(
            f"{name:<16} {r['val_loss']:>9.4f} {verdict:>12} "
            f"{r['params'] / 1e6:>8.2f}M  {r['changed']}"
        )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
