"""muP coordinate check (Yang et al., 2022, arXiv 2203.03466).

The only meaningful test of a muP implementation. Train the same toy stack at
several widths and record the average absolute activation ("coordinate size")
of every layer at every step. Under muP those curves lie on top of each other,
under standard parametrization they spread apart as width grows.

Run it with:

    python bench/coord_check.py --steps 8 --out docs/assets

Note on scope: this uses a residual MLP stack, not the full Transformer, since
``model.py`` lands in M5. It exercises muP changes (a) init and (b) learning
rate groups. Changes (c) attention scaling and (d) the output multiplier need
the real model and will be added to this same script then.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from mt.config import AttentionConfig, InitConfig, ModelConfig, MuPConfig
from mt.init import (
    init_weights,
    mark_output_layer,
    mark_residual_projection,
    output_logit_multiplier,
    width_multiplier,
)
from mt.layers.norm import RMSNorm
from mt.utils.seed import set_determinism

WIDTHS = (128, 256, 512, 1024)
BASE_WIDTH = 128


class Block(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up = nn.Linear(d_model, 4 * d_model, bias=False)
        self.down = mark_residual_projection(nn.Linear(4 * d_model, d_model, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(F.gelu(self.up(self.norm(x))))


class Stack(nn.Module):
    """Embedding, N residual blocks, output head. Records per-layer activations."""

    def __init__(self, d_model: int, n_layers: int, vocab: int, logit_mult: float = 1.0) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(Block(d_model) for _ in range(n_layers))
        self.norm = RMSNorm(d_model)
        self.head = mark_output_layer(nn.Linear(d_model, vocab, bias=False))
        self.logit_mult = logit_mult  # muP change (d)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
        x = self.embed(idx)
        coords = [x.abs().mean().item()]
        for block in self.blocks:
            x = block(x)
            coords.append(x.abs().mean().item())
        return self.head(self.norm(x)) * self.logit_mult, coords


def make_param_groups(model: nn.Module, cfg: ModelConfig, lr: float) -> list[dict]:
    """muP change (b): hidden matrices get their learning rate divided by mult.

    Embeddings, norm gains and biases keep the base learning rate, because
    their fan-in does not grow with width.
    """
    mult = width_multiplier(cfg)
    hidden, constant = [], []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Adam column of Table 8: hidden AND output weights both get lr / fan_in
            hidden.append(module.weight)
            if module.bias is not None:
                constant.append(module.bias)
        elif isinstance(module, (nn.Embedding, RMSNorm)):
            constant.extend(p for p in module.parameters(recurse=False) if p is not None)

    if not cfg.mup.enabled:
        return [{"params": hidden + constant, "lr": lr}]
    return [
        {"params": hidden, "lr": lr / mult},
        {"params": constant, "lr": lr},
    ]


def run_width(
    d_model: int, *, mup: bool, n_layers: int, steps: int, vocab: int, lr: float, seed: int
) -> list[list[float]]:
    """Train one width, return the coordinate profile after each step."""
    set_determinism(seed)
    cfg = ModelConfig(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab,
        mup=MuPConfig(enabled=mup, base_d_model=BASE_WIDTH),
        attention=AttentionConfig(scale="mup" if mup else "1/sqrt(d)", n_heads=8, n_kv_heads=2),
        init=InitConfig(scheme="fixed", std=0.02, scaled_residual=True),
    )
    model = Stack(d_model, n_layers, vocab, logit_mult=output_logit_multiplier(cfg))
    init_weights(model, cfg)
    opt = torch.optim.AdamW(make_param_groups(model, cfg, lr), betas=(0.9, 0.95))

    # A learnable task, not random targets. muP predicts width invariance for
    # *correlated* updates, and random labels produce pure gradient noise whose
    # updates cancel, which makes the coordinates shrink as sqrt(width).
    idx = torch.randint(0, vocab, (8, 32))
    targets = (idx + 1) % vocab

    profiles = []
    for _ in range(steps):
        logits, coords = model(idx)
        profiles.append(coords)
        loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return profiles


def plot(results: dict, out_path: Path, steps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = plt.cm.viridis([0.1, 0.4, 0.65, 0.9])

    for ax, mup in zip(axes, (False, True), strict=True):
        for color, width in zip(colors, WIDTHS, strict=True):
            final = results[(width, mup)][-1]
            ax.plot(
                range(len(final)),
                final,
                marker="o",
                markersize=4,
                color=color,
                label=f"d_model = {width}",
            )
        ax.set_title(
            f"{'muP' if mup else 'standard parametrization'}"
            f"{'  (curves should overlap)' if mup else '  (curves spread with width)'}"
        )
        ax.set_xlabel("layer index")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("coordinate size (mean |activation|)")
    axes[1].legend(frameon=False)
    fig.suptitle(f"muP coordinate check after {steps} AdamW steps", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def spread(results: dict, mup: bool) -> float:
    """Max over layers of (max across widths / min across widths). 1.0 is perfect."""
    profiles = [results[(w, mup)][-1] for w in WIDTHS]
    worst = 0.0
    for layer in zip(*profiles, strict=True):
        worst = max(worst, max(layer) / max(min(layer), 1e-12))
    return worst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--vocab", type=int, default=512)
    # 1e-2 pushes this toy into a regime where the linearized muP argument stops
    # holding and the spread triples. See docs/mup_coord_check.md.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("docs/assets"))
    args = p.parse_args()

    results = {}
    jobs = [(w, m) for m in (False, True) for w in WIDTHS]
    for width, mup in tqdm(jobs, desc="coord check", unit="run"):
        results[(width, mup)] = run_width(
            width,
            mup=mup,
            n_layers=args.layers,
            steps=args.steps,
            vocab=args.vocab,
            lr=args.lr,
            seed=args.seed,
        )

    std_spread = spread(results, mup=False)
    mup_spread = spread(results, mup=True)
    print(f"\nwidth spread after {args.steps} steps (1.0 = perfectly width invariant)")
    print(f"  standard parametrization : {std_spread:6.2f}x")
    print(f"  muP                      : {mup_spread:6.2f}x")
    if mup_spread < std_spread:
        print(f"  muP is {std_spread / mup_spread:.1f}x flatter across widths")
    else:
        print("  WARNING: muP did not flatten the profile, the implementation is wrong")

    plot(results, args.out / "mup_coord_check.png", args.steps)


if __name__ == "__main__":
    main()
