"""Minimal training loop.

Deliberately small. This exists so ablations can be run end to end, not to
compete with a real training stack, and everything it does not do is listed in
the README under "what is not in this repo".

What it does cover, because each one changes results rather than only speed:
mixed precision with fp32 master weights, gradient accumulation, gradient
clipping, the aux-loss-free balancing step, optional activation checkpointing,
and JSONL logging of every loss term separately.

    python -m mt.train --config configs/llama_style_150m.yaml
    python -m mt.train --config configs/base.yaml --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from tqdm import tqdm

from mt.config import Config
from mt.model import Transformer
from mt.optim import build_optimizer, build_scheduler
from mt.utils.numerics import autocast_dtype, pick_device, resolve_precision
from mt.utils.seed import set_determinism


class ByteDataset:
    """Raw bytes of a text file, or deterministic noise when none is given.

    Byte level means no tokenizer, so an ablation measures the architecture
    rather than a vocabulary choice. ``vocab_size`` is 256 by construction.
    """

    VOCAB_SIZE = 256

    def __init__(self, path: Path | None, seq_len: int, device: torch.device) -> None:
        self.seq_len = seq_len
        self.device = device
        if path is not None and path.exists():
            # copy, since frombuffer would alias a read-only bytes object
            data = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
            self.source = f"{path} ({len(data):,} bytes)"
        else:
            # a learnable synthetic task, so the loss curve means something
            g = torch.Generator().manual_seed(0)
            base = torch.randint(0, 64, (1 << 16,), generator=g, dtype=torch.uint8)
            data = (base + torch.arange(len(base)) % 4).to(torch.uint8)
            self.source = "synthetic bytes (no --data given)"
        self.data = data.long()

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,))
        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + 1 + self.seq_len] for s in starts])
        return x.to(self.device), y.to(self.device)


class JSONLLogger:
    """One JSON object per line, so a run can be replotted without parsing logs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("a", encoding="utf-8")

    def log(self, record: dict) -> None:
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def train(cfg: Config, *, data_path: Path | None = None, out_dir: Path | None = None) -> Path:
    """Run the training loop and return the run directory."""
    train_cfg = cfg.train
    set_determinism(train_cfg.seed)

    device = pick_device()
    precision = resolve_precision(train_cfg.precision, device)
    if precision != train_cfg.precision:
        print(
            f"[precision] {train_cfg.precision} would be emulated on this GPU, "
            f"using {precision} instead"
        )
    amp_dtype = autocast_dtype(precision)

    dataset = ByteDataset(data_path, train_cfg.seq_len, device)
    cfg.model.vocab_size = ByteDataset.VOCAB_SIZE
    cfg.model.max_seq_len = max(cfg.model.max_seq_len, train_cfg.seq_len)

    model = Transformer(cfg.model, z_loss_coef=train_cfg.z_loss_coef).to(device)
    model.gradient_checkpointing = train_cfg.activation_checkpointing
    if train_cfg.compile:
        model = torch.compile(model)

    optimizer = build_optimizer(model, cfg.model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    # master weights stay fp32, only the matmuls run reduced
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))

    run_dir = Path(out_dir or train_cfg.out_dir) / time.strftime("run-%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(run_dir / "metrics.jsonl")
    (run_dir / "config.json").write_text(
        json.dumps(cfg.model_dump(), indent=2, default=str), encoding="utf-8"
    )

    print(f"data      : {dataset.source}")
    print(f"device    : {device}, precision {precision}")
    print(f"params    : {model.n_params():,} total, {model.n_active_params():,} active")
    print(f"run dir   : {run_dir}")

    model.train()
    bar = tqdm(range(train_cfg.max_steps), desc="training", unit="step")
    for step in bar:
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}

        for _ in range(train_cfg.grad_accum_steps):
            x, y = dataset.batch(train_cfg.micro_batch_size)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss, aux = model(x, y)
            scaler.scale(loss / train_cfg.grad_accum_steps).backward()
            for key, value in aux.as_dict().items():
                totals[key] = totals.get(key, 0.0) + value / train_cfg.grad_accum_steps

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # the aux-loss-free bias moves once per optimizer step, not per micro batch
        model.balance_step()

        if step % train_cfg.log_interval == 0 or step == train_cfg.max_steps - 1:
            record = {
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "step_time_s": time.perf_counter() - t0,
                "tokens": (step + 1)
                * train_cfg.micro_batch_size
                * train_cfg.grad_accum_steps
                * train_cfg.seq_len,
                **totals,
            }
            for i, metrics in enumerate(model.routing_metrics()):
                record.update({f"{k}/layer{i}": v for k, v in metrics.as_dict().items()})
            logger.log(record)
            bar.set_postfix(loss=f"{totals.get('loss/ce', 0.0):.3f}", lr=f"{record['lr']:.2e}")

        if train_cfg.ckpt_interval and (step + 1) % train_cfg.ckpt_interval == 0:
            save_checkpoint(model, optimizer, step, cfg, run_dir / "ckpt.pt")

    save_checkpoint(model, optimizer, train_cfg.max_steps - 1, cfg, run_dir / "ckpt.pt")
    logger.close()
    print(f"\ndone, metrics in {run_dir / 'metrics.jsonl'}")
    return run_dir


def save_checkpoint(model, optimizer, step: int, cfg: Config, path: Path) -> None:
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.model_dump(),
        },
        path,
    )


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply ``--set model.moe.n_experts=8`` style overrides.

    Re-validates through the schema afterwards, so an override cannot smuggle
    in an inconsistent config that the YAML files are checked against.
    """
    if not overrides:
        return cfg
    raw = cfg.model_dump()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = raw
        *parents, leaf = key.split(".")
        for part in parents:
            if part not in node:
                raise ValueError(f"unknown config section {part!r} in {key!r}")
            node = node[part]
        if leaf not in node:
            raise ValueError(f"unknown config field {leaf!r} in {key!r}")
        node[leaf] = yaml.safe_load(value)  # gives int, float, bool or str
    return Config.model_validate(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--data", type=Path, default=None, help="raw text or binary file")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config field, e.g. --set model.d_model=128",
    )
    args = p.parse_args()

    cfg = apply_overrides(Config.from_yaml(args.config), args.overrides)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
        cfg.train.warmup_steps = min(cfg.train.warmup_steps, max(args.max_steps // 10, 1))
    if args.seq_len is not None:
        cfg.train.seq_len = args.seq_len
    if args.batch_size is not None:
        cfg.train.micro_batch_size = args.batch_size

    train(cfg, data_path=args.data, out_dir=args.out)


if __name__ == "__main__":
    main()
