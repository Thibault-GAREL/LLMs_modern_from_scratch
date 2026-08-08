"""Training loop (M5). Kept to a CPU-sized smoke test."""

from __future__ import annotations

import json

import pytest
import torch

from mt.config import AttentionConfig, Config, FFNConfig, ModelConfig, TrainConfig
from mt.train import ByteDataset, apply_overrides, train
from mt.utils.seed import set_determinism


@pytest.fixture(autouse=True)
def _seed():
    set_determinism(0)


def tiny_config(**train_kw) -> Config:
    train_cfg = {
        "max_steps": 12,
        "warmup_steps": 2,
        "seq_len": 16,
        "micro_batch_size": 4,
        "precision": "fp32",
        "log_interval": 4,
        "ckpt_interval": 1000,
        "lr": 3e-3,
    }
    train_cfg.update(train_kw)
    return Config(
        model=ModelConfig(
            d_model=32,
            n_layers=2,
            vocab_size=256,
            max_seq_len=32,
            attention=AttentionConfig(kind="gqa", n_heads=4, n_kv_heads=2),
            ffn=FFNConfig(kind="swiglu", multiple_of=1),
        ),
        train=TrainConfig(**train_cfg),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_byte_dataset_shapes_and_offset():
    ds = ByteDataset(None, seq_len=16, device=torch.device("cpu"))
    x, y = ds.batch(4)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert x.max() < ByteDataset.VOCAB_SIZE
    # targets are the inputs shifted by one
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_byte_dataset_reads_a_file(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"modern transformer " * 200)
    ds = ByteDataset(path, seq_len=8, device=torch.device("cpu"))
    assert "corpus.txt" in ds.source
    x, _ = ds.batch(2)
    assert x.shape == (2, 8)


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------


def test_override_sets_a_nested_field():
    cfg = apply_overrides(tiny_config(), ["model.d_model=64", "train.lr=0.001"])
    assert cfg.model.d_model == 64
    assert cfg.train.lr == 0.001


def test_override_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown config field"):
        apply_overrides(tiny_config(), ["model.d_modle=64"])
    with pytest.raises(ValueError, match="unknown config section"):
        apply_overrides(tiny_config(), ["modle.d_model=64"])
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(tiny_config(), ["model.d_model"])


def test_override_is_revalidated_by_the_schema():
    """An override must not be able to build an inconsistent config."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        apply_overrides(tiny_config(), ["model.attention.n_kv_heads=3"])


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_training_reduces_the_loss(tmp_path):
    run_dir = train(tiny_config(), out_dir=tmp_path)
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(records) >= 3
    assert records[-1]["loss/ce"] < records[0]["loss/ce"]


def test_metrics_contain_every_expected_field(tmp_path):
    run_dir = train(tiny_config(), out_dir=tmp_path)
    first = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert {"step", "lr", "grad_norm", "step_time_s", "tokens", "loss/ce"} <= set(first)


def test_gradient_accumulation_matches_a_larger_batch(tmp_path):
    """Accumulating must be equivalent to one bigger batch, not merely similar."""
    from mt.model import Transformer
    from mt.optim import build_optimizer

    cfg = tiny_config()
    set_determinism(1)
    model_a = Transformer(cfg.model)
    set_determinism(1)
    model_b = Transformer(cfg.model)

    x = torch.randint(0, 256, (8, 16))
    y = torch.randint(0, 256, (8, 16))

    _, loss, _ = model_a(x, y)
    loss.backward()

    for chunk in range(2):
        sl = slice(chunk * 4, (chunk + 1) * 4)
        _, part, _ = model_b(x[sl], y[sl])
        (part / 2).backward()

    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-4, atol=1e-6)
    build_optimizer(model_a, cfg.model, cfg.train)  # must not raise


def test_checkpoint_round_trips(tmp_path):
    cfg = tiny_config(ckpt_interval=1000)
    run_dir = train(cfg, out_dir=tmp_path)
    ckpt = torch.load(run_dir / "ckpt.pt", weights_only=False)
    assert ckpt["step"] == cfg.train.max_steps - 1
    assert "model" in ckpt and "optimizer" in ckpt

    restored = Config.model_validate(ckpt["config"])
    from mt.model import Transformer

    model = Transformer(restored.model)
    model.load_state_dict(ckpt["model"])


def test_config_is_saved_next_to_the_metrics(tmp_path):
    run_dir = train(tiny_config(), out_dir=tmp_path)
    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["model"]["d_model"] == 32


def test_wsd_schedule_runs_end_to_end(tmp_path):
    cfg = tiny_config(schedule="wsd", decay_steps=4)
    run_dir = train(cfg, out_dir=tmp_path)
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    lrs = [r["lr"] for r in records]
    assert lrs[-1] < max(lrs), "the decay phase must lower the learning rate"


def test_activation_checkpointing_runs(tmp_path):
    cfg = tiny_config(activation_checkpointing=True)
    run_dir = train(cfg, out_dir=tmp_path)
    assert (run_dir / "metrics.jsonl").exists()
