"""The bilingual mixing logic, checked without touching the network.

``prepare_data.py`` runs on a rented pod, so a bug in it costs billed time.
The streaming and the tokenizer are faked here, which leaves exactly the part
worth testing: whether the languages come out at the requested ratio and
interleaved rather than concatenated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

prepare_data = pytest.importorskip("prepare_data")


@pytest.fixture
def fake_streams(monkeypatch):
    """Replace the dataset stream by deterministic per-language token blocks.

    English documents are all 1s, French all 2s, so the written file can be
    read back and attributed to a language token by token.
    """
    marker = {"en": 1, "fr": 2}

    def fake_stream(tokenizer, source, budget, label):
        lang = label.split("/")[-1]
        produced = 0
        while produced < budget:
            n = 50
            produced += n
            yield np.full(n, marker[lang], dtype=prepare_data.DTYPE)

    monkeypatch.setattr(prepare_data, "stream_tokens", fake_stream)
    return marker


def read_tokens(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=prepare_data.DTYPE)


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("en_ratio", [0.7, 0.5, 0.3])
def test_mixture_lands_near_the_requested_ratio(tmp_path, fake_streams, en_ratio):
    total = 20_000
    budgets = {"en": int(total * en_ratio), "fr": total - int(total * en_ratio)}
    counts = prepare_data.write_split(None, budgets, tmp_path, "train", seed=0)

    produced = sum(counts.values())
    actual = counts["en"] / produced
    assert abs(actual - en_ratio) < 0.05, (
        f"asked for {en_ratio:.0%} English, wrote {actual:.1%}"
    )


def test_languages_are_interleaved_not_concatenated(tmp_path, fake_streams):
    """Concatenating would give the model a curriculum nobody asked for."""
    prepare_data.write_split(None, {"en": 10_000, "fr": 10_000}, tmp_path, "train", 0)
    tokens = read_tokens(tmp_path / "train.bin")

    # count how many times the language flips along the file
    langs = tokens[::50]  # one sample per emitted document
    switches = int(np.sum(langs[1:] != langs[:-1]))
    assert switches > 50, f"only {switches} language switches, that is a concatenation"


def test_both_languages_are_present(tmp_path, fake_streams):
    counts = prepare_data.write_split(None, {"en": 5_000, "fr": 5_000}, tmp_path, "t", 0)
    tokens = read_tokens(tmp_path / "t.bin")
    assert set(np.unique(tokens)) == {1, 2}
    assert counts["en"] > 0 and counts["fr"] > 0


def test_a_zero_budget_drops_the_language(tmp_path, fake_streams):
    """An English-only run must not emit a single French token."""
    counts = prepare_data.write_split(None, {"en": 5_000, "fr": 0}, tmp_path, "t", 0)
    assert counts == {"en": pytest.approx(5_000, abs=100)}
    assert set(np.unique(read_tokens(tmp_path / "t.bin"))) == {1}


def test_output_size_matches_the_budget(tmp_path, fake_streams):
    budgets = {"en": 7_000, "fr": 3_000}
    prepare_data.write_split(None, budgets, tmp_path, "train", 0)
    written = len(read_tokens(tmp_path / "train.bin"))
    # documents are emitted whole, so the last one can overshoot slightly
    assert 10_000 <= written <= 10_000 + 100


def test_the_split_is_reproducible(tmp_path, fake_streams):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    prepare_data.write_split(None, {"en": 5_000, "fr": 5_000}, a, "t", seed=3)
    prepare_data.write_split(None, {"en": 5_000, "fr": 5_000}, b, "t", seed=3)
    np.testing.assert_array_equal(read_tokens(a / "t.bin"), read_tokens(b / "t.bin"))


def test_a_different_seed_gives_a_different_interleaving(tmp_path, fake_streams):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    prepare_data.write_split(None, {"en": 5_000, "fr": 5_000}, a, "t", seed=1)
    prepare_data.write_split(None, {"en": 5_000, "fr": 5_000}, b, "t", seed=2)
    assert not np.array_equal(read_tokens(a / "t.bin"), read_tokens(b / "t.bin"))


# ---------------------------------------------------------------------------
# The output is what TokenDataset expects
# ---------------------------------------------------------------------------


def test_written_file_loads_as_a_dataset(tmp_path, fake_streams):
    """The two halves of the pipeline must actually fit together."""
    import torch

    from mt.data import TokenDataset

    prepare_data.write_split(None, {"en": 7_000, "fr": 3_000}, tmp_path, "train", 0)
    (tmp_path / "meta.json").write_text(
        json.dumps({"vocab_size": 32000, "dtype": "uint16", "mixture": {"en": 0.7}}),
        encoding="utf-8",
    )
    ds = TokenDataset(tmp_path, "train", seq_len=64, device=torch.device("cpu"))
    x, y = ds.batch(2)
    assert x.shape == (2, 64)
    assert set(int(v) for v in x.unique()) <= {1, 2}


def test_dtype_holds_the_target_vocabulary():
    """uint16 covers CroissantLLM's 32k, and prepare_data refuses anything larger."""
    assert np.iinfo(prepare_data.DTYPE).max >= 32_000
    assert np.dtype(prepare_data.DTYPE).itemsize == 2


def test_sources_point_at_the_expected_datasets():
    en, fr = prepare_data.SOURCES["en"], prepare_data.SOURCES["fr"]
    assert "fineweb-edu" in en["path"]
    assert "fineweb-2" in fr["path"] and fr["name"] == "fra_Latn"
