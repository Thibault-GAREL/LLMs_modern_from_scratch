"""Download, tokenize and mix a bilingual corpus into flat token files.

Produces the layout ``mt.data.TokenDataset`` expects::

    <out>/train.bin   token ids as uint16
    <out>/val.bin
    <out>/meta.json   vocab size, tokenizer, per-language counts, mixture

Defaults target a French and English model:

  **English** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  Common Crawl filtered by an educational-quality classifier. The best public
  filtering for a small model.

  **French** [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
  `fra_Latn`, the same pipeline with per-language filters and stopwords.

  **Tokenizer** [CroissantLLM](https://huggingface.co/croissantllm/CroissantLLMBase),
  32k, trained for a 1:1 French-English model. An English-centric tokenizer
  spends 30 to 50% more tokens on the same French text, which is 30 to 50% more
  compute for the same content, so this choice is not cosmetic.

Run it on the pod rather than at home: the download dominates, and a data
centre link is far faster than a domestic one.

    python scripts/prepare_data.py --out data/bilingual --tokens 6e9 --en-ratio 0.7
    python scripts/prepare_data.py --out data/tiny --tokens 2e6   # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

DTYPE = np.uint16  # any vocabulary below 65536 fits, and it halves the file size

SOURCES = {
    "en": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
    },
    "fr": {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "fra_Latn",
        "split": "train",
    },
}


def build_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.vocab_size >= 2**16:
        raise SystemExit(
            f"{name} has a {tok.vocab_size} vocabulary, which does not fit in uint16. "
            "Pick a smaller tokenizer or change DTYPE."
        )
    return tok


def stream_tokens(tokenizer, source: dict, budget: int, label: str):
    """Yield token id arrays from a streamed dataset until ``budget`` is met.

    Streaming rather than downloading the whole split: FineWeb-Edu alone is
    28 GB and we only need a slice of it.
    """
    from datasets import load_dataset

    ds = load_dataset(
        source["path"], name=source["name"], split=source["split"], streaming=True
    )
    eos = tokenizer.eos_token_id
    if eos is None:
        raise SystemExit(f"{tokenizer.name_or_path} has no eos token, cannot pack documents")

    produced = 0
    bar = tqdm(total=budget, unit="tok", unit_scale=True, desc=f"{label:<8}", leave=True)
    for doc in ds:
        text = doc.get("text")
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(eos)  # a document boundary the model can learn to respect
        arr = np.asarray(ids, dtype=DTYPE)
        produced += len(arr)
        bar.update(len(arr))
        yield arr
        if produced >= budget:
            break
    bar.close()
    if produced < budget * 0.98:
        print(
            f"[warning] {label}: only {produced:,} tokens available, asked for {budget:,}"
        )


def write_split(
    tokenizer, budgets: dict[str, int], out_dir: Path, split: str, seed: int
) -> dict[str, int]:
    """Interleave the languages document by document and write one ``.bin``.

    Interleaving matters. Concatenating English then French would give the
    model a curriculum it never asked for, and the loss curve would show a
    cliff at the boundary rather than a mixture.
    """
    rng = np.random.default_rng(seed)
    streams = {
        lang: stream_tokens(tokenizer, SOURCES[lang], budget, f"{split}/{lang}")
        for lang, budget in budgets.items()
        if budget > 0
    }
    remaining = dict(budgets)
    counts = {lang: 0 for lang in streams}

    path = out_dir / f"{split}.bin"
    with path.open("wb") as fh:
        while streams:
            total = sum(max(remaining[lang], 0) for lang in streams)
            if total <= 0:
                break
            langs = list(streams)
            weights = np.array([max(remaining[lang], 0) for lang in langs], dtype=float)
            lang = langs[int(rng.choice(len(langs), p=weights / weights.sum()))]
            try:
                arr = next(streams[lang])
            except StopIteration:
                streams.pop(lang)
                continue
            fh.write(arr.tobytes())
            counts[lang] += len(arr)
            remaining[lang] -= len(arr)
            if remaining[lang] <= 0:
                streams.pop(lang)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--tokens", type=float, default=6e9, help="total training tokens to produce"
    )
    p.add_argument(
        "--en-ratio", type=float, default=0.7, help="share of English, the rest is French"
    )
    p.add_argument("--val-tokens", type=float, default=5e6)
    p.add_argument("--tokenizer", type=str, default="croissantllm/CroissantLLMBase")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not 0.0 <= args.en_ratio <= 1.0:
        raise SystemExit("--en-ratio must lie in [0, 1]")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(args.tokenizer)

    total, val_total = int(args.tokens), int(args.val_tokens)
    plan = {
        "train": {
            "en": int(total * args.en_ratio),
            "fr": total - int(total * args.en_ratio),
        },
        "val": {
            "en": int(val_total * args.en_ratio),
            "fr": val_total - int(val_total * args.en_ratio),
        },
    }

    print(f"tokenizer : {args.tokenizer}, vocab {tokenizer.vocab_size:,}")
    print(f"mixture   : {args.en_ratio:.0%} English, {1 - args.en_ratio:.0%} French")
    for split, budgets in plan.items():
        print(f"{split:<9} : " + ", ".join(f"{k} {v:,}" for k, v in budgets.items()))
    print(f"output    : {out.resolve()}\n")

    t0 = time.perf_counter()
    counts = {
        split: write_split(tokenizer, budgets, out, split, args.seed + i)
        for i, (split, budgets) in enumerate(plan.items())
    }

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "dtype": str(np.dtype(DTYPE)),
        "tokenizer": args.tokenizer,
        "mixture": {"en": args.en_ratio, "fr": round(1 - args.en_ratio, 4)},
        "sources": SOURCES,
        "counts": counts,
        "seed": args.seed,
        "prepared_s": round(time.perf_counter() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\ndone")
    for split, c in counts.items():
        size_mb = (out / f"{split}.bin").stat().st_size / 2**20
        print(f"  {split:<6} {sum(c.values()):>14,} tokens  {size_mb:>9,.0f} MiB  {c}")
    print(f"  meta   {out / 'meta.json'}")


if __name__ == "__main__":
    main()
