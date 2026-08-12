"""Generate text from a trained checkpoint.

    python scripts/generate.py --ckpt outputs/models/<run>/ckpt.pt \\
        --prompt "The capital of France is" --tokens 60

Without ``--prompt`` it runs one English and one French prompt, which is the
quickest way to see whether a bilingual run learned both sides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mt.config import Config
from mt.generate import SamplingConfig, generate
from mt.model import Transformer
from mt.utils.numerics import pick_device

DEFAULT_PROMPTS = [
    ("en", "The history of the city is"),
    ("fr", "L'histoire de la ville est"),
]


def load(ckpt_path: Path, device) -> tuple[Transformer, Config]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config.model_validate(ckpt["config"])
    model = Transformer(cfg.model).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"checkpoint : {ckpt_path} at step {ckpt['step']:,}")
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default="croissantllm/CroissantLLMBase")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoTokenizer

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model, _ = load(args.ckpt, device)
    print(f"device     : {device}, {model.n_params():,} parameters\n")

    cfg = SamplingConfig(temperature=args.temperature, top_p=args.top_p)
    # the generator must live on the same device as the logits, torch refuses
    # to mix them rather than moving one silently
    gen = torch.Generator(device=device).manual_seed(args.seed)
    prompts = (
        [("custom", args.prompt)] if args.prompt else DEFAULT_PROMPTS
    )

    for lang, prompt in prompts:
        # not return_tensors="pt": transformers 5.x disables its torch
        # integration below torch 2.5, and a pod may well run 2.4
        token_ids = tokenizer(prompt)["input_ids"]
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        out = generate(model, ids, args.tokens, cfg, generator=gen)
        # a plain list, for the same reason: the tokenizer has no torch bridge
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"[{lang}] {prompt!r}")
        print(f"  -> {text}\n")


if __name__ == "__main__":
    main()
