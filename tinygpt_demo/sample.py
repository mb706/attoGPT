from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import GPTConfig, TinyGPT
from .text import normalize_text
from .tokenizers import load_tokenizer_for_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    tokenizer = load_tokenizer_for_checkpoint(args.checkpoint, ckpt)
    config = GPTConfig(**ckpt["config"])
    model = TinyGPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    prompt = normalize_text(args.prompt, mode=ckpt["meta"]["text_mode"])
    ids = tokenizer.encode(prompt, add_special=True)[:-1]
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    y = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        bad_token_ids=tokenizer.bad_token_ids,
    )
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
