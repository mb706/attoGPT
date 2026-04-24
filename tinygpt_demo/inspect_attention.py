from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .model import GPTConfig, TinyGPT
from .text import normalize_text
from .tokenizers import load_tokenizer_for_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="once upon a time")
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--topn", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    tokenizer = load_tokenizer_for_checkpoint(args.checkpoint, ckpt)
    config = GPTConfig(**ckpt["config"])
    model = TinyGPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = normalize_text(args.prompt, mode=ckpt["meta"]["text_mode"])
    ids = tokenizer.encode(prompt, add_special=True)[:-1]
    ids = ids[-config.block_size :]
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    with torch.no_grad():
        logits, _, att = model(x, return_attn=True)
        probs = F.softmax(logits[0, args.position], dim=-1)
        pred_probs, pred_ids = torch.topk(probs, args.topn)

    tokens = [tokenizer.token_str(i) for i in ids]
    pos = args.position if args.position >= 0 else len(ids) + args.position
    print(f"prompt tokens ({len(tokens)}): {tokens}")
    print(f"inspecting position {pos}: {tokens[pos]}")
    print("top next-token predictions:")
    for prob, idx in zip(pred_probs.tolist(), pred_ids.tolist()):
        print(f"  {tokenizer.token_str(idx)!r}: {prob:.4f}")
    print("attention by head:")
    assert att is not None
    for head in range(att.size(1)):
        weights = att[0, head, pos]
        vals, idx = torch.topk(weights, min(args.topn, weights.numel()))
        parts = [
            f"{int(i)}:{tokens[int(i)]!r}={float(v):.3f}"
            for v, i in zip(vals.tolist(), idx.tolist())
            if float(v) > 0
        ]
        print(f"  head {head}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
