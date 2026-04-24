from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data import MemmapTokens, read_meta
from .model import GPTConfig, TinyGPT


@torch.no_grad()
def eval_split(
    model: TinyGPT,
    data: MemmapTokens,
    block_size: int,
    batch_size: int,
    device: str,
    max_batches: int | None = None,
) -> tuple[float, int]:
    model.eval()
    starts = np.arange(0, len(data) - block_size - 1, block_size, dtype=np.int64)
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        x_np = np.stack([data.data[i : i + block_size] for i in batch_starts]).astype(np.int64)
        y_np = np.stack([data.data[i + 1 : i + block_size + 1] for i in batch_starts]).astype(
            np.int64
        )
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        _, loss, _ = model(x, y)
        tokens = int(y.numel())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return total_loss / total_tokens, total_tokens


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = GPTConfig(**ckpt["config"])
    model = TinyGPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    data_dir = Path(ckpt["tokenizer_dir"])
    meta = read_meta(data_dir)
    split = MemmapTokens(data_dir / f"{args.split}.bin", dtype=meta["dtype"])
    loss, tokens = eval_split(
        model,
        split,
        config.block_size,
        args.batch_size,
        args.device,
        max_batches=args.max_batches,
    )
    token_per_byte = meta[args.split]["tokens"] / max(1, meta[args.split]["bytes"])
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "loss": loss,
        "bpb": loss / math.log(2) * token_per_byte,
        "tokens_evaluated": tokens,
        "block_size": config.block_size,
        "batch_size": args.batch_size,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
