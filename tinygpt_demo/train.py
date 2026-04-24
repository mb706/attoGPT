from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import MemmapTokens, get_batch, read_meta
from .model import GPTConfig, TinyGPT
from .optim import make_optimizers
from .tokenizers import load_tokenizer


def cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_ratio)))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_data: MemmapTokens,
    val_data: MemmapTokens,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    out = {}
    for name, split in [("train", train_data), ("val", val_data)]:
        losses = torch.empty(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, device)
            _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[name] = float(losses.mean().item())
    model.train()
    return out


def train_once(args: argparse.Namespace, trial=None) -> dict[str, float | str | int]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = args.device

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(data_dir)
    meta = read_meta(data_dir)
    dtype = meta["dtype"]
    train_data = MemmapTokens(data_dir / "train.bin", dtype=dtype)
    val_data = MemmapTokens(data_dir / "val.bin", dtype=dtype)
    val_token_per_byte = meta["val"]["tokens"] / max(1, meta["val"]["bytes"])

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        norm=args.norm,
        activation=args.activation,
        pos=args.pos,
        topk_attn=args.topk_attn,
        qk_norm=args.qk_norm,
        tie_weights=not args.untie_weights,
    )
    model = TinyGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizers = make_optimizers(
        model,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
    )

    config_record = {
        "model": config.to_dict(),
        "params": n_params,
        "data_dir": str(data_dir),
        "meta": meta,
        "train_args": vars(args),
    }
    (out_dir / "config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")

    best_val = float("inf")
    best_step = 0
    start = time.time()
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "train_loss",
                "val_loss",
                "val_bpb",
                "lr",
                "dt_sec",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()

        for step in range(args.max_steps + 1):
            if step % args.eval_interval == 0 or step == args.max_steps:
                losses = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    args.block_size,
                    args.eval_batch_size,
                    args.eval_iters,
                    device,
                )
                val_bpb = losses["val"] / math.log(2) * val_token_per_byte
                dt = time.time() - start
                row = {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "val_bpb": val_bpb,
                    "lr": optimizers[-1].param_groups[0]["lr"],
                    "dt_sec": dt,
                    "tokens_per_sec": (step * args.batch_size * args.block_size) / max(dt, 1e-9),
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"step {step:5d} train {losses['train']:.4f} "
                    f"val {losses['val']:.4f} bpb {val_bpb:.4f} "
                    f"tok/s {row['tokens_per_sec']:.0f}",
                    flush=True,
                )
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    best_step = step
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": config.to_dict(),
                            "tokenizer_dir": str(data_dir),
                            "meta": meta,
                            "step": step,
                            "val_loss": best_val,
                            "val_bpb": val_bpb,
                            "params": n_params,
                        },
                        out_dir / "best.pt",
                    )
                if trial is not None:
                    trial.report(val_bpb, step)
                    if trial.should_prune():
                        raise RuntimeError("TRIAL_PRUNED")

            if step == args.max_steps:
                break

            lr_now = cosine_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup_steps)
            for opt in optimizers:
                for group in opt.param_groups:
                    if isinstance(opt, torch.optim.AdamW):
                        group["lr"] = lr_now
                    else:
                        group["lr"] = args.muon_lr * (lr_now / max(args.lr, 1e-12))

            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            _, loss, _ = model(x, y)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            for opt in optimizers:
                opt.step()

    final = {
        "best_val_loss": best_val,
        "best_val_bpb": best_val / math.log(2) * val_token_per_byte,
        "best_step": best_step,
        "params": n_params,
        "out_dir": str(out_dir),
        "elapsed_sec": time.time() - start,
    }
    (out_dir / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--mlp-mult", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--norm", choices=["rmsnorm", "layernorm"], default="rmsnorm")
    p.add_argument("--activation", choices=["swiglu", "gelu", "relu2"], default="swiglu")
    p.add_argument("--pos", choices=["rope", "learned"], default="rope")
    p.add_argument("--topk-attn", type=int, default=0)
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--untie-weights", action="store_true")
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    try:
        train_once(args)
    except RuntimeError as exc:
        if str(exc) == "TRIAL_PRUNED":
            raise
        raise


if __name__ == "__main__":
    main()
