from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import (
    build_tokenizer,
    download_tinystories,
    iter_parquet_texts,
    write_meta,
    write_token_file,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--raw-dir", default="raw_data/hf")
    p.add_argument("--train-shards", type=int, default=1)
    p.add_argument("--max-train-docs", type=int, default=60000)
    p.add_argument("--max-val-docs", type=int, default=10000)
    p.add_argument("--tokenizer", choices=["char", "word", "bpe"], default="word")
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--text-mode", choices=["simple", "period_only", "apostrophe"], default="simple")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_files, val_file = download_tinystories(Path(args.raw_dir), train_shards=args.train_shards)

    def train_iter():
        return iter_parquet_texts(
            train_files,
            mode=args.text_mode,
            max_docs=args.max_train_docs,
        )

    tokenizer = build_tokenizer(args.tokenizer, train_iter(), vocab_size=args.vocab_size)
    tokenizer.save(out_dir)
    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    train_stats = write_token_file(out_dir / "train.bin", train_iter(), tokenizer, dtype=dtype)
    val_stats = write_token_file(
        out_dir / "val.bin",
        iter_parquet_texts([val_file], mode=args.text_mode, max_docs=args.max_val_docs),
        tokenizer,
        dtype=dtype,
    )
    meta = {
        "source": "roneneldan/TinyStories",
        "train_files": [str(p) for p in train_files],
        "val_file": str(val_file),
        "train_shards": args.train_shards,
        "max_train_docs": args.max_train_docs,
        "max_val_docs": args.max_val_docs,
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "requested_vocab_size": args.vocab_size,
        "text_mode": args.text_mode,
        "dtype": np.dtype(dtype).name,
        "train": train_stats,
        "val": val_stats,
    }
    write_meta(out_dir, meta)
    print(meta)


if __name__ == "__main__":
    main()
