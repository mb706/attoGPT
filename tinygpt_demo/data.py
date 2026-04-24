from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from .text import normalize_text
from .tokenizers import (
    DemoTokenizer,
    train_bpe_tokenizer,
    train_char_tokenizer,
    train_word_tokenizer,
)


TINYSTORIES_TRAIN = [
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "data/train-00002-of-00004-a26307300439e943.parquet",
    "data/train-00003-of-00004-d243063613e5a057.parquet",
]
TINYSTORIES_VAL = "data/validation-00000-of-00001-869c898b519ad725.parquet"


def download_tinystories(raw_dir: Path, train_shards: int = 1) -> tuple[list[Path], Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_files = []
    for filename in TINYSTORIES_TRAIN[:train_shards]:
        train_files.append(
            Path(
                hf_hub_download(
                    "roneneldan/TinyStories",
                    filename=filename,
                    repo_type="dataset",
                    local_dir=raw_dir,
                )
            )
        )
    val_file = Path(
        hf_hub_download(
            "roneneldan/TinyStories",
            filename=TINYSTORIES_VAL,
            repo_type="dataset",
            local_dir=raw_dir,
        )
    )
    return train_files, val_file


def iter_parquet_texts(
    paths: list[Path] | tuple[Path, ...],
    mode: str,
    max_docs: int | None = None,
    batch_size: int = 8192,
) -> Iterator[str]:
    seen = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
            for raw in batch.column(0).to_pylist():
                text = normalize_text(raw, mode=mode)
                if text:
                    yield text
                    seen += 1
                    if max_docs is not None and seen >= max_docs:
                        return


def build_tokenizer(
    kind: str,
    train_texts: Iterable[str],
    vocab_size: int,
) -> DemoTokenizer:
    if kind == "char":
        return train_char_tokenizer(train_texts)
    if kind == "word":
        return train_word_tokenizer(train_texts, vocab_size=vocab_size)
    if kind == "bpe":
        return train_bpe_tokenizer(train_texts, vocab_size=vocab_size)
    raise ValueError(f"unknown tokenizer kind: {kind}")


def write_token_file(
    path: Path,
    texts: Iterable[str],
    tokenizer: DemoTokenizer,
    dtype: np.dtype,
) -> dict[str, int | float]:
    num_docs = 0
    num_tokens = 0
    num_bytes = 0
    unk = 0
    with path.open("wb") as f:
        for text in texts:
            ids = tokenizer.encode(text, add_special=True)
            arr = np.asarray(ids, dtype=dtype)
            arr.tofile(f)
            num_docs += 1
            num_tokens += int(arr.size)
            num_bytes += len(text.encode("utf-8"))
            unk += sum(1 for idx in ids if idx == tokenizer.unk_id)
    return {
        "docs": num_docs,
        "tokens": num_tokens,
        "bytes": num_bytes,
        "unk_tokens": unk,
        "unk_rate": float(unk / max(1, num_tokens)),
    }


class MemmapTokens:
    def __init__(self, path: Path, dtype: str):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(path, dtype=self.dtype, mode="r")

    def __len__(self) -> int:
        return int(self.data.shape[0])


def get_batch(
    split: MemmapTokens,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(split) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"dataset too short for block_size={block_size}")
    starts = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([split.data[i : i + block_size] for i in starts]).astype(np.int64)
    y = np.stack([split.data[i + 1 : i + block_size + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def read_meta(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))


def write_meta(data_dir: Path, meta: dict) -> None:
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
