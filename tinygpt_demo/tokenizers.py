from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol

from tokenizers import Tokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

from .text import word_tokens


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class DemoTokenizer(Protocol):
    kind: str
    stoi: dict[str, int]
    itos: list[str]

    @property
    def vocab_size(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def unk_id(self) -> int: ...

    @property
    def bad_token_ids(self) -> list[int]: ...

    def encode(self, text: str, add_special: bool = True) -> list[int]: ...

    def decode(self, ids: list[int], skip_special: bool = True) -> str: ...

    def token_str(self, idx: int) -> str: ...

    def save(self, out_dir: Path) -> None: ...


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: list[str] | None = None):
        if chars is None:
            chars = list("\n !,.?abcdefghijklmnopqrstuvwxyz")
        self.itos = SPECIAL_TOKENS + [c for c in chars if c not in SPECIAL_TOKENS]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def bad_token_ids(self) -> list[int]:
        return [self.pad_id, self.unk_id]

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)

    def token_str(self, idx: int) -> str:
        token = self.itos[int(idx)]
        if token == "\n":
            return "\\n"
        if token == " ":
            return "<space>"
        return token

    def save(self, out_dir: Path) -> None:
        (out_dir / "tokenizer.json").write_text(
            json.dumps({"kind": self.kind, "itos": self.itos}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_file(cls, path: Path) -> "CharTokenizer":
        meta = json.loads(path.read_text(encoding="utf-8"))
        tok = cls([])
        tok.itos = meta["itos"]
        tok.stoi = {s: i for i, s in enumerate(tok.itos)}
        return tok


class WordTokenizer:
    kind = "word"

    def __init__(self, words: list[str]):
        punct = [".", ",", "!", "?"]
        vocab = SPECIAL_TOKENS + punct + [w for w in words if w not in punct and w not in SPECIAL_TOKENS]
        self.itos = vocab
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def bad_token_ids(self) -> list[int]:
        return [self.pad_id, self.unk_id, self.bos_id]

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(tok, self.unk_id) for tok in word_tokens(text)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out = ""
        for idx in ids:
            token = self.itos[int(idx)]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            if token in {".", ",", "!", "?"}:
                out = out.rstrip() + token + " "
            else:
                out += token + " "
        return out.strip()

    def token_str(self, idx: int) -> str:
        return self.itos[int(idx)]

    def save(self, out_dir: Path) -> None:
        (out_dir / "tokenizer.json").write_text(
            json.dumps({"kind": self.kind, "itos": self.itos}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_file(cls, path: Path) -> "WordTokenizer":
        meta = json.loads(path.read_text(encoding="utf-8"))
        tok = cls([])
        tok.itos = meta["itos"]
        tok.stoi = {s: i for i, s in enumerate(tok.itos)}
        return tok


class BPETokenizer:
    kind = "bpe"

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        self.itos = [""] * len(vocab)
        for token, idx in vocab.items():
            self.itos[idx] = token
        self.stoi = vocab

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id("<pad>")

    @property
    def bos_id(self) -> int:
        return self.tokenizer.token_to_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id("<eos>")

    @property
    def unk_id(self) -> int:
        return self.tokenizer.token_to_id("<unk>")

    @property
    def bad_token_ids(self) -> list[int]:
        return [self.pad_id, self.unk_id, self.bos_id]

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def token_str(self, idx: int) -> str:
        token = self.itos[int(idx)]
        return token.replace("Ġ", "▁")

    def save(self, out_dir: Path) -> None:
        self.tokenizer.save(str(out_dir / "tokenizer.bpe.json"))
        (out_dir / "tokenizer.json").write_text(
            json.dumps({"kind": self.kind, "file": "tokenizer.bpe.json"}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dir(cls, out_dir: Path) -> "BPETokenizer":
        tokenizer = Tokenizer.from_file(str(out_dir / "tokenizer.bpe.json"))
        return cls(tokenizer)


def train_word_tokenizer(texts: Iterable[str], vocab_size: int) -> WordTokenizer:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tok for tok in word_tokens(text) if tok not in {".", ",", "!", "?"})
    max_words = max(0, vocab_size - len(SPECIAL_TOKENS) - 4)
    words = [word for word, _ in counter.most_common(max_words)]
    return WordTokenizer(words)


def train_char_tokenizer(texts: Iterable[str]) -> CharTokenizer:
    chars: set[str] = set()
    for text in texts:
        chars.update(text)
    ordered = sorted(chars)
    return CharTokenizer(ordered)


def train_bpe_tokenizer(texts: Iterable[str], vocab_size: int) -> BPETokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return BPETokenizer(tokenizer)


def load_tokenizer(data_dir: str | Path) -> DemoTokenizer:
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "tokenizer.json").read_text(encoding="utf-8"))
    kind = meta["kind"]
    if kind == "char":
        return CharTokenizer.from_file(data_dir / "tokenizer.json")
    if kind == "word":
        return WordTokenizer.from_file(data_dir / "tokenizer.json")
    if kind == "bpe":
        return BPETokenizer.from_dir(data_dir)
    raise ValueError(f"unknown tokenizer kind: {kind}")


def load_tokenizer_for_checkpoint(checkpoint_path: str | Path, ckpt: dict) -> DemoTokenizer:
    """Load tokenizer from checkpoint metadata, falling back to the checkpoint directory."""
    checkpoint_path = Path(checkpoint_path)
    candidates: list[Path] = []
    if ckpt.get("tokenizer_dir"):
        candidates.append(Path(ckpt["tokenizer_dir"]))
    candidates.append(checkpoint_path.resolve().parent)
    for candidate in candidates:
        if (candidate / "tokenizer.json").exists():
            return load_tokenizer(candidate)
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"could not find tokenizer.json; searched: {searched}")
