from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch


TENSOR_NAMES = [
    "tok_emb.weight",
    "pos_emb.weight",
    "block.ln1.weight",
    "block.attn.qkv.weight",
    "block.attn.proj.weight",
    "block.ln2.weight",
    "block.mlp.fc.weight",
    "block.mlp.proj.weight",
    "ln_f.weight",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/final/best.pt")
    parser.add_argument("--tokenizer", default="artifacts/final/tokenizer.json")
    parser.add_argument("--config", default="artifacts/final/config.json")
    parser.add_argument("--summary", default="artifacts/final/summary.json")
    parser.add_argument("--eval", default="artifacts/final/eval_val_full.json")
    parser.add_argument("--out-dir", default="docs/model")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"]

    manifest = {
        "format": "tiny-gpt-web-f32-v1",
        "source_checkpoint": args.checkpoint,
        "config": ckpt["config"],
        "checkpoint": {
            "step": ckpt.get("step"),
            "val_loss": ckpt.get("val_loss"),
            "val_bpb": ckpt.get("val_bpb"),
            "params": ckpt.get("params"),
        },
        "dtype": "float32-le",
        "tensors": {},
    }

    offset_floats = 0
    bin_path = out_dir / "model.bin"
    sha = hashlib.sha256()
    with bin_path.open("wb") as f:
        for name in TENSOR_NAMES:
            tensor = state[name].detach().cpu().contiguous()
            arr = tensor.numpy().astype("<f4", copy=False)
            payload = arr.tobytes(order="C")
            f.write(payload)
            sha.update(payload)
            manifest["tensors"][name] = {
                "shape": list(arr.shape),
                "offset": offset_floats,
                "length": int(arr.size),
            }
            offset_floats += int(arr.size)

    manifest["num_floats"] = offset_floats
    manifest["num_bytes"] = offset_floats * 4
    manifest["sha256"] = sha.hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    shutil.copyfile(args.tokenizer, out_dir / "tokenizer.json")
    shutil.copyfile(args.config, out_dir / "training_config.json")
    shutil.copyfile(args.summary, out_dir / "summary.json")
    shutil.copyfile(args.eval, out_dir / "eval_val_full.json")
    print(json.dumps({"out_dir": str(out_dir), "num_bytes": manifest["num_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
