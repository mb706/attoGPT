#!/usr/bin/env bash
set -euo pipefail

python -m tinygpt_demo.prepare \
  --out-dir data/ts_word_4096_200k \
  --raw-dir raw_data/hf \
  --train-shards 1 \
  --max-train-docs 200000 \
  --max-val-docs 21990 \
  --tokenizer word \
  --vocab-size 4096 \
  --text-mode simple

python -m tinygpt_demo.train \
  --data-dir data/ts_word_4096_200k \
  --out-dir runs/final_word4096_topk5 \
  --threads 16 \
  --block-size 128 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --max-steps 12000 \
  --eval-interval 500 \
  --eval-iters 24 \
  --n-embd 160 \
  --n-head 2 \
  --mlp-mult 4 \
  --activation gelu \
  --norm rmsnorm \
  --pos learned \
  --topk-attn 5 \
  --optimizer muon \
  --lr 0.0042 \
  --min-lr 0.00008 \
  --muon-lr 0.028 \
  --weight-decay 0.05 \
  --warmup-steps 120

python -m tinygpt_demo.eval_checkpoint \
  --checkpoint runs/final_word4096_topk5/best.pt \
  --split val \
  --batch-size 64 \
  --threads 16 \
  --out runs/final_word4096_topk5/eval_val_full.json

python scripts/export_web_model.py \
  --checkpoint runs/final_word4096_topk5/best.pt \
  --tokenizer data/ts_word_4096_200k/tokenizer.json \
  --config runs/final_word4096_topk5/config.json \
  --summary runs/final_word4096_topk5/summary.json \
  --eval runs/final_word4096_topk5/eval_val_full.json \
  --out-dir docs/model
