# attoGPT

(*100% vibe coded.*)

**[Explore the Model](https://mb706.github.io/attoGPT/) (webpage)**

attoGPT is a from-scratch, CPU-trained, extremely small GPT-style next-token model for demonstration and interpretability. The model is intentionally constrained to one decoder block so the attention and MLP components are easy to inspect.

The final checked-in result is a word-level TinyStories model with sparse causal attention. At each position, each head keeps only the top 5 causal attention weights and zeros the rest before mixing values. This makes it possible to say exactly which previous tokens were incorporated by each head for a next-token prediction.

## Final Artifact

The final checkpoint bundle is tracked in `artifacts/final/`:

- `best.pt`: trained PyTorch checkpoint, 3.8 MiB on disk.
- `tokenizer.json`: word/punctuation vocabulary required for inference.
- `config.json`: training configuration and data metadata.
- `summary.json`: sampled validation summary from training.
- `eval_val_full.json`: full sequential validation result.

Final model:

- Architecture: exactly one decoder block.
- Parameters: 983,520.
- Tokenizer: lowercase word/punctuation tokenizer, vocab size 4096.
- Context length: 128 tokens.
- Embedding width: 160.
- Attention: 2 heads, causal, `topk_attn=5`.
- MLP: GELU, 4x expansion.
- Norm: RMSNorm.
- Position encoding: learned absolute positions.
- Optimizer used for training: Muon for 2D hidden matrices plus AdamW for embeddings/norm/special tensors.
- Training corpus: TinyStories, 200,000 training docs from the first train shard, full validation split.
- Full validation: 2.4790 nats/token, 0.8338 bits per byte over 4,421,888 validation tokens.

The research log in `RESEARCH.md` documents the environment, tokenizer comparisons, HPO results, sparsity sweep, final training run, and rationale for the selected setup.

## Quick Start

Install dependencies in a fresh environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-cpu.txt
```

Generate text from the checked-in final checkpoint:

```bash
python -m tinygpt_demo.sample \
  --checkpoint artifacts/final/best.pt \
  --prompt "once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.7 \
  --top-k 30
```

Inspect the sparse attention sources for a prompt:

```bash
python -m tinygpt_demo.inspect_attention \
  --checkpoint artifacts/final/best.pt \
  --prompt "once upon a time there was a little dog who wanted a red ball" \
  --topn 8
```

Expected attention behavior for the final prompt above: after `ball`, each of the 2 heads reports exactly 5 nonzero source tokens because the model was trained with `topk_attn=5`.

## Browser Explorer

The `docs/` directory is a static GitHub Pages site for exploring the model in the browser. It includes a web-exported copy of the final checkpoint:

- `docs/index.html`: the interactive model explorer.
- `docs/model.js`: vanilla JavaScript inference for this one-block GPT.
- `docs/model/model.bin`: flat Float32 model weights, about 3.9 MB.
- `docs/model/manifest.json`: tensor offsets, shapes, config, and checksum.
- `docs/model/tokenizer.json`: the final word tokenizer.

To preview locally:

```bash
python3 -m http.server 8765 --directory docs
```

Then open `http://127.0.0.1:8765/`.

To host on GitHub Pages, configure Pages to serve from the `docs/` folder on the `master` branch.

## Reproducing The Final Run

The exact end-to-end command sequence is in `scripts/reproduce_final.sh`.

Run it from the repository root after installing dependencies:

```bash
. .venv/bin/activate
bash scripts/reproduce_final.sh
```

This will:

1. Download the TinyStories parquet files needed from Hugging Face.
2. Build the 4096-word tokenizer and memmapped train/validation token files under `data/`.
3. Train the final one-block sparse-attention GPT under `runs/final_word4096_topk5/`.
4. Run a full sequential validation pass and write `eval_val_full.json`.
5. Export the trained checkpoint to `docs/model/` for the browser explorer.

On the original environment, the final training run used 16 CPU threads on an AMD Ryzen 7 PRO 8840HS and took about 1.70 hours. The processed data and run outputs are intentionally ignored by git because they can be regenerated.

## Important Commands

Prepare the final dataset only:

```bash
python -m tinygpt_demo.prepare \
  --out-dir data/ts_word_4096_200k \
  --raw-dir raw_data/hf \
  --train-shards 1 \
  --max-train-docs 200000 \
  --max-val-docs 21990 \
  --tokenizer word \
  --vocab-size 4096 \
  --text-mode simple
```

Train the final model:

```bash
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
```

Evaluate the best checkpoint over the full validation token file:

```bash
python -m tinygpt_demo.eval_checkpoint \
  --checkpoint runs/final_word4096_topk5/best.pt \
  --split val \
  --batch-size 64 \
  --threads 16 \
  --out runs/final_word4096_topk5/eval_val_full.json
```

## Repository Layout

- `tinygpt_demo/model.py`: one-block GPT implementation with optional top-k sparse attention.
- `tinygpt_demo/prepare.py`: TinyStories download, normalization, tokenizer training, and token memmap creation.
- `tinygpt_demo/train.py`: training loop with AdamW or Muon.
- `tinygpt_demo/hpo.py`: Optuna successive-halving search used during investigation.
- `tinygpt_demo/sample.py`: text generation.
- `tinygpt_demo/inspect_attention.py`: per-head attention source inspection.
- `tinygpt_demo/eval_checkpoint.py`: sequential checkpoint evaluation.
- `scripts/export_web_model.py`: exports a PyTorch checkpoint into the static browser format.
- `artifacts/final/`: checked-in final checkpoint bundle.
- `docs/`: GitHub Pages model explorer and browser-loadable model bundle.
- `RESEARCH.md`: detailed research log and experimental results.

## Ignored Files

The following are ignored and should be regenerated locally:

- `.venv/`: local Python environment.
- `raw_data/`: downloaded TinyStories parquet files.
- `data/`: processed token memmaps and tokenizer experiments.
- `runs/`: intermediate checkpoints, HPO databases, metrics, and training outputs.
- `__pycache__/` and other Python caches.

Only the final compact result under `artifacts/final/` is checked in.

## Sources

- TinyStories paper: https://arxiv.org/abs/2305.07759
- TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- Karpathy autoresearch: https://github.com/karpathy/autoresearch
- nanoGPT: https://github.com/karpathy/nanoGPT
- modded-nanogpt speedrun: https://github.com/KellerJordan/modded-nanogpt
