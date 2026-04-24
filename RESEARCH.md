# Tiny Single-Layer GPT Research Log

Current date: 2026-04-24

## Goal

Train a demonstration GPT from scratch that is recognizably an English next-token predictor while remaining extremely small and easy to inspect. Hard architectural constraint: exactly one decoder block with causal self-attention, optionally multi-head and optionally top-k sparse attention so each head incorporates only a small number of previous token values.

The desired outcome is not compute-optimal pretraining. It is a compact, inspectable model with good qualitative samples for its size, plus code that exposes attention and MLP internals for visualization.

## Environment

- Host visible from container: AMD Ryzen 7 PRO 8840HS, 16 logical CPUs.
- Memory visible from container: ~89 GiB host RAM, though the user described a 16 GiB podman target; experiments should stay well below 16 GiB where possible.
- Disk available at `/workspace`: ~613 GiB. User requested staying below 200 GiB; planned footprint is under 5 GiB.
- Accelerators: no NVIDIA/ROCm device exposed (`nvidia-smi`, `rocminfo`, `/dev/kfd` unavailable).
- Framework decision: CPU PyTorch. Installed local env: `torch==2.11.0+cpu`, `numpy`, `tokenizers`, `optuna`, `pyarrow`, `pandas`, `matplotlib`, `tqdm`.

## External Notes Checked

- TinyStories paper: explicitly targets very small LMs and reports that models below 10M parameters, including only one transformer block, can generate fluent simple English stories. Source: https://arxiv.org/abs/2305.07759
- Karpathy autoresearch README: for small compute, recommends lower-entropy data such as TinyStories, smaller vocabularies, smaller max sequence length, lower evaluation tokens, and simpler attention patterns. Source: https://github.com/karpathy/autoresearch
- nanoGPT: useful reference for a compact GPT training loop; its CPU quick start is character-level Shakespeare, but the target here is ordinary simple English rather than Shakespeare. Source: https://github.com/karpathy/nanoGPT
- modded-nanogpt speedrun: useful ideas include Muon optimizer, RoPE/QK-norm-style architecture tweaks, multi-token prediction, sparse attention gates, and scheduled batch/sequence lengths. Most H100-specific kernel tricks are irrelevant on CPU. Source: https://github.com/KellerJordan/modded-nanogpt
- Newton-Muon paper: reports modest gains over Muon in a speedrun reproduction, but implementation complexity is higher and probably not the first thing to try on this CPU-bound tiny setting.

## Data Decision

Primary corpus: `roneneldan/TinyStories`, starting with one parquet training shard and the validation shard.

Rationale:
- It is ordinary English in the sense of narrative prose, but intentionally simple enough for tiny/one-block models.
- It is high quality and low entropy compared with web text.
- The corpus is large enough to over-train a tiny model without exceeding disk constraints.

Preprocessing candidates:
- `simple`: lowercase; keep letters, spaces, periods, commas, question marks, exclamation marks; collapse other punctuation to spaces or sentence punctuation.
- `apostrophe`: same as simple but keep apostrophes inside words. This may improve contractions but increases tokenizer complexity.
- `period_only`: lowercase and keep only periods. Simpler, but likely worse sample readability.

Initial default: `simple`, because commas and sentence-ending punctuation make outputs much easier to read and do not add much entropy.

## Tokenization Hypotheses

- Character-level: most transparent and tiny vocab, but single-layer model spends capacity on spelling and long-range word formation. Good for teaching mechanics, likely worse English at small context length.
- Word-level top-N: very interpretable attention over words and punctuation; best demo fit if `<unk>` rate is controlled. Downside: cannot generate rare words and may emit `<unk>` unless suppressed.
- Small BPE/subword: likely best loss/quality tradeoff at fixed vocab size, but tokens are less human-intuitive than words.

Initial search will compare:
- char vocab from cleaned corpus.
- word vocab sizes 1024, 2048, 4096.
- BPE vocab sizes 512, 1024, 2048 if training time is acceptable.

Metrics should include both token cross-entropy and bits-per-byte (BPB). BPB is important when tokenizers differ.

## Architecture Search Space

Fixed:
- One decoder block.
- Causal next-token objective.

Variables to try:
- Embedding width: 64, 96, 128, 192.
- Attention heads: 2, 4, 6, 8, constrained by width divisibility.
- Context length: 64, 128, maybe 192/256 if CPU is acceptable.
- Positional encoding: learned absolute vs RoPE.
- Norm: LayerNorm vs RMSNorm.
- MLP: GELU vs SwiGLU vs ReLU-squared.
- MLP multiplier: 2x, 3x, 4x.
- Attention sparsity: dense vs top-k values with k in {3, 5, 8, 16}. Dense may train better; top-k is preferred if quality remains acceptable.
- Optimizer: AdamW baseline; Muon for 2D hidden matrices plus AdamW for embeddings/bias/norm/head if implementation is stable.
- Weight tying: tied token embedding / LM head vs untied.

## Early Decisions

- Do not import nanoGPT wholesale. Build a small codebase tailored to the demo: one block, explicit attention return values, top-k masking, tokenizer variants, and experiment logs.
- Start with AdamW and dense attention to establish working baselines before adding sparse attention and Muon.
- Use simple successive-halving style HPO before any longer final run.

## Open Questions

- Which tokenizer gives best qualitative English under a strict one-block architecture?
- How much does top-k sparse attention hurt validation BPB, and what k is acceptable?
- Does RoPE help in a one-block tiny model on short TinyStories contexts?
- Does Muon beat AdamW in this very small CPU-only setting, or is optimizer overhead not worth it?
- What is the best model size for demo: small enough to inspect, but large enough to generate recognizable English?

## Experiment Log

### 2026-04-24: Data Preparation

Prepared reduced TinyStories splits from the first train parquet shard:

| data dir | tokenizer | train docs | val docs | vocab | train tokens | val tokens | val unk rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `data/ts_char_30k` | char | 30k | 5k | 36 | 26,399,531 | 4,029,711 | 0 |
| `data/ts_bpe_1024_30k` | BPE | 30k | 5k | 1024 | 7,854,855 | 1,191,915 | 0 |
| `data/ts_bpe_2048_30k` | BPE | 30k | 5k | 2048 | 6,990,829 | 1,059,037 | 0 |
| `data/ts_word_1024_30k` | word | 30k | 5k | 1024 | 6,161,342 | 929,453 | 7.76% |
| `data/ts_word_2048_30k` | word | 30k | 5k | 2048 | 6,161,342 | 929,453 | 3.59% |
| `data/ts_word_4096_30k` | word | 30k | 5k | 4096 | 6,161,342 | 929,453 | 0.90% |

Interpretation:
- Word-4096 is the strongest interpretability candidate: attention is over whole words/punctuation and unknown rate is below 1%.
- Word-1024 is likely too lossy unless model size must be extremely small.
- Character-level is maximally inspectable but sequence length explodes, making it less likely to produce coherent multi-word English with one block.
- BPE-1024/2048 remain important controls because they avoid unknowns while keeping token count moderate.

### 2026-04-24: Short Training Results

Equal-budget 30k-doc tokenizer baselines with a 1-block, 96-dim, 4-head model for 300 steps:

| run | tokenizer | val BPB | qualitative result |
|---|---:|---:|---|
| `runs/tok_word4096_base` | word-4096 | 1.348 | English-like but repetitive and grammatically rough |
| `runs/tok_bpe2048_base` | BPE-2048 | 1.563 | More readable early samples than word baseline, but worse BPB |
| `runs/tok_char_base` | char | 2.184 | Not suitable; spends capacity on spelling/local character patterns |

Word-level is preferred for both interpretability and loss after tuning.

Optuna successive-halving search on `data/ts_word_4096_30k`:

- Best short-budget trial: `runs/hpo_word4096/trial_0013`, 0.956 BPB at 600 steps.
- Winning recipe: width 160, 2 heads, GELU MLP, RMSNorm, learned absolute positions, Muon, weight decay 0.05, `topk_attn=16`.
- Near tie: `trial_0015`, `topk_attn=8`, 0.959 BPB.
- Muon consistently beat AdamW in the best trials.
- Learned absolute positions beat RoPE in the best trials for this one-layer, short-context setting.
- GELU beat SwiGLU/ReLU2 in the best trials, despite SwiGLU being the default hypothesis.

Focused sparsity sweep on the winning architecture, 30k docs, 600 steps:

| run | attention | val BPB | interpretation |
|---|---:|---:|---|
| `runs/sweep_word_dense` | dense | 0.935 | best loss, less clean for demo |
| `runs/sweep_word_topk5` | top-k 5 | 0.981 | good interpretability/quality tradeoff |
| `runs/sweep_word_topk3` | top-k 3 | 1.010 | visibly worse; too restrictive |

Larger data/vocab check with 200k training docs and full validation:

| run | vocab | attention | val BPB | params |
|---|---:|---:|---:|---:|
| `runs/vocab4096_200k_topk5_800` | 4096 | top-k 5 | 0.942 | 983,520 |
| `runs/vocab8192_200k_topk5_800` | 8192 | top-k 5 | 0.961 | 1,638,880 |

Conclusion: keep vocab-4096. Vocab-8192 reduces `<unk>` rate from ~0.81% to ~0.16% on validation, but the extra output/embedding parameters slow training and do not improve BPB in the tiny model.

Current final candidate:
- Dataset: `data/ts_word_4096_200k`.
- Architecture: one decoder block, vocab-4096 word tokenizer, block size 128, width 160, 2 heads, GELU, RMSNorm, learned positions, tied embeddings, `topk_attn=5`.
- Optimizer: Muon for hidden 2D matrices + AdamW for embeddings/norm/special tensors.
- Reason: `topk_attn=5` gives exactly five nonzero source tokens per head, and its quality remains acceptable; dense/top-k 8/16 are better but less aligned with the interpretability goal.

### 2026-04-24: Final Training Run

Final run: `runs/final_word4096_topk5`

Configuration:
- Dataset: `data/ts_word_4096_200k`, 200k TinyStories training docs from train shard 0, full 21,990-doc validation split.
- Tokenizer: lowercase word/punctuation tokenizer, vocab 4096, validation `<unk>` rate ~0.81%.
- Model: exactly one decoder block, block size 128, width 160, 2 attention heads, learned absolute positions, RMSNorm, GELU MLP, tied embeddings, no dropout.
- Sparse attention: `topk_attn=5`, so each head at each position has at most five nonzero causal attention sources.
- Optimizer: Muon for 2D hidden matrices, AdamW for embeddings/norm/special tensors, cosine LR.
- Parameters: 983,520.
- Checkpoint size: ~3.96 MB.

Training:
- Steps: 12,000.
- Batch: 64 x 128 = 8192 tokens/step.
- Approx tokens processed: 98.3M, or ~2.36 passes over the 41.7M-token training file.
- Runtime: 6132 seconds (~1.70 hours).
- Best sampled validation checkpoint: step 11,500.
- Sampled validation at best: loss 2.4663 nats/token, BPB 0.8295.
- Full sequential validation of `best.pt`: loss 2.4790 nats/token, BPB 0.8338 over 4,421,888 validation tokens.

Final attention inspection on prompt `once upon a time there was a little dog who wanted a red ball`:
- Next-token prediction after `ball`: period probability 0.7574.
- Head 0 nonzero sources: `wanted` 0.394, `who` 0.255, `a` 0.251, `ball` 0.050, `red` 0.049.
- Head 1 nonzero sources: `wanted` 0.280, `who` 0.241, `there` 0.170, `upon` 0.159, `was` 0.150.

Final decision:
- Stop at 12k steps. The last sampled eval at step 12k was worse than step 11.5k, and the late improvements were small enough that a longer extension is unlikely to materially improve the demonstration value.
- Keep top-k 5 rather than dense/top-k 8/16. Dense attention can lower loss, but top-k 5 makes the model far easier to explain and still produces recognizable English.
