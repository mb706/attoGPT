from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    mlp_mult: float = 4.0
    dropout: float = 0.0
    norm: str = "rmsnorm"
    activation: str = "swiglu"
    pos: str = "rope"
    topk_attn: int = 0
    qk_norm: bool = False
    tie_weights: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm: {kind}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        if config.pos == "rope" and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.block_size = config.block_size
        self.topk_attn = int(config.topk_attn)
        self.qk_norm = config.qk_norm
        self.pos = config.pos

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        causal = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, config.block_size, config.block_size))

        if config.pos == "rope":
            inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            t = torch.arange(config.block_size, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            self.register_buffer("rope_cos", emb.cos().view(1, 1, config.block_size, self.head_dim))
            self.register_buffer("rope_sin", emb.sin().view(1, 1, config.block_size, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.pos == "rope":
            cos = self.rope_cos[:, :, :seq_len, :]
            sin = self.rope_sin[:, :, :seq_len, :]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        causal = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)

        if self.topk_attn > 0 and self.topk_attn < seq_len:
            k_keep = min(self.topk_attn, seq_len)
            _, idx = torch.topk(scores, k=k_keep, dim=-1)
            top_mask = torch.zeros_like(scores, dtype=torch.bool)
            top_mask.scatter_(-1, idx, True)
            top_mask = top_mask & causal
            scores = scores.masked_fill(~top_mask, torch.finfo(scores.dtype).min)

        att = F.softmax(scores, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.proj(y)
        return y, att if return_attn else None


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = int(config.mlp_mult * config.n_embd)
        self.activation = config.activation
        if self.activation == "swiglu":
            self.fc = nn.Linear(config.n_embd, 2 * hidden, bias=False)
            self.proj = nn.Linear(hidden, config.n_embd, bias=False)
        else:
            self.fc = nn.Linear(config.n_embd, hidden, bias=False)
            self.proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            x, gate = self.fc(x).chunk(2, dim=-1)
            x = x * F.silu(gate)
        elif self.activation == "gelu":
            x = F.gelu(self.fc(x))
        elif self.activation == "relu2":
            x = F.relu(self.fc(x)).square()
        else:
            raise ValueError(f"unknown activation: {self.activation}")
        return self.proj(self.dropout(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = make_norm(config.norm, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = make_norm(config.norm, config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, att = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, att


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos == "learned":
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.pos_emb = None
        self.drop = nn.Dropout(config.dropout)
        self.block = DecoderBlock(config)
        self.ln_f = make_norm(config.norm, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        x, att = self.block(x, return_attn=return_attn)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, att

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        bad_token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if bad_token_ids:
                logits[:, bad_token_ids] = -float("inf")
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
