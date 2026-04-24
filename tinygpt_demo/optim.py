from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz orthogonalization used by common Muon implementations."""
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.float()
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    else:
        transposed = False
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Small single-process Muon optimizer for 2D hidden weights."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon parameter groups should contain only 2D tensors")
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss


def make_optimizers(
    model: torch.nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float,
    muon_lr: float,
) -> list[torch.optim.Optimizer]:
    if optimizer == "adamw":
        return [
            torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
            )
        ]
    if optimizer == "muon":
        muon_params = []
        adam_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            use_muon = p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name
            if use_muon:
                muon_params.append(p)
            else:
                adam_params.append(p)
        opts: list[torch.optim.Optimizer] = []
        if muon_params:
            opts.append(Muon(muon_params, lr=muon_lr, momentum=0.95, weight_decay=weight_decay))
        if adam_params:
            opts.append(
                torch.optim.AdamW(
                    adam_params,
                    lr=lr,
                    betas=(0.9, 0.95),
                    weight_decay=weight_decay,
                )
            )
        return opts
    raise ValueError(f"unknown optimizer: {optimizer}")
