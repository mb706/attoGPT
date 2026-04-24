from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import optuna

from .train import train_once


def objective(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    n_embd = trial.suggest_categorical("n_embd", [64, 96, 128, 160])
    n_head = trial.suggest_categorical("n_head", [2, 4, 5, 8])
    if n_embd % n_head != 0 or (n_embd // n_head) % 2 != 0:
        raise optuna.TrialPruned()
    activation = trial.suggest_categorical("activation", ["swiglu", "gelu", "relu2"])
    norm = trial.suggest_categorical("norm", ["rmsnorm", "layernorm"])
    pos = trial.suggest_categorical("pos", ["rope", "learned"])
    mlp_mult = trial.suggest_categorical("mlp_mult", [2.0, 3.0, 4.0])
    topk_attn = trial.suggest_categorical("topk_attn", [0, 5, 8, 16])
    qk_norm = trial.suggest_categorical("qk_norm", [False, True])
    lr = trial.suggest_float("lr", 8e-4, 6e-3, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.05])
    optimizer = trial.suggest_categorical("optimizer", ["adamw", "muon"])
    muon_lr = trial.suggest_float("muon_lr", 0.005, 0.04, log=True)

    out_dir = Path(base_args.out_root) / f"trial_{trial.number:04d}"
    args = vars(base_args).copy()
    args.update(
        {
            "out_dir": str(out_dir),
            "n_embd": n_embd,
            "n_head": n_head,
            "activation": activation,
            "norm": norm,
            "pos": pos,
            "mlp_mult": mlp_mult,
            "topk_attn": topk_attn,
            "qk_norm": qk_norm,
            "lr": lr,
            "min_lr": lr * 0.1,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "muon_lr": muon_lr,
            "untie_weights": False,
        }
    )
    try:
        result = train_once(SimpleNamespace(**args), trial=trial)
    except RuntimeError as exc:
        if str(exc) == "TRIAL_PRUNED":
            raise optuna.TrialPruned()
        raise
    return float(result["best_val_bpb"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--study-name", default="tinygpt")
    p.add_argument("--storage", default=None)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=args.eval_interval, reduction_factor=3)
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        pruner=pruner,
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, timeout=args.timeout)
    summary = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
    }
    (Path(args.out_root) / "hpo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
