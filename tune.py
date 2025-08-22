#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple hyperparameter tuner for train.py.

- Runs multiple trials with random hyperparameters.
- Compares runs by minimum val RMSD from history.csv.
- Copies the best run's model.pt to model/model.pt and writes metadata.
- Periodic cleanup: remove created runs every N trials (default 10), configurable via --cleanup_every.

Usage (examples):
    python tune.py --max_trials 10 --epochs 30 --device cuda
    python tune.py --max_trials 50 --epochs 15 --device cpu
    python tune.py --max_trials 30 --cleanup_every 10 --keep_best_run
"""

import os
import sys
import json
import time
import math
import shutil
import random
import argparse
import subprocess
from datetime import datetime

# OpenMP guard to avoid libiomp5md conflicts on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import pandas as pd
except Exception as e:
    print("pandas is required by tune.py to parse history.csv. Please install pandas.")
    raise


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sample_params(rng: random.Random):
    """Randomly sample hyperparameters from sensible ranges.

    - hidden: 64..512 step 32 (integer)
    - depth: 2..5 (integer)
    - dropout: U[0.0, 0.4]
    - lr: logU[1e-4, 3e-3]
    - weight_decay: with 25% chance 0.0, else logU[1e-6, 1e-3]
    - batch_size: even numbers 4..16
    - corr_w: U[0.0, 0.2]
    - features: choose from a few reasonable combos
    """
    features_choices = [
        "base+baseline",
        "base+baseline+interact",
        "base+interact",
    ]

    # helpers
    def log_uniform(a: float, b: float) -> float:
        la, lb = math.log(a), math.log(b)
        return math.exp(rng.uniform(la, lb))

    hidden = rng.randrange(64, 513, 32)
    depth = rng.randint(2, 5)
    dropout = round(rng.uniform(0.0, 0.4), 3)
    lr = log_uniform(1e-4, 3e-3)
    if rng.random() < 0.25:
        weight_decay = 0.0
    else:
        weight_decay = log_uniform(1e-6, 1e-3)
    batch_size = rng.randrange(4, 18, 2)
    corr_w = round(rng.uniform(0.0, 0.2), 3)
    features = rng.choice(features_choices)

    return {
        "hidden": int(hidden),
        "depth": int(depth),
        "dropout": float(dropout),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "corr_w": float(corr_w),
        "features": features,
    }


def run_train(train_py: str, out_dir: str, csv_path: str, device: str, epochs: int, val_ratio: float, base_seed: int, params: dict) -> int:
    """Run a single training trial. Returns subprocess return code."""
    cmd = [
        sys.executable, train_py,
        "--csv", csv_path,
        "--epochs", str(epochs),
        "--device", device,
        "--val_ratio", str(val_ratio),
        "--out_dir", out_dir,
        "--seed", str(base_seed),
        "--hidden", str(params["hidden"]),
        "--depth", str(params["depth"]),
        "--dropout", str(params["dropout"]),
        "--lr", str(params["lr"]),
        "--weight_decay", str(params["weight_decay"]),
        "--batch_size", str(params["batch_size"]),
        "--corr_w", str(params["corr_w"]),
        "--features", params["features"],
    ]
    # Pass-through caching flags if present in environment variables (set by main via args)
    cache_dir = os.environ.get("TUNE_CACHE_DIR")
    no_cache_flag = os.environ.get("TUNE_NO_CACHE") == "1"
    if cache_dir:
        cmd += ["--cache_dir", cache_dir]
    if no_cache_flag:
        cmd += ["--no_cache"]
    env = os.environ.copy()
    # Ensure our OMP guard propagates
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def evaluate_run(out_dir: str) -> float:
    """Read history.csv and return min val_rmse (RMSD)."""
    hist_path = os.path.join(out_dir, "history.csv")
    if not os.path.isfile(hist_path):
        raise FileNotFoundError(f"history.csv not found in {out_dir}")
    df = pd.read_csv(hist_path)
    if "val_rmse" not in df.columns:
        raise ValueError("history.csv does not contain 'val_rmse'. Please retrain with updated train.py")
    return float(df["val_rmse"].min())


def save_best(out_dir: str, model_dir: str, params: dict, score: float, extra: dict):
    os.makedirs(model_dir, exist_ok=True)
    src_model = os.path.join(out_dir, "model.pt")
    dst_model = os.path.join(model_dir, "model.pt")
    shutil.copy2(src_model, dst_model)

    meta = {
        "best_val_rmsd": score,
        "out_dir": out_dir,
        "params": params,
        "timestamp": now_tag(),
    }
    meta.update(extra)
    with open(os.path.join(model_dir, "best_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def append_log(log_path: str, record: dict):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Append as JSON lines
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("data", "train_set.csv"))
    ap.add_argument("--device", default=("cuda" if torch_cuda_available() else "cpu"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--max_trials", type=int, default=3000, help="Number of trials; set -1 for infinite")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--model_dir", default="model")
    ap.add_argument("--no_cleanup", action="store_true", help="Do not cleanup created runs after tuning")
    ap.add_argument("--keep_best_run", action="store_true", help="Keep the best run directory when cleaning up")
    ap.add_argument("--purge_all_runs", action="store_true", help="Delete all subdirectories under runs_root at the end (dangerous)")
    ap.add_argument("--cleanup_every", type=int, default=10, help="Clean created runs every N trials (ignored if --no_cleanup or --purge_all_runs)")
    # Cache passthrough to train.py
    ap.add_argument("--cache_dir", default="cache", help="Dataset cache directory passed to train.py")
    ap.add_argument("--no_cache", action="store_true", help="Disable train.py dataset caching")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.runs_root, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    best_score = float("inf")
    best_out = None
    log_path = os.path.join(args.model_dir, "tuning_history.jsonl")

    trial = 0
    created_runs = []
    try:
        while args.max_trials < 0 or trial < args.max_trials:
            trial += 1
            params = sample_params(rng)
            tag = f"{now_tag()}_T{trial:03d}"
            out_dir = os.path.join(args.runs_root, tag)
            os.makedirs(out_dir, exist_ok=True)
            created_runs.append(out_dir)

            # Set passthrough cache flags via environment for run_train
            os.environ["TUNE_CACHE_DIR"] = args.cache_dir if not args.no_cache else ""
            os.environ["TUNE_NO_CACHE"] = "1" if args.no_cache else "0"

            rc = run_train(
                train_py=os.path.join(os.path.dirname(__file__), "train.py"),
                out_dir=out_dir,
                csv_path=args.csv,
                device=args.device,
                epochs=args.epochs,
                val_ratio=args.val_ratio,
                base_seed=args.seed + trial,
                params=params,
            )
            record = {
                "trial": trial,
                "out_dir": out_dir,
                "params": params,
                "return_code": rc,
            }
            if rc != 0:
                record["status"] = "failed"
                append_log(log_path, record)
                print(f"Trial {trial} failed with return code {rc}.")
                continue

            try:
                score = evaluate_run(out_dir)
                record["val_rmsd"] = score
                record["status"] = "ok"
                append_log(log_path, record)
                print(f"Trial {trial} RMSD={score:.6f}")

                if score < best_score:
                    best_score = score
                    best_out = out_dir
                    save_best(out_dir, args.model_dir, params, best_score, extra={"trial": trial})
                    print(f"New best RMSD={best_score:.6f}. Saved to {os.path.join(args.model_dir, 'model.pt')}")
            except Exception as e:
                record["status"] = "error"
                record["error"] = str(e)
                append_log(log_path, record)
                print(f"Trial {trial} evaluation error: {e}")

            # Periodic cleanup
            if (not args.no_cleanup) and (not args.purge_all_runs) and args.cleanup_every > 0:
                if trial % args.cleanup_every == 0:
                    cleaned = 0
                    for path in list(created_runs):
                        if args.keep_best_run and best_out is not None and os.path.abspath(path) == os.path.abspath(best_out):
                            continue
                        if os.path.isdir(path):
                            shutil.rmtree(path, ignore_errors=True)
                            cleaned += 1
                        created_runs.remove(path)
                    print(f"Periodic cleanup: removed {cleaned} runs after trial {trial}.")
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")

    if best_out is None:
        print("No successful trials. Nothing saved.")
    else:
        print(f"Best RMSD={best_score:.6f} from run {best_out}. Best model is in {os.path.join(args.model_dir, 'model.pt')}")

    # Final cleanup runs per user request
    try:
        if args.purge_all_runs:
            # Dangerous: remove all subdirectories under runs_root
            for name in os.listdir(args.runs_root):
                path = os.path.join(args.runs_root, name)
                if os.path.isdir(path):
                    if args.keep_best_run and best_out is not None and os.path.abspath(path) == os.path.abspath(best_out):
                        continue
                    shutil.rmtree(path, ignore_errors=True)
            print(f"Purged all runs under {args.runs_root} (keep_best_run={args.keep_best_run}).")
        elif not args.no_cleanup:
            # Default: remove only runs created in this tuning session (remaining after periodic cleanup)
            for path in created_runs:
                if args.keep_best_run and best_out is not None and os.path.abspath(path) == os.path.abspath(best_out):
                    continue
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
            print(f"Final cleanup: removed remaining {len(created_runs) - (1 if (args.keep_best_run and best_out in created_runs) else 0)} created runs.")
    except Exception as e:
        print(f"Cleanup skipped due to error: {e}")


def torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
