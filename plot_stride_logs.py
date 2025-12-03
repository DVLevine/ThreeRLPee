"""
Quick helper to plot stride training logs (JSONL) produced by train_stride_ppo.py or train_stride_rl.py.

Usage:
  python plot_stride_logs.py --log runs/stride_ppo_20250101-120000/metrics.jsonl
  python plot_stride_logs.py --log runs/stride_rl_20250101-120000 --out figure.png
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def _load_records(log_path: Path) -> List[dict]:
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _extract_series(records: List[dict]) -> Tuple[List[float], List[float], str]:
    if not records:
        return [], [], "return"
    key_x = "epoch" if "epoch" in records[0] else "episode"
    key_y = "avg_return" if "avg_return" in records[0] else "return"
    xs = [float(r.get(key_x, i)) for i, r in enumerate(records)]
    ys = [float(r.get(key_y, 0.0)) for r in records]
    return xs, ys, key_y


def main():
    parser = argparse.ArgumentParser(description="Plot stride training JSONL logs.")
    parser.add_argument("--log", type=str, required=True, help="Path to metrics.jsonl or run directory containing it.")
    parser.add_argument("--out", type=str, default=None, help="Optional output image path (defaults next to log).")
    args = parser.parse_args()

    log_path = Path(args.log)
    if log_path.is_dir():
        log_path = log_path / "metrics.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    records = _load_records(log_path)
    xs, ys, y_label = _extract_series(records)
    if not xs:
        print(f"No data to plot in {log_path}")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label=y_label)
    plt.xlabel("epoch" if "epoch" in records[0] else "episode")
    plt.ylabel(y_label)
    plt.title(log_path.parent.name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = Path(args.out) if args.out else log_path.with_suffix(".png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
