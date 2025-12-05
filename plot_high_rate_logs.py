"""
Plot iteration/episode vs reward for high-rate trainers.

Supports:
- PPO CSV from train_high_rate_ppo.py (header: update,ep_return,...)
- RL CSV from train_high_rate_rl.py (no header: ep, return, len)

Examples:
  python plot_high_rate_logs.py --log logs/ppo_log.csv --out runs/ppo_reward.png
  python plot_high_rate_logs.py --log logs/train_log.csv --smooth 10
"""
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def _load_series(path: Path) -> Tuple[List[float], List[float], str, str]:
    """
    Returns xs, ys, x_label, y_label.
    Handles both headerless RL logs and headered PPO logs.
    """
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        has_header = any(ch.isalpha() for ch in first)

        if has_header:
            reader = csv.DictReader(f)
            records = list(reader)
            x_candidates = ("update", "iteration", "epoch", "episode", "ep")
            y_candidates = ("ep_return", "return", "avg_return", "reward")
            x_key = next((k for k in x_candidates if k in reader.fieldnames), None)
            y_key = next((k for k in y_candidates if k in reader.fieldnames), None)
            if x_key is None or y_key is None:
                raise ValueError(f"Could not find x/y columns in header: {reader.fieldnames}")
            xs = [float(r.get(x_key, i)) for i, r in enumerate(records)]
            ys = [float(r.get(y_key, 0.0)) for r in records]
            return xs, ys, x_key, y_key
        else:
            # train_high_rate_rl.py: ep, return, len
            reader = csv.reader(f)
            xs = []
            ys = []
            for row in reader:
                if len(row) < 2:
                    continue
                xs.append(float(row[0]))
                ys.append(float(row[1]))
            return xs, ys, "episode", "return"


def _smooth(data: List[float], window: int) -> List[float]:
    if window <= 1 or len(data) == 0:
        return data
    out = []
    cumsum = [0.0]
    for i, v in enumerate(data, start=1):
        cumsum.append(cumsum[-1] + v)
        start = max(0, i - window)
        avg = (cumsum[i] - cumsum[start]) / (i - start)
        out.append(avg)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot reward vs iteration/episode for high-rate training logs.")
    parser.add_argument("--log", required=True, help="Path to CSV log (ppo_log.csv or train_log.csv).")
    parser.add_argument("--out", default=None, help="Optional output image path (defaults to alongside log).")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window (in steps).")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    xs, ys, x_label, y_label = _load_series(log_path)
    if args.smooth > 1:
        ys_plot = _smooth(ys, args.smooth)
        y_label_plot = f"{y_label} (ma={args.smooth})"
    else:
        ys_plot = ys
        y_label_plot = y_label

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys_plot, label=y_label_plot)
    plt.xlabel(x_label)
    plt.ylabel(y_label_plot)
    plt.title(log_path.name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = Path(args.out) if args.out else log_path.with_suffix(".png")
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved to {out_path}")


if __name__ == "__main__":
    main()
