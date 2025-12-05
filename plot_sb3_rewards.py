import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_monitor(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read SB3 monitor.csv (VecMonitor output).
    Returns (episodes, rewards).
    """
    rewards = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                # first header line with metadata or column names
                if "r" in row[0]:
                    header_seen = True
                continue
            if not header_seen:
                # Skip until we hit the column header
                continue
            # Expected columns: r, l, t
            try:
                rew = float(row[0])
            except Exception:
                continue
            rewards.append(rew)
    episodes = np.arange(1, len(rewards) + 1)
    return episodes, np.asarray(rewards, dtype=np.float64)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = min(k, x.size)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    smooth = (cumsum[k:] - cumsum[:-k]) / float(k)
    # Pad the front to keep lengths aligned
    pad = np.full(k - 1, smooth[0])
    return np.concatenate([pad, smooth])


def plot(episodes: np.ndarray, rewards: np.ndarray, smooth: int, save: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to plot: {exc}")

    y = moving_average(rewards, smooth) if smooth and smooth > 1 else rewards
    plt.figure()
    plt.plot(episodes, y, label=f"reward (smooth={smooth})" if smooth and smooth > 1 else "reward")
    if smooth and smooth > 1:
        plt.plot(episodes, rewards, color="gray", alpha=0.3, linewidth=0.8, label="raw")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("SAC stride training reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"[plot] saved to {save}")
    else:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Plot episode rewards from SAC stride training (monitor.csv).")
    p.add_argument("--run-dir", type=str, required=True, help="Run directory containing monitor/monitor.csv.")
    p.add_argument("--smooth", type=int, default=20, help="Moving average window (episodes). Use 1 for no smoothing.")
    p.add_argument("--save", type=str, default=None, help="Optional path to save the plot (png). If omitted, shows interactively.")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    monitor_path = run_dir / "monitor" / "monitor.csv"
    if not monitor_path.exists():
        raise SystemExit(f"monitor.csv not found at {monitor_path}. Ensure training was run after adding VecMonitor.")

    episodes, rewards = read_monitor(monitor_path)
    if len(rewards) == 0:
        raise SystemExit("No rewards found in monitor.csv.")

    print(f"[data] episodes={len(rewards)}  mean_return={rewards.mean():.3f}  last_100_mean={rewards[-100:].mean():.3f}")
    save_path = Path(args.save) if args.save else None
    plot(episodes, rewards, args.smooth, save_path)


if __name__ == "__main__":
    main()
