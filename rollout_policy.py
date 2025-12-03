# rollout_policy.py
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from policy_3lp import PolicyConfig, LinearBasisActor, MLPActor
from train_3lp_ac import _make_env, _rollout_states, _params_to_tuple, maybe_visualize


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt


def build_actor(cfg_dict, policy_type):
    cfg_dict = dict(cfg_dict)
    if "hidden_sizes" in cfg_dict:
        cfg_dict["hidden_sizes"] = tuple(cfg_dict["hidden_sizes"])
    cfg = PolicyConfig(**cfg_dict)
    if policy_type == "linear":
        actor = LinearBasisActor(cfg).to("cpu")
    elif policy_type == "mlp":
        actor = MLPActor(cfg).to("cpu")
    else:
        raise ValueError(f"Unknown policy_type {policy_type}")
    return actor, cfg


def main():
    parser = argparse.ArgumentParser(description="Roll out a saved policy checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.pt saved by train_3lp_ac.py.")
    parser.add_argument("--steps", type=int, default=200, help="Max steps for rollout.")
    parser.add_argument("--dense", action="store_true", help="Use dense stride sampling when available.")
    parser.add_argument("--viz", action="store_true", help="Visualize the rollout synchronously.")
    parser.add_argument("--loop", action="store_true", help="Loop visualization window.")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional CSV path to save states (flat q).")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path)

    env_type = ckpt["env_type"]
    policy_type = ckpt.get("policy_type", "linear")
    cfg_dict = ckpt["config"]
    params_tuple = ckpt.get("params")
    t_ds = ckpt.get("t_ds")
    t_ss = ckpt.get("t_ss")

    actor, cfg = build_actor(cfg_dict, policy_type)
    actor.load_state_dict(ckpt["actor_state"])

    env = _make_env(env_type, debug_env=False)

    state_lists = _rollout_states(
        env,
        actor,
        device="cpu",
        max_steps=args.steps,
        dense_stride=args.dense,
        n_substeps=120,
        log_prefix="rollout",
        override_params=None,
        override_phase_times=None,
    )

    if args.out_csv:
        import csv

        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["q" + str(i) for i in range(len(state_lists[0] if state_lists else []))])
            for st in state_lists:
                if hasattr(st, "q"):
                    writer.writerow([float(v) for v in st.q])
                else:
                    writer.writerow([float(v) for v in st])
        print(f"Saved rollout states to {args.out_csv}")

    if args.viz:
        # Rebuild params for visualization
        if params_tuple is not None and hasattr(env, "sim") and env.sim is not None:
            try:
                env.sim.set_params(env.sim.get_params().__class__(*params_tuple))
            except Exception:
                pass
            if t_ds is not None and t_ss is not None:
                try:
                    env.sim.set_phase_times(t_ds, t_ss)
                except Exception:
                    pass
        maybe_visualize(env, actor, device="cpu", max_steps=args.steps, loop=args.loop, dense_stride=args.dense, n_substeps=120, log_prefix="rollout")


if __name__ == "__main__":
    main()
