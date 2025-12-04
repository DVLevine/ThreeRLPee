"""
Tools to sanity check the high-rate 3LP pipeline:
 - run a rollout with a saved policy (PPO .pt or RL .npz) or a zero/random policy
 - plot the applied U/V torque parameters and resulting joint torques
 - optionally replay the rollout in the Open3D visualizer when available

Usage:
  source .venv/bin/activate
  python viz_high_rate_3lp.py --algo ppo --policy weights/ppo_model.pt --steps 400 --plot runs/torque.png --visualize
  python viz_high_rate_3lp.py --algo rl --policy weights/policy.npz --v-cmd 1.0 --plot out.png
  python viz_high_rate_3lp.py --check-backend
"""
import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from env_high_rate_3lp import ThreeLPHighRateEnv
from train_high_rate_ppo import ActorCritic, preprocess_obs
from train_high_rate_rl import build_actor_features


def backend_report() -> Dict[str, object]:
    """Probe pybind + visualizer availability without raising."""
    report: Dict[str, object] = {
        "threelp_available": False,
        "has_visualizer": False,
        "has_canonical_rollout": False,
        "has_uv_torque": False,
        "module_path": None,
        "error": None,
    }
    try:
        import threelp  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostic path
        report["error"] = str(exc)
        return report

    report["threelp_available"] = True
    report["has_visualizer"] = hasattr(threelp, "visualize_trajectory")
    report["has_canonical_rollout"] = hasattr(threelp, "visualize_canonical_rollout")
    report["has_uv_torque"] = hasattr(threelp, "compute_uv_torque")
    report["module_path"] = getattr(threelp, "__file__", None)
    return report


def _print_backend(report: Dict[str, object]) -> None:
    print("[backend] threelp:", "ok" if report.get("threelp_available") else f"missing ({report.get('error')})")
    if report.get("module_path"):
        print(f"[backend] module path: {report['module_path']}")
    print("[backend] visualize_trajectory:", "yes" if report.get("has_visualizer") else "no")
    print("[backend] visualize_canonical_rollout:", "yes" if report.get("has_canonical_rollout") else "no")
    print("[backend] compute_uv_torque:", "yes" if report.get("has_uv_torque") else "no")


def _infer_hidden_dim(state_dict: dict) -> Optional[int]:
    """Try to infer hidden width from saved PPO weights."""
    for key in ("actor.0.weight", "critic.0.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[0])
    return None


def _make_policy(
    algo: str,
    env: ThreeLPHighRateEnv,
    policy_path: Optional[Path],
    hidden_dim: Optional[int],
    rng: np.random.Generator,
    eval_noise: float = 0.0,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Optional[torch.nn.Module]]:
    algo = algo.lower()
    v_nom = 0.5 * (env.v_cmd_range[0] + env.v_cmd_range[1])

    if algo == "ppo":
        state_dict = None
        inferred_hidden = None
        if policy_path:
            state_dict = torch.load(policy_path, map_location="cpu")
            inferred_hidden = _infer_hidden_dim(state_dict)
        hd = hidden_dim if hidden_dim is not None else (inferred_hidden or 64)

        def _build_model(h: int):
            return ActorCritic(obs_dim=11, action_dim=env.action_dim, hidden_dim=h)

        model = _build_model(hd)
        if state_dict is not None:
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                # Retry with inferred hidden dim if mismatch.
                if inferred_hidden is not None and inferred_hidden != hd:
                    model = _build_model(inferred_hidden)
                    model.load_state_dict(state_dict, strict=True)
                else:
                    raise exc
        model.eval()

        def _policy(obs_np: np.ndarray) -> np.ndarray:
            z = preprocess_obs(env, obs_np, v_nom)
            if not isinstance(z, torch.Tensor):
                z_t = torch.as_tensor(z, dtype=torch.float32, device="cpu")
            else:
                z_t = z.to("cpu")
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(z_t)
            if eval_noise > 0.0:
                noise = torch.randn_like(action) * eval_noise
                action = action + noise
            return action.cpu().numpy()

        return _policy, model

    if algo == "rl":
        if policy_path is None:
            raise ValueError("RL policy requires --policy pointing to a .npz file.")
        data = np.load(policy_path)
        W_a = np.asarray(data["W_a"], dtype=np.float64)

        def _policy(obs_np: np.ndarray) -> np.ndarray:
            z, _ = build_actor_features(env, obs_np, v_nom)
            action = W_a @ z
            if eval_noise > 0.0:
                action = action + rng.normal(scale=eval_noise, size=action.shape)
            return action

        return _policy, None

    if algo == "zero":
        return lambda obs_np: np.zeros(env.action_dim, dtype=np.float32), None

    if algo == "random":
        return lambda obs_np: rng.normal(scale=5.0, size=env.action_dim), None

    raise ValueError(f"Unknown algo '{algo}' (expected ppo, rl, zero, random)")


def rollout_policy(
    env: ThreeLPHighRateEnv,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    steps: int,
    seed: Optional[int],
    v_cmd: Optional[float],
    perturb: bool = False,
) -> Dict[str, List]:
    log: Dict[str, List] = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "p_running": [],
        "tau_corr": [],
        "tau_total": [],
        "tau_ref": [],
        "state_world": [],
        "x_can": [],
        "phi": [],
        "phase": [],
        "fallen": [],
        "terminated": [],
    }
    options = {}
    if v_cmd is not None:
        options["v_cmd"] = float(v_cmd)
    if perturb:
        options["perturb"] = True

    obs, _ = env.reset(seed=seed, options=options if options else None)

    for _ in range(steps):
        action = np.asarray(policy_fn(obs), dtype=np.float64).reshape(-1)
        obs_next, reward, terminated, truncated, info = env.step(action)

        log["obs"].append(obs)
        log["actions"].append(action)
        log["rewards"].append(reward)
        log["terminated"].append(bool(terminated))
        log["fallen"].append(bool(info.get("fallen", False)))
        log["phi"].append(float(info.get("phi_stride", obs[8])))
        log["phase"].append(info.get("phase", ""))
        if "p_running" in info:
            log["p_running"].append(np.asarray(info["p_running"], dtype=np.float64))
        if "tau_corr" in info:
            log["tau_corr"].append(np.asarray(info["tau_corr"], dtype=np.float64))
        if "tau_total" in info and info["tau_total"] is not None:
            log["tau_total"].append(np.asarray(info["tau_total"], dtype=np.float64))
        if "tau_ref" in info and info["tau_ref"] is not None:
            log["tau_ref"].append(np.asarray(info["tau_ref"], dtype=np.float64))
        if "state_world" in info:
            log["state_world"].append(np.asarray(info["state_world"], dtype=np.float64))
        if "x_can" in info:
            log["x_can"].append(np.asarray(info["x_can"], dtype=np.float64))

        obs = obs_next
        if terminated or truncated:
            break

    return log


def plot_torques(log: Dict[str, List], dt: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    p_params = np.asarray(log.get("p_running", []))
    tau_resid = np.asarray(log.get("tau_corr", []))
    tau_total = np.asarray(log.get("tau_total", []))
    tau_ref = np.asarray(log.get("tau_ref", []))
    if p_params.size == 0 and tau_resid.size == 0 and tau_total.size == 0:
        print("[plot] nothing to plot (missing torque data)")
        return

    n = max(len(p_params), len(tau_resid), len(tau_total))
    t = np.arange(n) * dt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    if p_params.size > 0:
        axes[0].plot(t[: len(p_params)], p_params)
        axes[0].set_ylabel("p_running (U/V params)")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "p_running unavailable", ha="center", va="center")
        axes[0].set_axis_off()

    if tau_total.size > 0 or tau_ref.size > 0:
        if tau_total.size > 0:
            axes[1].plot(t[: len(tau_total)], tau_total, label="τ total (p_running)")
        if tau_ref.size > 0:
            axes[1].plot(t[: len(tau_ref)], tau_ref, "--", label="τ reference")
        axes[1].set_ylabel("τ total")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "tau_total unavailable", ha="center", va="center")
        axes[1].set_axis_off()

    if tau_resid.size > 0:
        axes[2].plot(t[: len(tau_resid)], tau_resid, color="tab:red")
        axes[2].set_ylabel("τ residual (delta action)")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "tau_corr unavailable", ha="center", va="center")
        axes[2].set_axis_off()

    axes[2].set_xlabel("time [s]")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved torque figure to {out_path}")


def _visualize_with_reference(log: Dict[str, List], env: ThreeLPHighRateEnv, loop: bool) -> bool:
    """
    If visualize_canonical_rollout exists, render both policy and reference in canonical space.
    Returns True on success, False otherwise (caller can fall back).
    """
    try:
        import threelp  # type: ignore
    except Exception:
        return False
    if not hasattr(threelp, "visualize_canonical_rollout"):
        return False
    if not log.get("x_can"):
        return False
    try:
        params = env.sim.get_params() if hasattr(env, "sim") else threelp.ThreeLPParams.Adult()
        ref_stride = threelp.build_canonical_stride(float(env.v_cmd), env.t_ds, env.t_ss, params)
        states = [np.asarray(s, dtype=np.float64) for s in log["x_can"]]
        actions = [np.asarray(a, dtype=np.float64) for a in log.get("actions", [])]
        if not states or not actions:
            return False
        # visualize_canonical_rollout expects len(actions) == len(states) - 1.
        n = min(len(states), len(actions) + 1)
        states = states[:n]
        actions = actions[: max(0, n - 1)]
        threelp.visualize_canonical_rollout(
            states=states,
            actions=actions,
            ref_stride=ref_stride,
            params=params,
            t_ds=env.t_ds,
            t_ss=env.t_ss,
            dense_substeps=120,
            show_reference=True,
            show_policy=True,
            loop=loop,
            wait_for_close=loop,
        )
        return True
    except Exception as exc:  # pragma: no cover - GUI path
        print(f"[viz] visualize_canonical_rollout error: {exc}")
        return False


def maybe_visualize(log: Dict[str, List], env: ThreeLPHighRateEnv, loop: bool = False, include_reference: bool = False) -> None:
    try:
        import threelp  # type: ignore
    except Exception as exc:
        print(f"[viz] threelp not available ({exc}); skipping visualization")
        return
    if not hasattr(threelp, "visualize_trajectory"):
        print("[viz] visualize_trajectory missing in pybind; rebuild with PY_VISUALIZER=ON")
        return

    # Try canonical rollout with reference overlay if requested and supported.
    if include_reference:
        if _visualize_with_reference(log, env, loop):
            return
        print("[viz] falling back to world-frame visualization (reference overlay unavailable)")

    states_raw = log.get("state_world", [])
    if not states_raw:
        print("[viz] no state_world data logged; re-run with updated env_high_rate_3lp.py")
        return

    state_objs = []
    for q in states_raw:
        st = threelp.ThreeLPState()
        st.q = [float(v) for v in q]
        state_objs.append(st)
    params = env.sim.get_params() if hasattr(env, "sim") else threelp.ThreeLPParams.Adult()
    try:
        threelp.visualize_trajectory(
            state_objs,
            env.t_ds,
            env.t_ss,
            params,
            fps=60.0,
            loop=loop,
            wait_for_close=loop,
        )
    except Exception as exc:  # pragma: no cover - GUI path
        print(f"[viz] visualize_trajectory error: {exc}")


def save_rollout(log: Dict[str, List], out_path: Path) -> None:
    payload = {k: np.asarray(v) for k, v in log.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    print(f"[save] rollout dumped to {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-rate 3LP visualization and sanity-check utility.")
    parser.add_argument("--algo", choices=["ppo", "rl", "zero", "random"], default="ppo", help="Policy type to roll out.")
    parser.add_argument("--policy", type=str, default=None, help="Path to weights (.pt for PPO, .npz for RL).")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dim for PPO ActorCritic (auto if omitted and .pt provided).")
    parser.add_argument("--steps", type=int, default=400, help="Max steps to roll out.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument("--v-cmd", type=float, default=None, help="Optional fixed command speed for the rollout.")
    parser.add_argument("--perturb", action="store_true", help="Add reset noise using env.reset(..., options={'perturb': True}).")
    parser.add_argument("--plot", type=str, default=None, help="If set, write torque plot to this path (PNG recommended).")
    parser.add_argument("--save-rollout", type=str, default=None, help="Optional .npz path to dump the rollout arrays.")
    parser.add_argument("--visualize", action="store_true", help="Replay rollout in Open3D if visualize_trajectory is available.")
    parser.add_argument("--visualize-ref", action="store_true", help="If visualize_canonical_rollout is available, show reference and policy together.")
    parser.add_argument("--loop", action="store_true", help="Loop playback window when --visualize is set.")
    parser.add_argument("--eval-noise", type=float, default=0.0, help="Optional Gaussian noise std added to policy output.")
    parser.add_argument("--check-backend", action="store_true", help="Print pybind/visualizer status and exit.")

    # Env parameters
    parser.add_argument("--t-ds", type=float, default=0.1)
    parser.add_argument("--t-ss", type=float, default=0.6)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-env-steps", type=int, default=2000)
    parser.add_argument("--action-clip", type=float, default=100.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    report = backend_report()
    _print_backend(report)
    if args.check_backend:
        return

    policy_path = Path(args.policy) if args.policy else None
    if policy_path is not None and not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    env = ThreeLPHighRateEnv(
        t_ds=args.t_ds,
        t_ss=args.t_ss,
        dt=args.dt,
        max_steps=args.max_env_steps,
        action_clip=args.action_clip,
    )
    rng = np.random.default_rng(args.seed)
    policy_fn, _ = _make_policy(args.algo, env, policy_path, args.hidden_dim, rng, eval_noise=args.eval_noise)

    log = rollout_policy(env, policy_fn, steps=args.steps, seed=args.seed, v_cmd=args.v_cmd, perturb=args.perturb)
    total_return = float(np.sum(log["rewards"])) if log["rewards"] else 0.0
    print(f"[rollout] steps={len(log['rewards'])} return={total_return:.2f} fallen={any(log['fallen'])}")

    if args.plot:
        plot_torques(log, env.dt, Path(args.plot))
    if args.save_rollout:
        save_rollout(log, Path(args.save_rollout))
    if args.visualize:
        maybe_visualize(log, env, loop=args.loop, include_reference=args.visualize_ref)


if __name__ == "__main__":
    main()
