#!/usr/bin/env python3
"""Task success of a SmolVLA policy in the LIBERO simulator — LeRobot vs Sapient.

Runs LIBERO episodes through LeRobot's own environment wrapper and processors
(the exact observation pipeline `lerobot-eval` uses) and swaps only the policy:

  --backend lerobot   LeRobot's PyTorch SmolVLA, f32 on CPU (the reference)
  --backend sapient   `sapient serve` over HTTP (POST /v1/actions), at
                      --precision fast | balanced | exact

Both execute `--exec` actions of each 50-action chunk before asking for the
next one, so the comparison isolates the inference engine and its precision.
`--parity` instead compares one chunk from each backend on the same simulator
frame with the same start noise.

Each finished episode is appended to `--out` as one JSON line; re-running with
the same `--out` skips episodes already recorded.

Needs LeRobot with SmolVLA and LIBERO in a throwaway virtual environment. On
macOS LeRobot's `libero` extra is Linux-only; install `hf-libero==0.1.4` and
`robomimic==0.2.0` with `--no-deps` plus their other requirements (robosuite
1.4.0, bddl 1.0.1, mujoco<3.9, …) — the EGL probe they pull in does not build
on macOS and is not needed (rendering uses MUJOCO_GL=cgl).

Usage:
  sapient serve HuggingFaceVLA/smolvla_libero &
  python3 scripts/vla_sim_eval.py --backend sapient --precision fast \
      --suite libero_spatial --episodes 5 --out results.jsonl
  python3 scripts/vla_sim_eval.py --backend lerobot --suite libero_spatial \
      --episodes 5 --out results.jsonl
  python3 scripts/vla_sim_eval.py --parity --suite libero_spatial
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.request

os.environ.setdefault("MUJOCO_GL", "cgl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def data_uri(img: torch.Tensor) -> str:
    """[3, H, W] float in [0, 1] (exact k/255 values) → PNG data URI."""
    u8 = (img.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sapient_chunk(url, model, precision, obs, task, seed=0, noise=None, queued=None) -> np.ndarray:
    body = {
        "model": model,
        "task": task,
        "images": [data_uri(obs[k][0]) for k in IMAGE_KEYS],
        "state": obs["observation.state"][0].tolist(),
        "precision": precision,
        "seed": seed,
    }
    if noise is not None:
        body["noise"] = noise.tolist()
    if queued:
        body["queued"] = [np.asarray(a, dtype=np.float32).tolist() for a in queued]
    req = urllib.request.Request(
        url + "/v1/actions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return np.asarray(d["actions"], dtype=np.float32)  # [50, action_dim], robot units


class LeRobotPolicy:
    def __init__(self, repo):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained(repo)
        self.policy.to("cpu")
        self.policy.model.float()
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=repo,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def chunk(self, obs, task, noise=None) -> np.ndarray:
        batch = dict(obs)
        batch["task"] = [task]
        batch = self.pre(batch)
        with torch.inference_mode():
            kw = {} if noise is None else {"noise": torch.from_numpy(noise)[None]}
            actions = self.policy.predict_action_chunk(batch, **kw)  # [1, 50, action_dim]
        actions = self.post(actions)
        return actions[0].float().cpu().numpy()


def make_libero(suite: str, task_id: int):
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    cfg = LiberoEnv(task=suite, task_ids=[task_id])
    envs = make_env(cfg, n_envs=1)
    env = envs[suite][task_id]
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=None)
    return env, env_pre, env_post


def observe(env_pre, raw, task):
    from lerobot.envs.utils import preprocess_observation

    obs = preprocess_observation(raw)
    obs["task"] = [task]
    return env_pre(obs)


def auto_trigger(d: int, c: int) -> int:
    """The stall-minimizing rule (asynchronous whenever d < c).

    This is what `sapient act --threshold auto` did up to v0.6.2 and what the
    `--mode auto` results in benchmarks/ were measured with. Those results are
    why the engine's `auto_trigger` (vla_async.rs) now goes synchronous as soon
    as d > c/2.
    """
    return 0 if d >= c else min(d + d // 5 + 2, c)


def run_episode(env, env_pre, env_post, get_chunk, horizon, episode, task_id, delay=0, mode="sync"):
    """One episode. Every chunk contributes at most `horizon` actions.

    With `delay` d > 0, a chunk requested on tick t becomes usable on tick t + d
    (the simulator does not pause for inference); meanwhile the robot executes
    queued actions or, with none queued, holds its pose (zero motion, last gripper
    command) — a stall, which counts against the episode's step limit. Chunks are
    aligned by executed actions: the actions executed while a chunk was being
    computed are dropped from it. `mode`: `sync` requests only when the queue is
    empty; `auto` uses the stall-minimizing rule (see `auto_trigger`); `inpaint`
    requests like `auto` and also sends the actions still queued, which the
    policy keeps as the first rows of the new chunk and continues (hard
    inpainting, `POST /v1/actions` field `queued`).

    `jump` in the result is the mean, over chunk switches, of the largest change
    in any motion dimension between the last executed action and the first
    action taken from a new chunk.
    """
    raw, _ = env.reset(seed=[episode])
    task = list(env.call("task_description"))[0]
    max_steps = env.call("_max_episode_steps")[0]
    queue: list[np.ndarray] = []
    executed = 0  # actions executed so far
    pending = None  # (deliver_on_tick, executed_at_request, chunk)
    trigger = 0 if mode == "sync" else auto_trigger(delay, horizon)
    gripper = -1.0
    infer_s, n_infer, success, step, stalls = 0.0, 0, False, 0, 0
    last_action, switched, jumps = None, False, []
    while step < max_steps:
        if pending is not None and step >= pending[0]:
            _, at, chunk = pending
            pending = None
            usable = list(chunk[: horizon][max(0, executed - at) :])
            queue = usable  # newest plan replaces any overlap
            switched = True
        if pending is None and len(queue) <= trigger:
            obs = observe(env_pre, raw, task)
            # Common random numbers: every backend gets the same start noise
            # for the same (task, episode, chunk), so differences in outcome
            # come from the engine, not from sampling luck.
            rng = np.random.default_rng([task_id, episode, n_infer])
            noise = rng.standard_normal((50, 32)).astype(np.float32)
            t = time.perf_counter()
            chunk = get_chunk(obs, task, noise, list(queue) if mode == "inpaint" else None)
            infer_s += time.perf_counter() - t
            n_infer += 1
            pending = (step + delay, executed, chunk)
            if delay == 0:
                pending = None
                queue = list(chunk[:horizon])
                switched = True
        if queue:
            a = queue.pop(0)
            executed += 1
            gripper = float(a[-1])
            if switched and last_action is not None:
                jumps.append(float(np.abs(np.asarray(a)[:-1] - last_action[:-1]).max()))
            switched = False
            last_action = np.asarray(a)
        else:
            a = np.zeros(7, dtype=np.float32)
            a[-1] = gripper
            stalls += 1
        action = torch.from_numpy(np.asarray(a, dtype=np.float32))[None]
        action = env_post({"action": action})["action"]
        raw, _, terminated, truncated, info = env.step(action.numpy())
        step += 1
        if "final_info" in info:
            fi = info["final_info"]
            success = bool(np.asarray(fi.get("is_success", [False]))[0]) if isinstance(fi, dict) else False
        elif "is_success" in info:
            success = bool(np.asarray(info["is_success"])[0])
        if success or bool(np.asarray(terminated)[0]) or bool(np.asarray(truncated)[0]):
            break
    return {
        "success": success,
        "steps": step,
        "stalls": stalls,
        "jump": round(float(np.mean(jumps)), 4) if jumps else None,
        "inferences": n_infer,
        "infer_s": round(infer_s, 2),
        "task": task,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["lerobot", "sapient"], default="sapient")
    ap.add_argument("--precision", default="fast", help="sapient: fast | balanced | exact")
    ap.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0-9", help="task ids, e.g. 0-9 or 0,3,5")
    ap.add_argument("--episodes", type=int, default=5, help="init states per task (0..N-1)")
    ap.add_argument("--exec", type=int, default=10, help="actions executed per chunk (horizon c)")
    ap.add_argument("--delay", type=int, default=0, help="simulated inference delay d in control ticks")
    ap.add_argument(
        "--mode",
        choices=["sync", "auto", "inpaint"],
        default="sync",
        help="request policy; inpaint = auto + continue the queued actions (sapient only)",
    )
    ap.add_argument("--out", default="vla_sim_results.jsonl")
    ap.add_argument("--parity", action="store_true", help="compare one chunk from both backends")
    a = ap.parse_args()

    if "-" in a.tasks:
        lo, hi = map(int, a.tasks.split("-"))
        task_ids = list(range(lo, hi + 1))
    else:
        task_ids = [int(t) for t in a.tasks.split(",")]

    if a.parity:
        env, env_pre, _ = make_libero(a.suite, task_ids[0])
        raw, _ = env.reset(seed=[0])
        task = list(env.call("task_description"))[0]
        obs = observe(env_pre, raw, task)
        noise = torch.randn(50, 32, generator=torch.Generator().manual_seed(7)).numpy()
        ref = LeRobotPolicy(a.model).chunk(obs, task, noise)
        print(f"task: {task}")
        for prec in ("exact", "balanced", "fast"):
            got = sapient_chunk(a.url, a.model, prec, obs, task, noise=noise)
            d = got - ref
            print(
                f"sapient {prec:8} vs LeRobot f32: max {np.abs(d).max():.3e}  "
                f"rms {np.sqrt((d**2).mean()):.3e}  (ref rms {np.sqrt((ref**2).mean()):.3f})"
            )
        return

    label = "lerobot-f32" if a.backend == "lerobot" else f"sapient-{a.precision}"
    if a.delay or a.mode != "sync":
        label += f"-d{a.delay}-{a.mode}"
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if r["config"] == label and r["suite"] == a.suite and r["exec"] == a.exec:
                done.add((r["task_id"], r["episode"]))

    if a.backend == "lerobot":
        pol = LeRobotPolicy(a.model)
        if a.mode == "inpaint":
            raise SystemExit("--mode inpaint needs --backend sapient")
        get_chunk = lambda obs, task, noise, queued=None: pol.chunk(obs, task, noise)  # noqa: E731
    else:
        get_chunk = lambda obs, task, noise, queued=None: sapient_chunk(  # noqa: E731
            a.url, a.model, a.precision, obs, task, noise=noise, queued=queued
        )

    for tid in task_ids:
        if all((tid, ep) in done for ep in range(a.episodes)):
            continue
        # LeRobot's LIBERO env walks through the task's fixed init states, one
        # per reset — so episodes run in order, and recorded ones still reset.
        env, env_pre, env_post = make_libero(a.suite, tid)
        for ep in range(a.episodes):
            if (tid, ep) in done:
                env.reset(seed=[ep])
                continue
            t = time.perf_counter()
            r = run_episode(env, env_pre, env_post, get_chunk, a.exec, ep, tid, a.delay, a.mode)
            r.update(
                config=label,
                suite=a.suite,
                task_id=tid,
                episode=ep,
                exec=a.exec,
                delay=a.delay,
                mode=a.mode,
                wall_s=round(time.perf_counter() - t, 1),
            )
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(
                f"{label} task {tid} ep {ep}: {'SUCCESS' if r['success'] else 'fail'} "
                f"in {r['steps']} steps ({r['stalls']} stalled), {r['inferences']} chunks, {r['wall_s']} s",
                flush=True,
            )
        env.close()


if __name__ == "__main__":
    main()
