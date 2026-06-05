#!/usr/bin/env python3
"""Text-driven STS (speech-to-speech) tester for SAPIENT — no microphone needed.

You type a prompt in the terminal; the script drives the `sapient` CLI under the
hood to (1) generate a text reply with an LLM, then (2) synthesize that reply to
voice and play it through the default audio output (e.g. a Bluetooth speaker on a
Raspberry Pi). It's the text-in / voice-out half of the `converse` pipeline,
scriptable and mic-free — handy for testing TTS + playback over SSH.

How it works
------------
  prompt → `sapient chat <llm> --prompt ...`  → reply text (captured from stdout)
         → `sapient speak <tts> "<reply>"`     → audio played on the device

`chat --prompt` runs ONE chat turn with the model's chat template + end-of-turn
stopping, so the reply is a clean bounded answer (unlike `run`, which does a raw
completion that rambles and repeats). It prints only the reply to stdout.

Prerequisites for audio on a Raspberry Pi over SSH
--------------------------------------------------
  * The ALSA→PipeWire bridge must be installed:  sudo apt install -y pipewire-alsa
  * XDG_RUNTIME_DIR must point at your user runtime dir so cpal finds the audio
    socket. This script sets it automatically (to /run/user/<uid>) if it's unset.

Usage
-----
  python3 sts_test.py                         # interactive loop, defaults
  python3 sts_test.py --once "Tell me a joke" # single turn then exit
  python3 sts_test.py --llm openhorizon/qwen2.5-1.5b --tts openhorizon/kokoro-82m
  python3 sts_test.py --voice af_bella --history   # keep conversation context
  python3 sts_test.py --no-play -o reply.wav       # write WAV only (no playback)

Type 'exit', 'quit', or Ctrl-D to leave the loop.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Strips ANSI escape sequences (colors, spinner cursor moves) from captured text.
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def find_sapient(explicit: str | None) -> str:
    """Locate the `sapient` binary. The SSH non-login shell often lacks the PATH
    entries a desktop login has, so check the common install locations too."""
    if explicit:
        if Path(explicit).is_file():
            return explicit
        sys.exit(f"error: --sapient path not found: {explicit}")
    found = shutil.which("sapient")
    if found:
        return found
    for cand in (
        Path.home() / ".local/bin/sapient",
        Path("/usr/local/bin/sapient"),
        Path.home() / ".cargo/bin/sapient",
    ):
        if cand.is_file():
            return str(cand)
    sys.exit(
        "error: `sapient` not found on PATH or in ~/.local/bin, /usr/local/bin, "
        "~/.cargo/bin.\n       Pass it explicitly with --sapient /path/to/sapient"
    )


def ensure_audio_env() -> None:
    """On Linux, make sure XDG_RUNTIME_DIR is set so cpal can reach PipeWire/Pulse.
    Without it, playback fails with 'no audio output device' over SSH."""
    if sys.platform.startswith("linux") and not os.environ.get("XDG_RUNTIME_DIR"):
        runtime = f"/run/user/{os.getuid()}"
        if Path(runtime).is_dir():
            os.environ["XDG_RUNTIME_DIR"] = runtime
            print(f"[info] set XDG_RUNTIME_DIR={runtime}", file=sys.stderr)


def generate_reply(sapient: str, llm: str, prompt: str, backend: str) -> str:
    """Run one chat turn and return the reply. `sapient chat --prompt` applies the
    chat template + end-of-turn stopping (so the reply is a clean, bounded answer)
    and prints only the reply to stdout; progress/load lines go to stderr, which
    we let stream to the terminal so you still see download/load progress."""
    proc = subprocess.run(
        [sapient, "chat", llm, "--prompt", prompt, "--backend", backend],
        stdout=subprocess.PIPE,
        stderr=None,  # inherit → progress shown live
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"`sapient chat` exited with code {proc.returncode}")
    return ANSI_RE.sub("", proc.stdout).strip()


def speak(
    sapient: str,
    tts: str,
    text: str,
    voice: str | None,
    backend: str,
    out: str,
    no_play: bool,
) -> None:
    """Synthesize `text` and (unless --no-play) play it on the default output."""
    cmd = [sapient, "speak", tts, text, "--backend", backend, "-o", out]
    if voice:
        cmd += ["--voice", voice]
    if no_play:
        cmd += ["--no-play"]  # needs a recent sapient build; harmless to omit otherwise
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        # A negative return code on Unix means the process was killed by a signal
        # (e.g. -6 = SIGABRT from a Rust panic such as the Kokoro token-index bug).
        if proc.returncode < 0:
            raise RuntimeError(
                f"`sapient speak` crashed (signal {-proc.returncode}). "
                "If this is the Kokoro index-out-of-bounds panic, try a different "
                "reply, or use --tts openhorizon/orpheus-3b."
            )
        raise RuntimeError(f"`sapient speak` exited with code {proc.returncode}")


def build_prompt(turns: list[tuple[str, str]], user_text: str, system: str | None) -> str:
    """With --history, fold prior turns into a single prompt so the one-shot
    `run` path has conversational context. Without it, just the latest message."""
    if not turns and not system:
        return user_text
    parts: list[str] = []
    if system:
        parts.append(system)
    for u, a in turns:
        parts.append(f"User: {u}")
        parts.append(f"Assistant: {a}")
    parts.append(f"User: {user_text}")
    parts.append("Assistant:")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Text-driven STS tester for SAPIENT (no mic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--llm", default="openhorizon/qwen2.5-0.5b-q4", help="LLM model alias for the text reply")
    ap.add_argument("--tts", default="openhorizon/kokoro-82m", help="TTS model alias for the voice")
    ap.add_argument("--voice", default=None, help="Voice name (defaults to the TTS model's default)")
    ap.add_argument("--backend", default="auto", help="Inference backend: auto | cpu | metal | wgpu")
    ap.add_argument("--system", default=None, help="Optional system prompt prefix (implies context)")
    ap.add_argument("--history", action="store_true", help="Keep conversation context across turns")
    ap.add_argument("--no-play", action="store_true", help="Write the WAV but don't play it")
    ap.add_argument("-o", "--out", default="reply.wav", help="WAV output path")
    ap.add_argument("--sapient", default=None, help="Path to the sapient binary")
    ap.add_argument("--once", default=None, help="Run a single prompt then exit (non-interactive)")
    args = ap.parse_args()

    sapient = find_sapient(args.sapient)
    ensure_audio_env()
    print(f"[info] sapient={sapient}", file=sys.stderr)
    print(f"[info] llm={args.llm}  tts={args.tts}  backend={args.backend}", file=sys.stderr)

    turns: list[tuple[str, str]] = []

    def handle(user_text: str) -> None:
        user_text = user_text.strip()
        if not user_text:
            return
        prompt = build_prompt(turns if args.history else [], user_text, args.system)
        t0 = time.time()
        reply = generate_reply(sapient, args.llm, prompt, args.backend)
        t_llm = time.time() - t0
        if not reply:
            print("[warn] empty reply from LLM; skipping TTS", file=sys.stderr)
            return
        print(f"\n🤖 {reply}\n   (LLM {t_llm:.1f}s)\n")
        t1 = time.time()
        speak(sapient, args.tts, reply, args.voice, args.backend, args.out, args.no_play)
        print(f"   (TTS+play {time.time() - t1:.1f}s)\n", file=sys.stderr)
        if args.history:
            turns.append((user_text, reply))

    # Single-shot mode.
    if args.once is not None:
        try:
            handle(args.once)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    # Interactive loop.
    print("Type a message and press Enter. 'exit'/'quit' or Ctrl-D to stop.\n")
    while True:
        try:
            user_text = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.strip().lower() in {"exit", "quit", ":q"}:
            break
        try:
            handle(user_text)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            # Keep the loop alive so a single crash doesn't end the session.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
