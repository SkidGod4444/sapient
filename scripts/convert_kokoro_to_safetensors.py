#!/usr/bin/env python3
"""Convert the Kokoro-82M PyTorch checkpoint + voice packs to safetensors.

Kokoro-82M (hexgrad/Kokoro-82M) ships only a PyTorch pickle (`kokoro-v1_0.pth`,
~327 MB f32) and per-voice `.pt` style packs (`[510, 1, 256]`). SAPIENT is
pure-Rust and never reads pickles at runtime, so — exactly like the SNAC path
that pulls the `mlx-community` safetensors mirror — we convert **once, offline**
to safetensors. The result can be hosted on a HF mirror or pointed at locally
via `SAPIENT_KOKORO_DIR`.

This produces, in the output dir:
  - `model.safetensors`   — all model weights (key names preserved verbatim)
  - `voices.safetensors`  — every downloaded voice as a `[510, 256]` tensor
                            keyed by voice name (squeezing the singleton axis)
  - `config.json`         — copied through unchanged
  - `kokoro_keys.txt`     — every weight key + shape (for the Rust loader)

Run:  python3 scripts/convert_kokoro_to_safetensors.py --out ~/.cache/kokoro-82m
Requires:  torch, safetensors, huggingface_hub  (dev-only; never at runtime).
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import save_file

REPO = "hexgrad/Kokoro-82M"
WEIGHTS = "kokoro-v1_0.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument(
        "--voices",
        default="",
        help="comma-separated voice names to convert (default: all in repo)",
    )
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # ── model weights ────────────────────────────────────────────────────────
    print(f"downloading {REPO}/{WEIGHTS} …")
    wpath = hf_hub_download(REPO, WEIGHTS)
    raw = torch.load(wpath, map_location="cpu", weights_only=True)
    # The checkpoint is a dict of 5 per-module sub-state-dicts
    # ({bert, bert_encoder, predictor, decoder, text_encoder}), each with an
    # inner "module." (DataParallel) prefix. Flatten to "{top}.{inner}".
    flat: dict[str, torch.Tensor] = {}
    for top, inner in raw.items():
        if isinstance(inner, dict):
            for k, v in inner.items():
                k = k[7:] if k.startswith("module.") else k
                flat[f"{top}.{k}"] = v
        else:
            flat[top] = inner
    # safetensors needs contiguous f32 tensors.
    sd = {k: v.contiguous().to(torch.float32) for k, v in flat.items()}

    keys_path = out / "kokoro_keys.txt"
    with keys_path.open("w") as fh:
        for k in sorted(sd):
            fh.write(f"{k}\t{tuple(sd[k].shape)}\n")
    print(f"wrote {len(sd)} weight keys → {keys_path}")

    save_file(sd, str(out / "model.safetensors"))
    print(f"wrote model.safetensors ({sum(v.numel() for v in sd.values())/1e6:.1f}M params)")

    # ── config ───────────────────────────────────────────────────────────────
    cfg = hf_hub_download(REPO, "config.json")
    shutil.copyfile(cfg, out / "config.json")
    print("copied config.json")

    # ── voices ───────────────────────────────────────────────────────────────
    repo_files = list_repo_files(REPO)
    voice_files = [f for f in repo_files if f.startswith("voices/") and f.endswith(".pt")]
    wanted = [v.strip() for v in args.voices.split(",") if v.strip()]
    if wanted:
        voice_files = [f for f in voice_files if Path(f).stem in wanted]
    voices: dict[str, torch.Tensor] = {}
    for vf in voice_files:
        name = Path(vf).stem
        vp = hf_hub_download(REPO, vf)
        t = torch.load(vp, map_location="cpu", weights_only=True)
        # voice packs are [510, 1, 256]; squeeze the singleton to [510, 256].
        t = t.squeeze(1).contiguous().to(torch.float32)
        voices[name] = t
    if voices:
        save_file(voices, str(out / "voices.safetensors"))
        any_shape = tuple(next(iter(voices.values())).shape)
        print(f"wrote voices.safetensors ({len(voices)} voices, each {any_shape})")
    else:
        print("no voice packs found/selected")

    print(f"\ndone → {out}")


if __name__ == "__main__":
    main()
