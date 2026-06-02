#!/usr/bin/env python3
"""Convert a SNAC neural-audio-codec checkpoint to a SAPIENT-loadable safetensors.

SAPIENT is pure-Rust and loads **safetensors** (and GGUF), but the released SNAC
codec (e.g. ``hubertsiuzdak/snac_24khz``) ships only ``pytorch_model.bin`` — a
PyTorch pickle SAPIENT cannot read. This one-time, offline step:

  1. downloads the SNAC checkpoint + config from the HF Hub,
  2. **folds every ``weight_norm`` parameter** (``weight_g`` / ``weight_v``) into a
     plain ``weight`` tensor — ``w = g · v / ‖v‖`` with the norm over all dims
     except the output channel (dim 0), exactly matching SAPIENT's
     ``forward::snac::weight_norm_fold`` — so the Rust decoder needs no runtime
     renormalization,
  3. writes ``snac.safetensors`` + copies ``config.json``.

Only the codec is converted here; the LM backbone (Orpheus / OuteTTS, a Llama-3.2)
is loaded straight from its existing safetensors/GGUF release.

Usage:
    pip install torch safetensors huggingface_hub
    python scripts/convert_snac_to_safetensors.py \
        --repo hubertsiuzdak/snac_24khz --out ./snac_24khz

The output directory then holds ``snac.safetensors`` + ``config.json``, ready for
SAPIENT's SNAC decoder (Phase 6d, ``sapient speak``).
"""

import argparse
import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file


def fold_weight_norm(state):
    """Fold weight_norm (weight_g/weight_v) pairs into plain `weight` tensors."""
    out = {}
    for k, t in state.items():
        if k.endswith("weight_g"):
            continue  # consumed alongside its weight_v
        if k.endswith("weight_v"):
            prefix = k[: -len("weight_v")]
            v = t
            g = state[prefix + "weight_g"]
            # L2 norm over every dim except the output-channel axis (dim 0).
            dims = tuple(range(1, v.dim()))
            norm = v.norm(p=2, dim=dims, keepdim=True)
            w = g * v / norm
            out[prefix + "weight"] = w.contiguous()
        else:
            out[k] = t.contiguous()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="hubertsiuzdak/snac_24khz", help="HF repo id")
    ap.add_argument("--out", default="./snac_24khz", help="output directory")
    ap.add_argument(
        "--weights-file",
        default="pytorch_model.bin",
        help="checkpoint filename in the repo",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Downloading {args.repo} …")
    ckpt = hf_hub_download(args.repo, args.weights_file)
    cfg_path = hf_hub_download(args.repo, "config.json")

    state = torch.load(ckpt, map_location="cpu")
    if "state_dict" in state:  # some checkpoints nest the params
        state = state["state_dict"]
    # Drop everything but float tensors (e.g. buffers stay; non-tensors skipped).
    state = {k: v for k, v in state.items() if torch.is_tensor(v)}

    folded = fold_weight_norm(state)
    n_folded = sum(1 for k in state if k.endswith("weight_v"))
    print(f"Folded {n_folded} weight_norm pairs; {len(folded)} tensors total.")

    out_weights = os.path.join(args.out, "snac.safetensors")
    save_file(folded, out_weights)

    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Wrote {out_weights}")
    print(f"Wrote {os.path.join(args.out, 'config.json')}")
    print("Done — point SAPIENT's SNAC decoder at this directory.")


if __name__ == "__main__":
    main()
