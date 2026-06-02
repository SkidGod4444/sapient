#!/usr/bin/env python3
"""Convert a SNAC neural-audio-codec checkpoint to a SAPIENT-loadable safetensors.

SAPIENT is pure-Rust and loads **safetensors** (and GGUF), but the released SNAC
codec (e.g. ``hubertsiuzdak/snac_24khz``) ships only ``pytorch_model.bin`` — a
PyTorch pickle SAPIENT cannot read. This one-time, offline step:

  1. loads the SNAC model via the ``snac`` package (which applies ``weight_norm``),
  2. **folds every ``weight_norm`` parameter** into a plain ``weight`` tensor —
     ``w = g · v / ‖v‖`` with the norm over all dims except the output channel
     (dim 0), exactly matching SAPIENT's ``forward::snac::weight_norm_fold`` — so
     the Rust decoder needs no runtime renormalization,
  3. writes ``snac.safetensors`` + ``config.json``.

The codec decoder is fully convolutional; only the codec is converted here (the
LM backbone — Orpheus/OuteTTS — loads straight from its safetensors/GGUF release).

Usage:
    pip install snac torch safetensors
    python scripts/convert_snac_to_safetensors.py --repo hubertsiuzdak/snac_24khz --out ./snac_24khz

Output dir then holds ``snac.safetensors`` + ``config.json``, ready for SAPIENT's
SNAC decoder (Phase 6d, ``sapient speak``). Validated by the ignored
`snac_coherence` test (Rust decode vs torch reference, max_err ~2e-6).
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file
from snac import SNAC


def fold_weight_norm(state):
    """Fold weight_norm pairs into plain `weight` tensors.

    Handles torch's current parametrization API (`...parametrizations.weight.
    original0` = g, `original1` = v) and the legacy `weight_g`/`weight_v` names.
    """
    out = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith("parametrizations.weight.original1") or k.endswith("weight_v"):
            if k.endswith("original1"):
                base = k[: -len("parametrizations.weight.original1")]
                g = state[base + "parametrizations.weight.original0"]
                wkey = base + "weight"
            else:
                base = k[: -len("weight_v")]
                g = state[base + "weight_g"]
                wkey = base + "weight"
            dims = tuple(range(1, v.dim()))
            out[wkey] = (g * v / v.norm(2, dim=dims, keepdim=True)).contiguous()
        elif k.endswith("parametrizations.weight.original0") or k.endswith("weight_g"):
            continue  # consumed with its v
        else:
            out[k] = v.contiguous()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="hubertsiuzdak/snac_24khz", help="HF repo id")
    ap.add_argument("--out", default="./snac_24khz", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.repo} …")
    model = SNAC.from_pretrained(args.repo).eval()

    folded = fold_weight_norm(model.state_dict())
    print(f"Folded weights → {len(folded)} tensors.")
    save_file(folded, os.path.join(args.out, "snac.safetensors"))

    cfg = {
        k: getattr(model, k)
        for k in [
            "sampling_rate", "decoder_dim", "decoder_rates", "latent_dim",
            "codebook_size", "codebook_dim", "vq_strides", "attn_window_size",
        ]
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Wrote {args.out}/snac.safetensors + config.json — ready for `sapient speak`.")


if __name__ == "__main__":
    main()
