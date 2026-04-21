#!/usr/bin/env python3
"""Quick sanity check for pi05-extracted `.pt` files.

Verifies:
  1. Expected keys present (vla_embeddings, ref_actions, action_dim, ref_instruction).
  2. Shapes match dataset's frame count T across all tensors.
  3. action_dim == 7 and ref_actions stay within ur5e norm_stats q01/q99 envelope.
  4. (Optional) embeddings actually depend on the instruction — compare against a
     second extraction run with a different prompt (pass --compare_dir).

Usage:
    python validate_pi05_embeddings.py --embeddings_dir /home/yifeng/aic_data/embeddings_pi05
    python validate_pi05_embeddings.py --embeddings_dir .../embeddings_pi05 --compare_dir .../embeddings_pi05_alt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def check_one(pt_path: Path, ur5e_stats: dict | None) -> list[str]:
    errors = []
    d = _load(pt_path)

    # 1. Expected keys
    for k in ("vla_embeddings", "ref_actions", "action_dim", "ref_instruction"):
        if k not in d:
            errors.append(f"missing key '{k}'")

    if errors:
        return errors

    T = d["vla_embeddings"].shape[0]
    ref = d["ref_actions"]

    # 2. Shape consistency
    if ref.shape[0] != T:
        errors.append(f"ref_actions T={ref.shape[0]} != vla_embeddings T={T}")
    if ref.ndim != 3:
        errors.append(f"ref_actions ndim={ref.ndim}, expected 3 (T, C, action_dim)")

    # 3. action_dim and envelope
    if int(d["action_dim"]) != int(ref.shape[-1]):
        errors.append(f"action_dim field ({d['action_dim']}) != ref_actions last dim ({ref.shape[-1]})")
    if int(d["action_dim"]) != 7:
        errors.append(f"action_dim={d['action_dim']}, expected 7 for ur5e")

    if ur5e_stats is not None and ref.ndim == 3:
        q01 = np.asarray(ur5e_stats["actions"]["q01"])
        q99 = np.asarray(ur5e_stats["actions"]["q99"])
        ref_np = ref.numpy()          # (T, C, 7)
        # Collapse over T and C to per-dim range
        lo = ref_np.reshape(-1, ref_np.shape[-1]).min(0)
        hi = ref_np.reshape(-1, ref_np.shape[-1]).max(0)
        # Generous tolerance: actions can exceed q01/q99 by design since those are
        # 1st/99th percentiles (not clip bounds). Warn if > ~50% outside both sides.
        width = q99 - q01 + 1e-6
        if (lo < q01 - 0.5 * width).any() or (hi > q99 + 0.5 * width).any():
            out_dims = [i for i in range(len(q01))
                        if lo[i] < q01[i] - 0.5 * width[i] or hi[i] > q99[i] + 0.5 * width[i]]
            errors.append(f"ref_actions dims {out_dims} far outside ur5e q01/q99 envelope "
                          f"(lo={lo.round(3).tolist()} hi={hi.round(3).tolist()})")

    return errors


def compare_instruction_dependence(dir_a: Path, dir_b: Path) -> list[str]:
    """Same episode, different instructions — embeddings should differ."""
    errs = []
    files_a = sorted(dir_a.glob("episode_*.pt"))
    files_b = sorted(dir_b.glob("episode_*.pt"))
    common = set(p.name for p in files_a) & set(p.name for p in files_b)
    if not common:
        errs.append(f"no overlapping episode_*.pt between {dir_a} and {dir_b}")
        return errs
    name = sorted(common)[0]
    a = _load(dir_a / name)["vla_embeddings"].float()
    b = _load(dir_b / name)["vla_embeddings"].float()
    if a.shape != b.shape:
        errs.append(f"embedding shapes differ ({a.shape} vs {b.shape}) — can't compare")
        return errs
    # Relative Frobenius distance: ||a - b|| / (||a|| + ||b||)
    num = torch.linalg.norm(a - b).item()
    denom = (torch.linalg.norm(a) + torch.linalg.norm(b)).item()
    rel = num / max(denom, 1e-6)
    print(f"  instruction-dependence signal (relative Frobenius): {rel:.4f}")
    if rel < 1e-4:
        errs.append(f"embeddings for '{dir_a.name}' and '{dir_b.name}' are effectively "
                    f"identical (rel={rel:.2e}) — language ablation likely still present")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--compare_dir", default=None,
                    help="Second embeddings dir extracted with a different instruction "
                         "(validation check #4 — embeddings should differ)")
    ap.add_argument("--ur5e_norm_stats",
                    default="/home/yifeng/workspace/pi05_base/pi05_base/assets/ur5e/norm_stats.json")
    args = ap.parse_args()

    ur5e = None
    try:
        with open(args.ur5e_norm_stats) as f:
            ur5e = json.load(f)["norm_stats"]
    except Exception as e:
        print(f"(skipping q01/q99 check: {e})")

    emb_dir = Path(args.embeddings_dir)
    files = sorted(emb_dir.glob("episode_*.pt"))
    if not files:
        print(f"FAIL: no episode_*.pt in {emb_dir}")
        sys.exit(2)

    any_err = False
    for pt in files:
        errs = check_one(pt, ur5e)
        if errs:
            any_err = True
            print(f"FAIL {pt.name}: {'; '.join(errs)}")
        else:
            d = _load(pt)
            print(f"OK   {pt.name}: T={d['vla_embeddings'].shape[0]} "
                  f"ref_actions={tuple(d['ref_actions'].shape)} "
                  f"instr='{d['ref_instruction']}'")

    if args.compare_dir:
        print("\nChecking instruction-dependence ...")
        errs = compare_instruction_dependence(emb_dir, Path(args.compare_dir))
        for e in errs:
            any_err = True
            print(f"FAIL compare: {e}")
        if not errs:
            print("OK: embeddings depend on instruction (language not ablated)")

    sys.exit(1 if any_err else 0)


if __name__ == "__main__":
    main()
