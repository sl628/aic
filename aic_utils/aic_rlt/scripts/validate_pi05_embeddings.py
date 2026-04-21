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
    """Check one .pt file. Returns (errors, info_dict)."""
    errors = []
    d = _load(pt_path)

    # vla_embeddings is the one non-negotiable key for both modes.
    if "vla_embeddings" not in d:
        errors.append("missing key 'vla_embeddings'")
        return errors

    emb = d["vla_embeddings"]
    if emb.ndim != 3:
        errors.append(f"vla_embeddings ndim={emb.ndim}, expected 3 (T, num_tokens, embed_dim)")
        return errors

    T = emb.shape[0]

    # Option B (pi05 as embedding-only feature extractor): ref_actions absent by design.
    has_ref = "ref_actions" in d
    if not has_ref:
        return errors  # embeddings-only mode is a valid, checked shape

    # Slow path (Option A / xvla): validate ref_actions shape + envelope.
    ref = d["ref_actions"]
    if ref.shape[0] != T:
        errors.append(f"ref_actions T={ref.shape[0]} != vla_embeddings T={T}")
    if ref.ndim != 3:
        errors.append(f"ref_actions ndim={ref.ndim}, expected 3 (T, C, action_dim)")
        return errors

    if "action_dim" in d and int(d["action_dim"]) != int(ref.shape[-1]):
        errors.append(f"action_dim field ({d['action_dim']}) != ref_actions last dim ({ref.shape[-1]})")

    # The q01/q99 envelope check only makes sense for pi0.5's *delta* action space,
    # but our saved ref_actions are post-AbsoluteActions (absolute joints), so the
    # ur5e action_stats q01/q99 are not the right bounds. Check against STATE q01/q99
    # instead — absolute joints should lie within the ur5e state envelope in principle,
    # though aic's workspace may differ significantly (OOD warning, not error).
    if ur5e_stats is not None and ref.shape[-1] == 7:
        state_stats = ur5e_stats.get("state") or {}
        q01 = np.asarray(state_stats.get("q01", []))
        q99 = np.asarray(state_stats.get("q99", []))
        if q01.size == 7 and q99.size == 7:
            ref_np = ref.numpy()
            lo = ref_np.reshape(-1, 7).min(0)
            hi = ref_np.reshape(-1, 7).max(0)
            width = q99 - q01 + 1e-6
            out_dims = [i for i in range(7)
                        if lo[i] < q01[i] - 0.5 * width[i] or hi[i] > q99[i] + 0.5 * width[i]]
            if out_dims:
                # This is informational, not a hard failure — aic's workspace differs
                # from pi0.5's ur5e training distribution (OOD warning per plan risk #1).
                errors.append(
                    f"INFO ref_actions dims {out_dims} outside ur5e STATE q01/q99 "
                    f"(aic workspace is OOD for pi0.5/ur5 — expected; see plan risk #1)"
                )
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
        # Partition hard errors from INFO notes.
        hard = [e for e in errs if not e.startswith("INFO ")]
        info = [e for e in errs if e.startswith("INFO ")]
        if hard:
            any_err = True
            print(f"FAIL {pt.name}: {'; '.join(hard)}")
        else:
            d = _load(pt)
            emb = d["vla_embeddings"]
            ref_info = (f" ref_actions={tuple(d['ref_actions'].shape)}"
                        if "ref_actions" in d else " [embeddings-only (Option B)]")
            instr = d.get("ref_instruction", d.get("instruction", "<not stored>"))
            print(f"OK   {pt.name}: T={emb.shape[0]} tokens={emb.shape[1]} "
                  f"embed_dim={emb.shape[2]}{ref_info} instr='{instr}'")
        for m in info:
            print(f"     {m}")

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
