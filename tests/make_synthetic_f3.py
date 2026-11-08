#!/usr/bin/env python3
"""Write synthetic files in the same layout as seismic-fault-labeling's F3 outputs.

  <out>/inlines/labels_sparse.npy   uint8 (il, xl, t): 1 fault, 0 background, 255 unlabeled
  <out>/inlines/meta.json           il_numbers, xl_first
  <out>/pred/prob_crop.npy          float32 (il, xl, t) on a crop
  <out>/pred/crop.json              {"lo": [...], "hi": [...]}

Two real faults; the "interpreter" picks only fault A, on every 10th inline, as
3-pixel-wide sticks. The "model" sees both faults, blurred, plus a little noise.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def make(out):
    out = Path(out)
    (out / "inlines").mkdir(parents=True, exist_ok=True)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n_il, n_xl, n_t = 120, 200, 150
    lab = np.full((n_il, n_xl, n_t), 255, np.uint8)
    truth = np.zeros((n_il, n_xl, n_t), bool)
    for i in range(n_il):
        for t in range(20, 130):
            truth[i, 60 + (t - 20) // 4 + i // 20, t] = True     # fault A (picked)
            truth[i, 150 - (t - 20) // 5, t] = True             # fault B (never picked)
    for i in range(0, 100, 10):
        pick = truth[i].copy()
        pick[140:, :] = False                                    # only fault A
        sec = np.zeros((n_xl, n_t), np.uint8)
        sec[ndi.binary_dilation(pick, iterations=1)] = 1         # 3-px sticks
        lab[i] = sec
    np.save(out / "inlines" / "labels_sparse.npy", lab)
    (out / "inlines" / "meta.json").write_text(json.dumps(
        {"il_numbers": list(range(100, 100 + n_il)), "xl_first": 300}))
    lo, hi = [0, 30, 10], [100, 190, 140]
    prob = ndi.gaussian_filter(truth.astype(np.float32), (0.7, 1.2, 1.2)) * 2.2
    prob = np.clip(prob + 0.15 * rng.random(prob.shape), 0, 1)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    np.save(out / "pred" / "prob_crop.npy", prob.astype(np.float32))
    (out / "pred" / "crop.json").write_text(json.dumps({"lo": lo, "hi": hi}))
    return out


if __name__ == "__main__":
    print("wrote", make(sys.argv[1] if len(sys.argv) > 1 else "/tmp/f3sim"))
