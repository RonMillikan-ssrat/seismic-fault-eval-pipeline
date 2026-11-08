#!/usr/bin/env python3
"""Known-answer tests for evaluation/fault_metrics.py.

Synthetic sections hold vertical faults (one pixel wide), so distances are exact
whole numbers and every expected value can be worked out by hand.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
import fault_metrics as fm  # noqa: E402

S, H, W = 40, 128, 128
R0, R1 = 20, 108                 # fault rows [R0, R1) -> length L
L = R1 - R0
results = []


def check(label, expected, got, tol=1e-9):
    ok = (np.isnan(expected) and np.isnan(got)) if isinstance(expected, float) and np.isnan(expected) \
        else abs(float(expected) - float(got)) <= tol
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label:52s} expected {float(expected):8.4f}  got {float(got):8.4f}")


def faults(cols, shape=(S, H, W), rows=(R0, R1)):
    m = np.zeros(shape, bool)
    for c in cols:
        m[:, rows[0]:rows[1], c] = True
    return m


def at(c, thr):
    """P, R, F1 from Counts at one threshold value."""
    i = int(np.argmin(np.abs(c.thresholds - thr)))
    return fm.prf(*[x[i] if np.ndim(x) else x for x in c.totals()])


label = faults([30, 90])

# 1. perfect prediction
c = fm.curve_counts(label.astype(float), label, tol=0)
p, r, f = at(c, 0.5)
check("1 perfect: precision (tol 0)", 1.0, p)
check("1 perfect: recall (tol 0)", 1.0, r)
check("1 perfect: pixel Dice", 1.0, fm.dice_iou(label, label)[0])

# 2. shifted 2 samples sideways: zero below tol 2, perfect at tol 2
pred = faults([32, 92])
for tol, exp in [(0, 0.0), (1, 0.0), (1.9, 0.0), (2, 1.0), (3, 1.0)]:
    p, r, f = at(fm.curve_counts(pred.astype(float), label, tol=tol), 0.5)
    check(f"2 shift 2: F1 at tol {tol}", exp, f)
check("2 shift 2: pixel Dice", 0.0, fm.dice_iou(pred, label, 0)[0])
# Dilating a 1-px line by a radius-1 disk -> 3 columns plus 1 pixel at each end (3L + 2).
# The two dilated lines overlap in one column of L pixels.
check("2 shift 2: Dice dilated by 1  = 2L/(6L+4)", 2 * L / (6 * L + 4), fm.dice_iou(pred, label, 1)[0])

# 3. only one of the two faults found
p, r, f = at(fm.curve_counts(faults([30]).astype(float), label, tol=2), 0.5)
check("3 half found: precision", 1.0, p)
check("3 half found: recall", 0.5, r)
check("3 half found: F1 = 2/3", 2 / 3, f)

# 4. an extra false fault far from both labels
p, r, f = at(fm.curve_counts(faults([30, 60, 90]).astype(float), label, tol=2), 0.5)
check("4 false fault: precision = 2/3", 2 / 3, p)
check("4 false fault: recall", 1.0, r)

# 5. probabilities: true faults 0.75, false fault 0.25, background 0
prob = np.zeros((S, H, W))
prob[label] = 0.75
prob[faults([60])] = 0.25
s = fm.summarize(fm.curve_counts(prob, label, tol=2))
check("5 probabilistic: best F1", 1.0, s["best_f1"])
check("5 probabilistic: best threshold in (0.25, 0.75]", 1.0,
      float(0.25 < s["best_threshold"] <= 0.75))
check("5 probabilistic: AP", 1.0, s["ap"])
p, r, f = at(fm.curve_counts(prob, label, tol=2), 0.2)
check("5 probabilistic: precision at t=0.2 (includes false fault)", 2 / 3, p)

# 6. incomplete labels: interpreter labeled only the fault at 30, the model finds both
partial = faults([30])
c_all = fm.curve_counts(faults([30, 90]).astype(float), partial, tol=2)
c_reg = fm.curve_counts(faults([30, 90]).astype(float), partial, tol=2,
                        region=fm.label_region(partial, 10))
check("6 incomplete labels: precision, whole section", 0.5, at(c_all, 0.5)[0])
check("6 incomplete labels: precision inside label region", 1.0, at(c_reg, 0.5)[0])
check("6 incomplete labels: recall unchanged by region", at(c_all, 0.5)[1], at(c_reg, 0.5)[1])

# 7. histogram counts == direct counts, on a random smooth probability field
rng = np.random.default_rng(1)
prob = ndi.gaussian_filter(rng.random((6, 96, 96)), sigma=(0, 3, 3))
prob = (prob - prob.min()) / (prob.max() - prob.min())
lab = np.zeros(prob.shape, bool)
for s_ in range(6):                                  # a few random dipping line segments
    for _ in range(3):
        r0, c0 = rng.integers(10, 50, 2)
        for k in range(40):
            lab[s_, r0 + k, min(95, c0 + k // 3)] = True
tol = 2.5
c = fm.curve_counts(prob, lab, tol)
max_diff = 0
for t in [0.1, 0.33, 0.5, 0.62, 0.9]:
    i = int(np.argmin(np.abs(c.thresholds - t)))
    t = c.thresholds[i]
    pb = prob >= t
    direct = np.array([pb.sum(), (pb & (fm.distance_to(lab) <= tol)).sum(),
                       (lab & (fm.distance_to(pb) <= tol)).sum()])
    hist = np.array([c.n_pred[:, i].sum(), c.tp_pred[:, i].sum(), c.tp_label[:, i].sum()])
    max_diff = max(max_diff, int(np.abs(direct - hist).max()))
check("7 histogram sweep == direct EDT counts (max diff)", 0, max_diff)

# 8. bootstrap behavior
# 8a identical sections -> zero-width interval
c = fm.curve_counts(faults([30]).astype(float), label, tol=2)
b = fm.block_bootstrap(c, 50, block=5, n_boot=500)
check("8a identical sections: F1 CI width", 0.0, b["f1_ci"][1] - b["f1_ci"][0])

# 8b paired: A finds both faults; B misses the second fault on a random half of sections
miss = rng.random(S) < 0.5
pb = faults([30, 90])
pb[miss, :, 90] = False
ca = fm.curve_counts(label.astype(float), label, tol=2)
cb = fm.curve_counts(pb.astype(float), label, tol=2)
d = fm.paired_bootstrap(ca, 50, cb, 50, block=5, n_boot=2000)
check("8b paired: CI for F1(A)-F1(B) excludes 0", 1.0, float(d["diff_ci"][0] > 0))

# 8c autocorrelated sections: block bootstrap gives a wider interval than resampling
#     single sections. Recall drifts slowly from section to section, as on real inlines.
S2 = 200
frac = 0.55 + 0.35 * np.sin(2 * np.pi * np.arange(S2) / 60)
lab2 = faults([20], shape=(S2, 64, 64), rows=(8, 56))
pr2 = np.zeros_like(lab2)
for s_ in range(S2):
    n = int(round(frac[s_] * 48))
    pr2[s_, 8:8 + n, 20] = True
c2 = fm.curve_counts(pr2.astype(float), lab2, tol=1)
w1 = np.subtract(*fm.block_bootstrap(c2, 50, block=1, n_boot=3000)["recall_ci"][::-1])
w20 = np.subtract(*fm.block_bootstrap(c2, 50, block=20, n_boot=3000)["recall_ci"][::-1])
print(f"      recall CI width: block 1 = {w1:.4f}, block 20 = {w20:.4f}  (ratio {w20 / w1:.2f})")
check("8c block-20 CI wider than single-section CI", 1.0, float(w20 > 1.3 * w1))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
