#!/usr/bin/env python3
"""Evaluation metrics for fault segmentation on seismic sections.

Arrays are stacks of 2-D sections: shape (n_sections, n_rows, n_cols), e.g.
(inline, time sample, crossline). Every metric is computed per section and then
summed, so a bootstrap can resample sections (or blocks of sections) cheaply.

Tolerance-based precision and recall
------------------------------------
Faults are 1-2 samples wide, so a prediction one sample off a label would score
zero with plain pixel overlap. With tolerance tol (in samples):
  precision = predicted pixels within tol of a labeled pixel / predicted pixels
  recall    = labeled pixels within tol of a predicted pixel / labeled pixels
The two directions are different questions, so they use different counts.

How the whole PR curve comes from one pass
------------------------------------------
Precision side: the label mask does not depend on the threshold, so compute once
  near_label = (distance to the nearest labeled pixel) <= tol   (Euclidean distance
  transform). At threshold t, matched predictions = pixels with prob >= t inside
  near_label. Counting "values >= t" for every t is one sort + searchsorted.
Recall side: a labeled pixel is matched at threshold t if some pixel within tol
  has prob >= t, i.e. if the MAXIMUM prob over a disk of radius tol is >= t.
  So compute one maximum filter (disk footprint) and count again with searchsorted.
Result: counts for every threshold without re-running anything per threshold.

Incomplete labels
-----------------
When an interpreter labeled only some faults, real but unlabeled faults count as
false positives. Passing region= (e.g. label_region(label, radius)) counts
predictions only inside that region, which estimates precision where labels are
trustworthy. Recall is unaffected: it only looks at labeled pixels.

Block bootstrap
---------------
Neighboring inlines are nearly identical, so resampling single sections treats
correlated samples as independent and gives intervals that are too narrow.
Resampling contiguous blocks of sections keeps that correlation inside each block.
Non-overlapping blocks are used; the last block may be shorter.

Threshold selection bias: choosing the best-F1 threshold on the same sections
you report on is optimistic. For a fair number, choose the threshold on one set
of inlines and report on another (pick_threshold + evaluate at that threshold).
"""
import argparse
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

# Steps of 0.01, plus 0.001 steps above 0.99: unthinned fault maps often reach their
# best F1 at very high thresholds, where a 0.01 grid is too coarse.
DEFAULT_THRESHOLDS = np.unique(np.round(np.concatenate(
    [np.linspace(0.0, 1.0, 101), np.linspace(0.99, 1.0, 11)]), 10))


# ------------------------------------------------------------------- helpers
def disk(radius):
    """Boolean disk footprint: offsets with dx^2 + dy^2 <= radius^2."""
    k = int(np.floor(radius))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= radius * radius + 1e-9


def distance_to(mask):
    """Per section: Euclidean distance (in samples) from each pixel to the nearest True pixel.
    Sections with no True pixels get +inf."""
    mask = np.asarray(mask, bool)
    out = np.empty(mask.shape, dtype=np.float64)
    for i, m in enumerate(mask):
        out[i] = ndi.distance_transform_edt(~m) if m.any() else np.inf
    return out


def dilate(mask, radius):
    """Per-section binary dilation with a disk (no dilation across sections)."""
    mask = np.asarray(mask, bool)
    if radius <= 0:
        return mask.copy()
    return ndi.binary_dilation(mask, structure=disk(radius)[None, :, :])


def label_region(label, radius):
    """Region where labels are trusted: within `radius` samples of a labeled fault."""
    return dilate(label, radius)


# Neighbor offsets (row, col) along each quantized gradient direction.
# Rows are axis 0 (time, down), cols are axis 1 (crossline, right);
# the angle is atan2(d/drow, d/dcol) folded into [0, 180).
_NMS_OFFSETS = [(0, 1),    # ~0 deg:   prob changes along cols -> compare left/right
                (1, 1),    # ~45 deg
                (1, 0),    # ~90 deg:  prob changes along rows -> compare up/down
                (1, -1)]   # ~135 deg


def thin_faults(prob, sigma=1.0):
    """Thin fault-probability bands to ~1-pixel ridges by non-maximum suppression.

    Per section:
      1. Smooth with a Gaussian (sigma, in samples) so a flat-topped band gets a
         single peak at its center.
      2. Sobel gradients give the direction across the band (the direction the
         probability changes fastest), quantized to 0, 45, 90 or 135 degrees.
      3. Keep a pixel only if its smoothed value is a peak along that direction:
         greater than the neighbor behind it and >= the neighbor ahead. The
         asymmetric test keeps exactly one pixel where two neighbors tie.
    Kept pixels keep their ORIGINAL probability; all others become 0. The result is
    still a probability map, so curve_counts() sweeps thresholds on it as before.

    Side effect: at the END of a band the probability changes fastest along the
    band, so the comparison runs along it and the last ~2-3 pixels are dropped
    (Canny edge thinning shortens line ends the same way). This costs a little
    recall at fault tips.

    Thinning is 2-D, in the plane of each section. True 3-D thinning would act
    across the fault surface; per-section is consistent with per-inline scoring.
    """
    prob = np.asarray(prob, dtype=np.float32)
    out = np.zeros_like(prob)
    for s in range(prob.shape[0]):
        p = ndi.gaussian_filter(prob[s], sigma) if sigma > 0 else prob[s]
        g_row, g_col = ndi.sobel(p, axis=0), ndi.sobel(p, axis=1)
        q = np.round(((np.degrees(np.arctan2(g_row, g_col)) + 180.0) % 180.0) / 45.0).astype(int) % 4
        h, w = p.shape
        pad = np.pad(p, 1, mode="constant", constant_values=-np.inf)
        keep = np.zeros(p.shape, bool)
        for k, (dr, dc) in enumerate(_NMS_OFFSETS):
            ahead = pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
            behind = pad[1 - dr:1 - dr + h, 1 - dc:1 - dc + w]
            keep |= (q == k) & (p > behind) & (p >= ahead)
        out[s] = np.where(keep, prob[s], 0.0)
    return out


def _count_ge(values, thresholds):
    """For each threshold t, how many values are >= t (sorted once, then searchsorted)."""
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    return v.size - np.searchsorted(v, thresholds, side="left")


# --------------------------------------------------------------- counting
@dataclass
class Counts:
    """Per-section counts at every threshold. Sum over sections to get totals."""
    thresholds: np.ndarray   # (T,)
    n_pred: np.ndarray       # (S, T) predicted pixels (inside region) with prob >= t
    tp_pred: np.ndarray      # (S, T) ... of those, within tol of a labeled pixel
    tp_label: np.ndarray     # (S, T) labeled pixels within tol of a pixel with prob >= t
    n_label: np.ndarray      # (S,)   labeled pixels
    tol: float

    @property
    def n_sections(self):
        return self.n_label.shape[0]

    def totals(self, sections=None):
        s = slice(None) if sections is None else sections
        return (self.n_pred[s].sum(axis=0), self.tp_pred[s].sum(axis=0),
                self.tp_label[s].sum(axis=0), self.n_label[s].sum())


def curve_counts(prob, label, tol, thresholds=DEFAULT_THRESHOLDS, region=None):
    """Counts for the full precision-recall curve (see module docstring)."""
    prob = np.asarray(prob, dtype=np.float64)
    label = np.asarray(label, bool)
    if prob.shape != label.shape or prob.ndim != 3:
        raise ValueError(f"prob {prob.shape} and label {label.shape} must be the same 3-D shape")
    region = np.ones_like(label) if region is None else np.asarray(region, bool)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    near_label = distance_to(label) <= tol
    # Max prob within the tolerance disk. Outside the array counts as "no prediction".
    reach = ndi.maximum_filter(prob, footprint=disk(tol)[None, :, :], mode="constant", cval=-1.0)

    S, T = prob.shape[0], thresholds.size
    n_pred = np.zeros((S, T), np.int64)
    tp_pred = np.zeros((S, T), np.int64)
    tp_label = np.zeros((S, T), np.int64)
    for s in range(S):
        r = region[s]
        n_pred[s] = _count_ge(prob[s][r], thresholds)
        tp_pred[s] = _count_ge(prob[s][r & near_label[s]], thresholds)
        tp_label[s] = _count_ge(reach[s][label[s]], thresholds)
    return Counts(thresholds, n_pred, tp_pred, tp_label, label.sum(axis=(1, 2)).astype(np.int64), tol)


def prf(n_pred, tp_pred, tp_label, n_label):
    """Precision, recall, F1 from summed counts. NaN where undefined (no predictions / no labels)."""
    n_pred, tp_pred, tp_label = (np.asarray(a, np.float64) for a in (n_pred, tp_pred, tp_label))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_pred > 0, tp_pred / n_pred, np.nan)
        r = tp_label / n_label if n_label > 0 else np.full_like(tp_label, np.nan)
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
        f1 = np.where(np.isnan(p) | np.isnan(r), np.nan, f1)
    return p, r, f1


def average_precision(p, r):
    """Area under the PR curve as a step sum: sum over thresholds (high -> low) of
    (R_k - R_{k-1}) * P_k, starting from R = 0. Thresholds with no predictions are skipped.
    The step sum is used instead of the trapezoid rule, which can overstate the area."""
    order = np.arange(len(p))[::-1]            # thresholds ascending -> walk high to low
    ap, r_prev = 0.0, 0.0
    for k in order:
        if np.isnan(p[k]):
            continue
        ap += max(r[k] - r_prev, 0.0) * p[k]
        r_prev = max(r_prev, r[k])
    return ap


def summarize(counts, sections=None):
    """Curve, AP and best-F1 threshold for the given sections (all by default)."""
    p, r, f1 = prf(*counts.totals(sections))
    best = int(np.nanargmax(f1))
    return dict(thresholds=counts.thresholds, precision=p, recall=r, f1=f1,
                ap=average_precision(p, r), best_index=best,
                best_threshold=float(counts.thresholds[best]),
                best_precision=float(p[best]), best_recall=float(r[best]), best_f1=float(f1[best]))


def pick_threshold(counts, sections):
    """Best-F1 threshold index chosen on `sections` only (e.g. a calibration set of inlines)."""
    return summarize(counts, sections)["best_index"]


# ----------------------------------------------------------- overlap scores
def dice_iou(pred, label, radius=0):
    """Dice and IoU of binary masks, optionally after dilating both by `radius`.
    radius=0 is the plain pixel score, which is harsh on 1-pixel-wide faults."""
    a, b = dilate(pred, radius), dilate(label, radius)
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    union = sa + sb - inter
    dice = 2 * inter / (sa + sb) if (sa + sb) else np.nan
    iou = inter / union if union else np.nan
    return float(dice), float(iou)


# --------------------------------------------------------------- bootstrap
def _blocks(n_sections, block):
    return [np.arange(i, min(i + block, n_sections)) for i in range(0, n_sections, block)]


def _block_sums(counts, t_index, block):
    blocks = _blocks(counts.n_sections, block)
    cols = (counts.n_pred[:, t_index], counts.tp_pred[:, t_index],
            counts.tp_label[:, t_index], counts.n_label)
    return np.array([[c[b].sum() for c in cols] for b in blocks], dtype=np.float64)  # (B, 4)


def _f1_from_sums(s):
    """P, R, F1 from summed [n_pred, tp_pred, tp_label, n_label] (last axis), vectorized."""
    n_pred, tp_pred, tp_label, n_label = (s[..., i] for i in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_pred > 0, tp_pred / n_pred, np.nan)
        r = np.where(n_label > 0, tp_label / n_label, np.nan)
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    f1 = np.where(np.isnan(p) | np.isnan(r), np.nan, f1)
    return p, r, f1


def block_bootstrap(counts, t_index, block=10, n_boot=2000, seed=0, ci=0.95):
    """Percentile CI for P, R, F1 at one threshold, resampling blocks of sections."""
    sums = _block_sums(counts, t_index, block)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[draw].sum(axis=1)                   # (n_boot, 4)
    p, r, f1 = _f1_from_sums(boot)
    p0, r0, f0 = _f1_from_sums(sums.sum(axis=0))
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    pct = lambda x: (float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi)))
    return dict(threshold=float(counts.thresholds[t_index]), block=block, n_blocks=len(sums),
                precision=float(p0), recall=float(r0), f1=float(f0),
                precision_ci=pct(p), recall_ci=pct(r), f1_ci=pct(f1), f1_samples=f1)


def paired_bootstrap(counts_a, ia, counts_b, ib, block=10, n_boot=2000, seed=0, ci=0.95):
    """CI for F1(A) - F1(B), drawing the SAME blocks for both models in each replicate.
    Pairing cancels the section-to-section difficulty both models share, so the
    interval on the difference is much tighter than comparing two separate CIs."""
    if counts_a.n_sections != counts_b.n_sections:
        raise ValueError("both models must be scored on the same sections")
    sa, sb = _block_sums(counts_a, ia, block), _block_sums(counts_b, ib, block)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(sa), size=(n_boot, len(sa)))
    fa = _f1_from_sums(sa[draw].sum(axis=1))[2]
    fb = _f1_from_sums(sb[draw].sum(axis=1))[2]
    d = fa - fb
    d0 = _f1_from_sums(sa.sum(axis=0))[2] - _f1_from_sums(sb.sum(axis=0))[2]
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return dict(diff=float(d0), diff_ci=(float(np.percentile(d, lo)), float(np.percentile(d, hi))),
                frac_le_0=float(np.mean(d <= 0)), block=block, n_blocks=len(sa))


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Score a fault probability volume against labels.")
    ap.add_argument("--prob", required=True, help=".npy, shape (sections, rows, cols), values 0-1")
    ap.add_argument("--label", required=True, help=".npy, same shape, nonzero = fault")
    ap.add_argument("--tol", type=float, default=2.0, help="tolerance in samples")
    ap.add_argument("--region-radius", type=float, default=0,
                    help="if > 0, count predictions only within this many samples of a label")
    ap.add_argument("--block", type=int, default=10, help="sections per bootstrap block")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    prob, label = np.load(args.prob), np.load(args.label) != 0
    region = label_region(label, args.region_radius) if args.region_radius > 0 else None
    c = curve_counts(prob, label, args.tol, region=region)
    s = summarize(c)
    b = block_bootstrap(c, s["best_index"], block=args.block, n_boot=args.n_boot)
    d0, i0 = dice_iou(prob >= s["best_threshold"], label, 0)
    dt, it = dice_iou(prob >= s["best_threshold"], label, args.tol)

    print(f"sections {c.n_sections}  tol {args.tol}  region radius {args.region_radius or 'none'}")
    print(f"AP (PR-AUC)            {s['ap']:.3f}")
    print(f"best-F1 threshold      {s['best_threshold']:.2f}  (chosen on the same data: optimistic)")
    for k in ("precision", "recall", "f1"):
        lo, hi = b[f"{k}_ci"]
        print(f"{k:10s} at best      {b[k]:.3f}   95% CI [{lo:.3f}, {hi:.3f}]  ({b['n_blocks']} blocks of {args.block})")
    print(f"Dice / IoU (pixel)     {d0:.3f} / {i0:.3f}")
    print(f"Dice / IoU (dilated {args.tol:g}) {dt:.3f} / {it:.3f}")


if __name__ == "__main__":
    main()
