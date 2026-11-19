"""Known-answer tests for evaluation/fault_metrics.py.

Synthetic sections hold vertical faults (one pixel wide), so distances are exact
whole numbers and every expected value can be worked out by hand.
(Converted from tests/check_metrics.py; same cases, same expected values.)
"""
import numpy as np
import pytest
from scipy import ndimage as ndi

import fault_metrics as fm
from helpers import at as _at

S, H, W = 40, 128, 128
R0, R1 = 20, 108                 # fault rows [R0, R1) -> length L
L = R1 - R0
EXACT = 1e-9


def approx(x, tol=EXACT):
    return pytest.approx(x, abs=tol)


def faults(cols, shape=(S, H, W), rows=(R0, R1)):
    m = np.zeros(shape, bool)
    for c in cols:
        m[:, rows[0]:rows[1], c] = True
    return m


def at(c, thr):
    return _at(c, thr, fm.prf)


@pytest.fixture(scope="module")
def label():
    return faults([30, 90])


# 1. perfect prediction
def test_perfect_prediction(label):
    p, r, _ = at(fm.curve_counts(label.astype(float), label, tol=0), 0.5)
    assert p == approx(1.0)
    assert r == approx(1.0)
    assert fm.dice_iou(label, label)[0] == approx(1.0)


# 2. shifted 2 samples sideways: zero below tol 2, perfect at tol 2
@pytest.mark.parametrize("tol, expected", [(0, 0.0), (1, 0.0), (1.9, 0.0), (2, 1.0), (3, 1.0)])
def test_shift_2_f1_vs_tolerance(label, tol, expected):
    _, _, f = at(fm.curve_counts(faults([32, 92]).astype(float), label, tol=tol), 0.5)
    assert f == approx(expected)


def test_shift_2_pixel_dice(label):
    pred = faults([32, 92])
    assert fm.dice_iou(pred, label, 0)[0] == approx(0.0)
    # Dilating a 1-px line by a radius-1 disk -> 3 columns plus 1 pixel at each end (3L + 2).
    # The two dilated lines overlap in one column of L pixels.
    assert fm.dice_iou(pred, label, 1)[0] == approx(2 * L / (6 * L + 4))


# 3. only one of the two faults found
def test_half_found(label):
    p, r, f = at(fm.curve_counts(faults([30]).astype(float), label, tol=2), 0.5)
    assert (p, r, f) == (approx(1.0), approx(0.5), approx(2 / 3))


# 4. an extra false fault far from both labels
def test_false_fault(label):
    p, r, _ = at(fm.curve_counts(faults([30, 60, 90]).astype(float), label, tol=2), 0.5)
    assert p == approx(2 / 3)
    assert r == approx(1.0)


# 5. probabilities: true faults 0.75, false fault 0.25, background 0
def test_probabilistic(label):
    prob = np.zeros((S, H, W))
    prob[label] = 0.75
    prob[faults([60])] = 0.25
    c = fm.curve_counts(prob, label, tol=2)
    s = fm.summarize(c)
    assert s["best_f1"] == approx(1.0)
    assert 0.25 < s["best_threshold"] <= 0.75
    assert s["ap"] == approx(1.0)
    assert at(c, 0.2)[0] == approx(2 / 3)      # t=0.2 includes the false fault


# 6. incomplete labels: interpreter labeled only the fault at 30, the model finds both
def test_incomplete_labels_region():
    partial = faults([30])
    pred = faults([30, 90]).astype(float)
    c_all = fm.curve_counts(pred, partial, tol=2)
    c_reg = fm.curve_counts(pred, partial, tol=2, region=fm.label_region(partial, 10))
    assert at(c_all, 0.5)[0] == approx(0.5)
    assert at(c_reg, 0.5)[0] == approx(1.0)
    assert at(c_reg, 0.5)[1] == approx(at(c_all, 0.5)[1])     # recall unchanged by region


# 7. histogram counts == direct counts, on a random smooth probability field
def test_histogram_sweep_matches_direct_counts():
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
    for t in [0.1, 0.33, 0.5, 0.62, 0.9]:
        i = int(np.argmin(np.abs(c.thresholds - t)))
        pb = prob >= c.thresholds[i]
        direct = [pb.sum(), (pb & (fm.distance_to(lab) <= tol)).sum(),
                  (lab & (fm.distance_to(pb) <= tol)).sum()]
        hist = [c.n_pred[:, i].sum(), c.tp_pred[:, i].sum(), c.tp_label[:, i].sum()]
        assert direct == hist, f"threshold {c.thresholds[i]:.3f}"


# 8. bootstrap behavior
def test_bootstrap_identical_sections_zero_width(label):
    c = fm.curve_counts(faults([30]).astype(float), label, tol=2)
    b = fm.block_bootstrap(c, 50, block=5, n_boot=500)
    assert b["f1_ci"][1] - b["f1_ci"][0] == approx(0.0)


def test_paired_bootstrap_excludes_zero(label):
    """A finds both faults; B misses the second fault on a random half of sections."""
    # Own seed: check_metrics.py drew these from an rng shared with test 7, which made
    # the result depend on test order. Any random half of the sections shows the effect.
    miss = np.random.default_rng(8).random(S) < 0.5
    assert 10 < miss.sum() < 30
    pb = faults([30, 90])
    pb[miss, :, 90] = False
    ca = fm.curve_counts(label.astype(float), label, tol=2)
    cb = fm.curve_counts(pb.astype(float), label, tol=2)
    d = fm.paired_bootstrap(ca, 50, cb, 50, block=5, n_boot=2000)
    assert d["diff_ci"][0] > 0


def test_block_bootstrap_wider_on_autocorrelated_sections():
    """Recall drifts slowly from section to section, as on real inlines, so a
    block bootstrap gives a wider interval than resampling single sections."""
    S2 = 200
    frac = 0.55 + 0.35 * np.sin(2 * np.pi * np.arange(S2) / 60)
    lab2 = faults([20], shape=(S2, 64, 64), rows=(8, 56))
    pr2 = np.zeros_like(lab2)
    for s_ in range(S2):
        pr2[s_, 8:8 + int(round(frac[s_] * 48)), 20] = True
    c2 = fm.curve_counts(pr2.astype(float), lab2, tol=1)
    w1 = np.subtract(*fm.block_bootstrap(c2, 50, block=1, n_boot=3000)["recall_ci"][::-1])
    w20 = np.subtract(*fm.block_bootstrap(c2, 50, block=20, n_boot=3000)["recall_ci"][::-1])
    assert w20 > 1.3 * w1, f"block 1 width {w1:.4f}, block 20 width {w20:.4f}"


# 9. thinning (non-maximum suppression)
PROF = np.exp(-0.5 * ((np.arange(-4, 5)) / 2.0) ** 2)


def test_thin_vertical_band_keeps_center_column():
    band = np.zeros((2, H, W))
    band[:, R0:R1, 56:65] = PROF * 0.9
    inner = fm.thin_faults(band)[:, R0 + 3:R1 - 3]            # away from the band's ends
    assert np.unique(np.nonzero(inner > 0.5)[2]).tolist() == [60]
    assert inner[inner > 0.5].max() == approx(0.9, 1e-6)     # kept pixels keep their probability


def test_thin_flat_topped_band_one_pixel_per_row():
    """Ties must leave exactly one pixel per row."""
    flat = np.zeros((2, H, W))
    flat[:, R0:R1, 50:56] = 0.9
    per_row = (fm.thin_faults(flat)[:, R0 + 3:R1 - 3] > 0.5).sum(axis=2)
    assert np.all(per_row == 1)


def test_thin_45_degree_band():
    yy, xx = np.mgrid[0:H, 0:W]
    dist45 = np.abs((xx - 20) - (yy - 10)) / np.sqrt(2)
    diag = np.where((dist45 <= 3) & (yy > 15) & (yy < 110), np.exp(-0.5 * (dist45 / 1.5) ** 2), 0.0)
    per_row = (fm.thin_faults(np.stack([diag, diag]))[:, 25:100] > 0.3).sum(axis=2)
    assert per_row.max() <= 2
    assert per_row.min() >= 1                                 # no row emptied


def test_thinning_precision_gain():
    """Label = one column, prediction = a 9-wide band centered on it. At t=0.1 all 9
    band columns are predicted; at tol 2 only 5 are within 2 of the label -> precision
    5/9. Thinned -> 1.0. Recall drops slightly: at each END of the band NMS compares
    up/down and drops the last couple of pixels (Canny NMS shortens line ends too)."""
    lab9 = faults([60])
    band9 = np.zeros((S, H, W))
    band9[:, R0:R1, 56:65] = PROF * 0.9
    p_raw = at(fm.curve_counts(band9, lab9, tol=2), 0.1)
    p_thin = at(fm.curve_counts(fm.thin_faults(band9), lab9, tol=2), 0.1)
    assert p_raw[0] == approx(5 / 9)
    assert p_thin[0] == approx(1.0)
    assert p_thin[1] >= 0.9
