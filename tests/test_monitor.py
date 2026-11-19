"""Known-answer tests for monitoring/monitor_batch.py.

Calibration (plain pandas): a hand-made drift table where the right thresholds
are obvious, including a signal that cannot separate harm.
Check (small synthetic lake + hand-made probability volumes):
  reference vs itself      OK
  gain x4                  OK   (raw PSI is huge but is context only)
  noise + 40% fewer faults ALERT
  low-pass + 60% more      WARN (rise rule; no alert threshold for rises)
(Converted from tests/check_monitor.py; same cases, same expected values.)
"""
import json

import numpy as np
import pandas as pd
import pytest

import monitor_batch as mb
from helpers import CROP, MASTER, REPO, run

# predicted fraction: falls of 10/20/40/60 % lose 0.02/0.08/0.30/0.50 F1; rises of 20/50 % lose 0.00/0.06.
# raw PSI: 8 for an unharmed AND a harmed survey, so it cannot separate harm.
TABLE = pd.DataFrame([
    dict(survey="R",  f1_fixed_drop=0.00, frac_pos=0.100, d_centroid_hz=0.0,   psi_rms_norm=0.0, psi_rms_raw=0.0),
    dict(survey="g",  f1_fixed_drop=0.00, frac_pos=0.100, d_centroid_hz=0.0,   psi_rms_norm=0.0, psi_rms_raw=8.0),
    dict(survey="n1", f1_fixed_drop=0.02, frac_pos=0.090, d_centroid_hz=1.0,   psi_rms_norm=0.0, psi_rms_raw=0.5),
    dict(survey="n2", f1_fixed_drop=0.08, frac_pos=0.080, d_centroid_hz=4.0,   psi_rms_norm=0.0, psi_rms_raw=3.0),
    dict(survey="n3", f1_fixed_drop=0.30, frac_pos=0.060, d_centroid_hz=8.0,   psi_rms_norm=0.1, psi_rms_raw=8.0),
    dict(survey="n4", f1_fixed_drop=0.50, frac_pos=0.040, d_centroid_hz=12.0,  psi_rms_norm=0.2, psi_rms_raw=8.0),
    dict(survey="l1", f1_fixed_drop=0.00, frac_pos=0.120, d_centroid_hz=-15.0, psi_rms_norm=0.3, psi_rms_raw=8.0),
    dict(survey="l2", f1_fixed_drop=0.06, frac_pos=0.150, d_centroid_hz=-20.0, psi_rms_norm=0.6, psi_rms_raw=8.0),
])


@pytest.fixture(scope="module")
def thresholds():
    return mb.calibrate(TABLE, "R", warn_drop=0.05, alert_drop=0.15)


@pytest.fixture(scope="module")
def rule(thresholds):
    return {(r["signal"], r["direction"]): r for r in thresholds["rules"]}


def test_fraction_fall_thresholds(rule):
    r = rule[("frac_pos_pct", "fall")]
    assert r["warn"] == pytest.approx(15)          # midpoint of 10% and 20%
    assert r["alert"] == pytest.approx(30)         # midpoint of 20% and 40%


def test_fraction_rise_thresholds(rule):
    r = rule[("frac_pos_pct", "rise")]
    assert r["warn"] == pytest.approx(35)          # midpoint of 20% and 50%
    assert r["alert"] is None                      # no rise lost 0.15


def test_centroid_rise_thresholds(rule):
    r = rule[("d_centroid_hz", "rise")]
    assert (r["warn"], r["alert"]) == (pytest.approx(2.5), pytest.approx(6))


@pytest.mark.parametrize("signal", ["psi_rms_raw", "psi_rms_norm"])
def test_psi_is_context_only(thresholds, signal):
    """raw PSI is 8.0 for unharmed and harmed surveys; normalized PSI is 0 for a harmed one."""
    assert (signal, "rise") in {(c["signal"], c["direction"]) for c in thresholds["context_only"]}


@pytest.mark.parametrize("values, status", [
    ({"frac_pos_pct": -18.0, "d_centroid_hz": 1.0, "psi_rms_raw": 8.0}, "WARN"),  # raw PSI does not alert
    ({"frac_pos_pct": -45.0, "d_centroid_hz": 9.0}, "ALERT"),
    ({"frac_pos_pct": 90.0, "d_centroid_hz": -30.0}, "WARN"),                      # no alert rule for rises
], ids=["fall-18pct", "fall-45pct", "rise-90pct"])
def test_evaluate(thresholds, values, status):
    assert mb.evaluate(values, thresholds)[1] == status


def test_spot_check_picks_uncertain_inlines():
    vol = np.full((20, 8, 8), 0.01, np.float32)
    vol[[3, 11]] = 0.5                                       # two maximally uncertain inlines
    top, rnd = mb.spot_check(vol, 2, seed=1)
    assert top == [3, 11]
    assert len(rnd) == 2 and not set(rnd) & {3, 11}


@pytest.fixture(scope="module")
def batch_setup(drift_lake, thresholds, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("monitor")
    base = np.zeros((30, 30, 200), np.float32)
    base[:, 10, 20:120] = 0.9                               # 100 fault voxels per inline
    fewer, more = base.copy(), base.copy()
    fewer[:, 10, 80:120] = 0.0                              # 40% fewer
    more[:, 20, 20:80] = 0.9                                # 60% more
    for s, p in {"S": base, "S-gain-4": base, "S-noise-0.5": fewer, "S-lowpass-20": more}.items():
        (tmp / "pred" / s).mkdir(parents=True)
        np.save(tmp / "pred" / s / "prob_crop.npy", p)
    thr = tmp / "thresholds.json"
    thr.write_text(json.dumps(thresholds))
    return drift_lake, tmp, thr


@pytest.mark.spark
@pytest.mark.parametrize("survey, want", [
    ("S", "OK"), ("S-gain-4", "OK"), ("S-noise-0.5", "ALERT"), ("S-lowpass-20", "WARN")])
def test_check_batch_status(batch_setup, survey, want):
    lake, tmp, thr = batch_setup
    out = tmp / "rep" / survey
    run(REPO / "monitoring/monitor_batch.py", "check", "--root", lake, "--reference", "S",
        "--survey", survey, "--pred-dir", tmp / "pred" / survey, "--ref-pred-dir", tmp / "pred" / "S",
        "--thresholds", thr, "--out", out, "--master", MASTER, *CROP)
    rep = json.loads((out / "monitor_report.json").read_text())
    assert rep["status"] == want, rep["values"]
