"""Known-answer tests for monitoring/drift_signals.py and sql/drift/.

Spark part (small synthetic lake, perturbed with spark/perturb.py):
  reference vs itself          PSI exactly 0 (raw and normalized)
  gain x4                      raw PSI large (every trace lands in the top decile);
                               normalized PSI ~0, because RMS / median is unchanged
  noise                        spectral centroid rises (white noise adds high frequencies)
  low-pass 20 Hz               spectral centroid falls
NumPy part (output signals on hand-made probability volumes):
  p = 0.5 everywhere           mean entropy = ln 2
  one fault vs the same fault cut into 10 pieces    10x the fragmentation
Analysis part: f1_drop and the reference's deltas are 0; table and figure are written.
(Converted from tests/check_drift.py; same cases, same expected values.)
"""
import numpy as np
import pandas as pd
import pytest

import drift_signals as ds
from helpers import CROP_BOUNDS


@pytest.fixture(scope="module")
def inp(drift_lake):
    return ds.input_signals(drift_lake, "S", CROP_BOUNDS, master="local[2]").set_index("survey")


@pytest.mark.spark
def test_reference_vs_itself_psi_zero(inp):
    assert abs(inp.loc["S", "psi_rms_raw"]) < 1e-12
    assert abs(inp.loc["S", "psi_rms_norm"]) < 1e-12


@pytest.mark.spark
def test_gain_moves_raw_psi_not_normalized(inp):
    assert inp.loc["S-gain-4", "psi_rms_raw"] > 1
    assert inp.loc["S-gain-4", "psi_rms_norm"] < 0.01
    assert abs(inp.loc["S-gain-4", "d_centroid_hz"]) < 1e-6


@pytest.mark.spark
def test_noise_raises_centroid(inp):
    assert inp.loc["S-noise-0.5", "d_centroid_hz"] > 5


@pytest.mark.spark
def test_lowpass_lowers_centroid(inp):
    assert inp.loc["S-lowpass-20", "d_centroid_hz"] < -5


def one_fault():
    whole = np.zeros((8, 40, 200), np.float32)
    whole[:, 20, :] = 0.9                                   # one continuous fault per section
    broken = whole.copy()
    broken[:, 20, 19::20] = 0.0                             # cut into 10 pieces
    return whole, broken


def test_entropy_of_half_is_ln2():
    o = ds.output_signals(np.full((8, 40, 60), 0.5, np.float32), step=1)
    assert o["mean_entropy"] == pytest.approx(np.log(2), abs=1e-5)


def test_fragmentation():
    whole, broken = one_fault()
    fw, fb = ds.output_signals(whole, step=1), ds.output_signals(broken, step=1)
    assert fw["n_components"] == 8                         # 1 piece per section
    assert fb["n_components"] == 80                        # 10 pieces per section
    assert 9 < fb["frag_per_kpx"] / fw["frag_per_kpx"] < 11.5


@pytest.mark.spark
def test_analysis_end_to_end(inp, tmp_path):
    """Fake study folder: per-survey predictions plus a summary.csv of F1 values."""
    whole, broken = one_fault()
    rng = np.random.default_rng(0)
    f1 = {"S": 0.60, "S-gain-4": 0.60, "S-noise-0.5": 0.40, "S-lowpass-20": 0.50}
    for s in f1:
        (tmp_path / s / "pred").mkdir(parents=True)
        p = broken if s == "S-noise-0.5" else whole
        np.save(tmp_path / s / "pred" / "prob_crop.npy", np.clip(p + 0.01 * rng.random(p.shape), 0, 1))
    pd.DataFrame([dict(survey=s, kind=("none" if s == "S" else s.split("-")[1]),
                       level=(0 if s == "S" else s.split("-")[2]), f1=v) for s, v in f1.items()]
                 ).to_csv(tmp_path / "summary.csv", index=False)
    df = ds.analyze(tmp_path, inp.reset_index(), "S")
    ref = df[df.survey == "S"].iloc[0]
    assert ref.f1_drop == 0 and ref.d_frag == 0 and ref.d_mean_prob == 0
    assert df.set_index("survey").loc["S-noise-0.5", "f1_drop"] == pytest.approx(0.2, abs=1e-9)
    ds.report(df, tmp_path)
    assert (tmp_path / "drift_table.csv").exists()
    assert (tmp_path / "drift_vs_f1.png").exists()
