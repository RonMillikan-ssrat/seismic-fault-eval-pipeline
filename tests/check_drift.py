#!/usr/bin/env python3
"""Known-answer tests for monitoring/drift_signals.py and sql/drift/.

Spark part (small synthetic lake in a temp folder, perturbed with spark/perturb.py):
  reference vs itself          PSI exactly 0 (raw and normalized)
  gain x4                      raw PSI large (every trace lands in the top decile);
                               normalized PSI ~0, because RMS / median is unchanged
  noise                        spectral centroid rises (white noise adds high frequencies)
  low-pass 20 Hz               spectral centroid falls
NumPy part (output signals on hand-made probability volumes):
  p = 0.5 everywhere           mean entropy = ln 2
  one fault vs the same fault cut into 10 pieces    10x the fragmentation
Analysis part: f1_drop and the reference's deltas are 0; table and figure are written.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "monitoring"))
import drift_signals as ds  # noqa: E402

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label:58s} {detail}")


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit("command failed: " + " ".join(map(str, args[:2])))
    return r.stdout


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    lake = tmp / "lake"
    run(REPO / "tests/make_synthetic_segy.py", tmp / "s.sgy", "--il", "100,139", "--xl", "300,349",
        "--ns", "200", "--dead", "0", "--missing", "0")
    run(REPO / "spark/segy_to_parquet.py", tmp / "s.sgy", "--survey", "S", "--out", lake, "--master", "local[2]")
    crop = ["--il-range", "105", "134", "--xl-range", "310", "339"]
    for kind, level in [("gain", "4"), ("noise", "0.5"), ("lowpass", "20")]:
        run(REPO / "spark/perturb.py", "--root", lake, "--survey", "S", "--kind", kind, "--levels", level,
            "--master", "local[2]", *crop)
    for s in ["S", "S-gain-4", "S-noise-0.5", "S-lowpass-20"]:
        run(REPO / "spark/trace_attributes.py", "--root", lake, "--survey", s, "--master", "local[2]",
            "--variants", "pandas", "--repeat", "1")

    inp = ds.input_signals(lake, "S", ((105, 134), (310, 339)), master="local[2]").set_index("survey")
    print(inp[["psi_rms_raw", "psi_rms_norm", "centroid_med_hz", "d_centroid_hz"]].round(4).to_string())
    check("reference vs itself: raw PSI = 0", abs(inp.loc["S", "psi_rms_raw"]) < 1e-12)
    check("reference vs itself: normalized PSI = 0", abs(inp.loc["S", "psi_rms_norm"]) < 1e-12)
    check("gain x4: raw PSI > 1 (large shift)", inp.loc["S-gain-4", "psi_rms_raw"] > 1,
          f"{inp.loc['S-gain-4', 'psi_rms_raw']:.2f}")
    check("gain x4: normalized PSI < 0.01 (no shift)", inp.loc["S-gain-4", "psi_rms_norm"] < 0.01,
          f"{inp.loc['S-gain-4', 'psi_rms_norm']:.4f}")
    check("gain x4: spectral centroid unchanged", abs(inp.loc["S-gain-4", "d_centroid_hz"]) < 1e-6)
    check("noise: spectral centroid rises > 5 Hz", inp.loc["S-noise-0.5", "d_centroid_hz"] > 5,
          f"{inp.loc['S-noise-0.5', 'd_centroid_hz']:+.1f} Hz")
    check("low-pass 20 Hz: spectral centroid falls > 5 Hz", inp.loc["S-lowpass-20", "d_centroid_hz"] < -5,
          f"{inp.loc['S-lowpass-20', 'd_centroid_hz']:+.1f} Hz")

    # NumPy output signals
    half = np.full((8, 40, 60), 0.5, np.float32)
    o = ds.output_signals(half, step=1)
    check("p = 0.5 everywhere: mean entropy = ln 2", abs(o["mean_entropy"] - np.log(2)) < 1e-5,
          f"{o['mean_entropy']:.5f}")
    whole = np.zeros((8, 40, 200), np.float32)
    whole[:, 20, :] = 0.9                                   # one continuous fault per section
    broken = whole.copy()
    broken[:, 20, 19::20] = 0.0                             # cut into 10 pieces
    fw, fb = ds.output_signals(whole, step=1), ds.output_signals(broken, step=1)
    check("fragmentation: 1 piece per section", fw["n_components"] == 8, f"{fw['n_components']}")
    check("fragmentation: 10 pieces per section", fb["n_components"] == 80, f"{fb['n_components']}")
    check("fragmentation per 1000 px rises ~10x",
          9 < fb["frag_per_kpx"] / fw["frag_per_kpx"] < 11.5, f"{fb['frag_per_kpx'] / fw['frag_per_kpx']:.2f}x")

    # Analysis end to end with a fake study folder
    study = tmp / "study"
    rng = np.random.default_rng(0)
    f1 = {"S": 0.60, "S-gain-4": 0.60, "S-noise-0.5": 0.40, "S-lowpass-20": 0.50}
    for s in f1:
        (study / s / "pred").mkdir(parents=True)
        p = whole.copy() if s != "S-noise-0.5" else broken.copy()
        np.save(study / s / "pred" / "prob_crop.npy", np.clip(p + 0.01 * rng.random(p.shape), 0, 1))
    pd.DataFrame([dict(survey=s, kind=("none" if s == "S" else s.split("-")[1]),
                       level=(0 if s == "S" else s.split("-")[2]), f1=v) for s, v in f1.items()]
                 ).to_csv(study / "summary.csv", index=False)
    df = ds.analyze(study, inp.reset_index(), "S")
    ref = df[df.survey == "S"].iloc[0]
    check("analysis: reference F1 drop and output deltas = 0",
          ref.f1_drop == 0 and ref.d_frag == 0 and ref.d_mean_prob == 0)
    check("analysis: noise F1 drop = 0.20", abs(df.set_index("survey").loc["S-noise-0.5", "f1_drop"] - 0.2) < 1e-9)
    ds.report(df, study)
    check("analysis: drift_table.csv and drift_vs_f1.png written",
          (study / "drift_table.csv").exists() and (study / "drift_vs_f1.png").exists())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
