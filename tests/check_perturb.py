#!/usr/bin/env python3
"""Known-answer tests for spark/perturb.py, on a small synthetic lake built in a temp folder.

  gain 4       output == 4 x input exactly (float32 rounding)
  tgain p      output / input == (t / t_mid)^p sample by sample
  noise L      std(output - input) == L x crop RMS (within 2%); rerun gives identical noise
  lowpass fc   energy above 2 x fc removed (< 0.1% of the input's energy there, Hann-tapered FFT)
  all          same traces (il, xl) as the crop; untouched survey still present
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spark"))
from perturb import perturb_block  # noqa: E402

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label:55s} {detail}")


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit("command failed: " + " ".join(map(str, args[:2])))
    return r.stdout


def read(root, survey):
    df = pd.read_parquet(f"{root}/traces/survey={survey}").sort_values(["il", "xl"]).reset_index(drop=True)
    return df, np.stack(df["samples"].to_numpy()).astype(np.float64)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    run(REPO / "tests/make_synthetic_segy.py", tmp / "s.sgy", "--il", "100,139", "--xl", "300,349",
        "--ns", "200", "--dead", "0", "--missing", "0.02")
    run(REPO / "spark/segy_to_parquet.py", tmp / "s.sgy", "--survey", "S", "--out", tmp / "lake",
        "--master", "local[2]")
    crop = ["--il-range", "105", "134", "--xl-range", "310", "339"]
    base = [REPO / "spark/perturb.py", "--root", tmp / "lake", "--survey", "S", "--master", "local[2]", *crop]
    print(run(*base, "--kind", "gain", "--levels", "4").strip().splitlines()[0])
    run(*base, "--kind", "tgain", "--levels", "1.5")
    run(*base, "--kind", "noise", "--levels", "0.5")
    run(*base, "--kind", "lowpass", "--levels", "20")

    src, x = read(tmp / "lake", "S")
    inside = src.il.between(105, 134) & src.xl.between(310, 339)
    src, x = src[inside].reset_index(drop=True), x[inside.to_numpy()]
    rms = np.sqrt((x ** 2).mean())
    dt = 0.004
    ns = x.shape[1]

    g, y = read(tmp / "lake", "S-gain-4")
    check("same (il, xl) positions as the crop", (g.il.to_numpy() == src.il.to_numpy()).all()
          and (g.xl.to_numpy() == src.xl.to_numpy()).all(), f"{len(g)} traces")
    check("gain 4: output == 4 x input", np.allclose(y, 4 * x, rtol=1e-6, atol=1e-3),
          f"max rel err {np.max(np.abs(y - 4 * x)) / np.abs(4 * x).max():.1e}")

    _, y = read(tmp / "lake", "S-tgain-1.5")
    t = (np.arange(ns) + 0.5) * dt
    gt = (t / (ns / 2 * dt)) ** 1.5
    check("tgain 1.5: output == input x (t/t_mid)^1.5", np.allclose(y, x * gt, rtol=1e-5, atol=1e-3))

    _, y = read(tmp / "lake", "S-noise-0.5")
    ratio = (y - x).std() / rms
    check("noise 0.5: std(added) / crop RMS = 0.5 (within 2%)", abs(ratio - 0.5) < 0.01, f"{ratio:.4f}")
    run(*base, "--kind", "noise", "--levels", "0.5")                      # rerun
    _, y2 = read(tmp / "lake", "S-noise-0.5")
    check("noise: rerun gives identical noise", np.array_equal(y, y2))

    _, y = read(tmp / "lake", "S-lowpass-20")
    f = np.fft.rfftfreq(ns, dt)
    # Hann taper before the FFT: the FFT treats each trace as periodic, so the jump
    # between its last and first sample leaks energy into every frequency and would
    # hide what the filter did (0.6% "left" untapered vs ~1e-7 tapered).
    w = np.hanning(ns)
    e_in = (np.abs(np.fft.rfft(x * w, axis=1)) ** 2)[:, f > 40].sum()
    e_out = (np.abs(np.fft.rfft(y * w, axis=1)) ** 2)[:, f > 40].sum()
    check("lowpass 20 Hz: energy above 40 Hz < 0.1% of input's", e_out / e_in < 1e-3, f"{e_out / e_in:.2e}")
    e_lo_in = (np.abs(np.fft.rfft(x * w, axis=1)) ** 2)[:, f < 10].sum()
    e_lo_out = (np.abs(np.fft.rfft(y * w, axis=1)) ** 2)[:, f < 10].sum()
    check("lowpass 20 Hz: energy below 10 Hz kept (> 95%)", e_lo_out / e_lo_in > 0.95, f"{e_lo_out / e_lo_in:.3f}")

    p = pd.read_parquet(tmp / "lake" / "perturbations")
    check("perturbations table: 4 surveys recorded", len(p) == 4, ", ".join(sorted(p.survey.astype(str))))
    s = pd.read_parquet(tmp / "lake" / "surveys")
    check("surveys table: parent + 4 perturbed", len(s) == 5)

    # perturb_block on its own: noise depends only on (seed, il, xl), not on batch order
    blk = np.ones((3, 50))
    a = perturb_block(blk, [1, 2, 3], [7, 8, 9], "noise", 1.0, dt, 0, 1.0, 0)
    b_ = perturb_block(blk[::-1], [3, 2, 1], [9, 8, 7], "noise", 1.0, dt, 0, 1.0, 0)[::-1]
    check("noise is per-trace, independent of batch order", np.array_equal(a, b_))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
