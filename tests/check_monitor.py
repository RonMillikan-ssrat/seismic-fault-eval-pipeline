#!/usr/bin/env python3
"""Known-answer tests for monitoring/monitor_batch.py.

Calibration (plain pandas): a hand-made drift table where the right thresholds
are obvious, including a signal that cannot separate harm.
Check (small synthetic lake + hand-made probability volumes):
  reference vs itself      OK
  gain x4                  OK   (raw PSI is huge but is context only)
  noise + 40% fewer faults ALERT
  low-pass + 60% more      WARN (rise rule; no alert threshold for rises)
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
import monitor_batch as mb  # noqa: E402

results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label:60s} {detail}")


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit("command failed: " + " ".join(map(str, args[:3])))
    return r.stdout


# ---- calibration on a hand-made table
# predicted fraction: falls of 10/20/40/60 % lose 0.02/0.08/0.30/0.50 F1; rises of 20/50 % lose 0.00/0.06.
# centroid: rises 0/0/5/10 Hz with the falls' drops... psi_raw: 8 for an unharmed AND a harmed survey.
tab = pd.DataFrame([
    dict(survey="R",  f1_fixed_drop=0.00, frac_pos=0.100, d_centroid_hz=0.0,   psi_rms_norm=0.0, psi_rms_raw=0.0),
    dict(survey="g",  f1_fixed_drop=0.00, frac_pos=0.100, d_centroid_hz=0.0,   psi_rms_norm=0.0, psi_rms_raw=8.0),
    dict(survey="n1", f1_fixed_drop=0.02, frac_pos=0.090, d_centroid_hz=1.0,   psi_rms_norm=0.0, psi_rms_raw=0.5),
    dict(survey="n2", f1_fixed_drop=0.08, frac_pos=0.080, d_centroid_hz=4.0,   psi_rms_norm=0.0, psi_rms_raw=3.0),
    dict(survey="n3", f1_fixed_drop=0.30, frac_pos=0.060, d_centroid_hz=8.0,   psi_rms_norm=0.1, psi_rms_raw=8.0),
    dict(survey="n4", f1_fixed_drop=0.50, frac_pos=0.040, d_centroid_hz=12.0,  psi_rms_norm=0.2, psi_rms_raw=8.0),
    dict(survey="l1", f1_fixed_drop=0.00, frac_pos=0.120, d_centroid_hz=-15.0, psi_rms_norm=0.3, psi_rms_raw=8.0),
    dict(survey="l2", f1_fixed_drop=0.06, frac_pos=0.150, d_centroid_hz=-20.0, psi_rms_norm=0.6, psi_rms_raw=8.0),
])
t = mb.calibrate(tab, "R", warn_drop=0.05, alert_drop=0.15)
rule = {(r["signal"], r["direction"]): r for r in t["rules"]}
ctx = {(c["signal"], c["direction"]) for c in t["context_only"]}
fall = rule[("frac_pos_pct", "fall")]
check("fraction fall: warn = midpoint of 10% and 20% = 15", abs(fall["warn"] - 15) < 1e-6, f"{fall['warn']}")
check("fraction fall: alert = midpoint of 20% and 40% = 30", abs(fall["alert"] - 30) < 1e-6, f"{fall['alert']}")
rise = rule[("frac_pos_pct", "rise")]
check("fraction rise: warn = midpoint of 20% and 50% = 35", abs(rise["warn"] - 35) < 1e-6, f"{rise['warn']}")
check("fraction rise: no alert threshold (no rise lost 0.15)", rise["alert"] is None)
cr = rule[("d_centroid_hz", "rise")]
check("centroid rise: warn 2.5 Hz, alert 6 Hz", abs(cr["warn"] - 2.5) < 1e-6 and abs(cr["alert"] - 6) < 1e-6,
      f"{cr['warn']}, {cr['alert']}")
check("raw PSI is context only (8.0 for unharmed and harmed)", ("psi_rms_raw", "rise") in ctx)
check("normalized PSI is context only (0 for a harmed survey)", ("psi_rms_norm", "rise") in ctx)

rows, status = mb.evaluate({"frac_pos_pct": -18.0, "d_centroid_hz": 1.0, "psi_rms_raw": 8.0}, t)
check("evaluate: -18% fraction -> WARN; raw PSI 8 does not alert", status == "WARN", status)
rows, status = mb.evaluate({"frac_pos_pct": -45.0, "d_centroid_hz": 9.0}, t)
check("evaluate: -45% fraction, +9 Hz -> ALERT", status == "ALERT", status)
rows, status = mb.evaluate({"frac_pos_pct": 90.0, "d_centroid_hz": -30.0}, t)
check("evaluate: +90% fraction -> WARN only (no alert rule for rises)", status == "WARN", status)

# spot check
vol = np.full((20, 8, 8), 0.01, np.float32)
vol[[3, 11]] = 0.5                                       # two maximally uncertain inlines
top, rnd = mb.spot_check(vol, 2, seed=1)
check("spot check: highest-entropy inlines are 3 and 11", top == [3, 11], f"{top}")
check("spot check: random picks exclude them, 2 of them", len(rnd) == 2 and not set(rnd) & {3, 11}, f"{rnd}")

# ---- check on a synthetic lake
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

    base = np.zeros((30, 30, 200), np.float32)
    base[:, 10, 20:120] = 0.9                               # 100 fault voxels per inline
    fewer, more = base.copy(), base.copy()
    fewer[:, 10, 80:120] = 0.0                              # 40% fewer
    more[:, 20, 20:80] = 0.9                                # 60% more
    preds = {"S": base, "S-gain-4": base, "S-noise-0.5": fewer, "S-lowpass-20": more}
    for s, p in preds.items():
        (tmp / "pred" / s).mkdir(parents=True)
        np.save(tmp / "pred" / s / "prob_crop.npy", p)
    thr = tmp / "thresholds.json"
    thr.write_text(json.dumps(t))

    expect = {"S": "OK", "S-gain-4": "OK", "S-noise-0.5": "ALERT", "S-lowpass-20": "WARN"}
    for s, want in expect.items():
        out = tmp / "rep" / s
        log = run(REPO / "monitoring/monitor_batch.py", "check", "--root", lake, "--reference", "S",
                  "--survey", s, "--pred-dir", tmp / "pred" / s, "--ref-pred-dir", tmp / "pred" / "S",
                  "--thresholds", thr, "--out", out, "--master", "local[2]", *crop)
        rep = json.loads((out / "monitor_report.json").read_text())
        v = rep["values"]
        check(f"check {s}: status {want}", rep["status"] == want,
              f"{rep['status']}  frac {v['frac_pos_pct']:+.0f}%, centroid {v['d_centroid_hz']:+.1f} Hz, "
              f"raw PSI {v['psi_rms_raw']:.2f}")
    print(log.strip())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
