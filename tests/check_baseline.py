#!/usr/bin/env python3
"""Known-answer tests for baseline/coherence.py and evaluation/compare_models.py.

Semblance has exact answers on simple synthetic cubes:
  * identical traces                         -> semblance 1, attribute 0
  * reflectors flipped in polarity across a vertical fault plane, 3-trace window:
    a window holding 2 traces of one polarity and 1 of the other has
    mean = a/3, so semblance = (a/3)^2 / a^2 = 1/9 and attribute = 8/9 exactly,
    on the two crosslines next to the plane, and 0 everywhere else
  * independent noise traces                 -> semblance near 1/N (N = 9 traces)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "baseline"))
sys.path.insert(0, str(REPO / "evaluation"))
sys.path.insert(0, str(REPO / "tests"))
import coherence as coh  # noqa: E402
import make_synthetic_f3  # noqa: E402
import fault_metrics as fm  # noqa: E402

results = []


def check(label, expected, got, tol=1e-6):
    ok = abs(float(expected) - float(got)) <= tol
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label:58s} expected {float(expected):9.4f}  got {float(got):9.4f}")


n_il, n_xl, n_t = 12, 40, 120
t = np.arange(n_t)
wave = np.sin(2 * np.pi * t / 20.0).astype(np.float32)

# 1. identical traces
cube = np.broadcast_to(wave, (n_il, n_xl, n_t)).copy()
check("1 identical traces: max attribute", 0.0, coh.fault_attribute(cube).max())

# 2. polarity flip across a vertical fault plane between xl 19 and 20
c = 20
cube = np.broadcast_to(wave, (n_il, n_xl, n_t)).copy()
cube[:, c:, :] *= -1
attr = coh.fault_attribute(cube, n_il=3, n_xl=3, half_t=4)
check("2 fault plane: attribute at xl 19 = 8/9", 8 / 9, attr[:, c - 1, :].mean())
check("2 fault plane: attribute at xl 20 = 8/9", 8 / 9, attr[:, c, :].mean())
others = np.delete(attr, [c - 1, c], axis=1)
check("2 fault plane: max attribute elsewhere", 0.0, others.max())

# 2b scored as a fault map: label on xl 20, rows = time, so sections are (il, t, xl)
label = np.zeros((n_il, n_t, n_xl), bool)
label[:, :, c] = True
cc = fm.curve_counts(attr.transpose(0, 2, 1), label, tol=1)
i = int(np.argmin(np.abs(cc.thresholds - 0.5)))
p, r, f1 = fm.prf(*[x[i] if np.ndim(x) else x for x in cc.totals()])
check("2b scored at tol 1, t=0.5: F1", 1.0, f1)

# 3. independent noise: semblance near 1/9 for a 3x3 trace window
rng = np.random.default_rng(0)
noise = rng.standard_normal((n_il, n_xl, n_t)).astype(np.float32)
s = coh.semblance(noise, n_il=3, n_xl=3, half_t=4)
check("3 white noise: mean semblance ~ 1/9 (within 0.03)", 1 / 9, s.mean(), tol=0.03)

# 4. dip without any fault: attribute rises with dip (the no-dip-steering limitation)
means = []
for dip in [0.0, 0.5, 1.0, 2.0]:                    # samples of shift per crossline
    shifted = np.stack([np.sin(2 * np.pi * (t - dip * x) / 20.0) for x in range(n_xl)])
    cube = np.broadcast_to(shifted, (n_il, n_xl, n_t)).astype(np.float32)
    means.append(float(coh.fault_attribute(cube)[:, 2:-2, 5:-5].mean()))
print("      mean attribute vs dip (0, 0.5, 1, 2 samples/trace): " + ", ".join(f"{m:.3f}" for m in means))
check("4 dip: attribute increases with dip", 1.0, float(np.all(np.diff(means) > 0)))

# 5. lake loader: write a tiny traces table in the lake layout, with a missing
#    position and a duplicate whose later copy must be ignored
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    part = root / "traces" / "survey=T"
    part.mkdir(parents=True)
    il_v, xl_v, data = [], [], []
    for il in range(100, 106):
        for xl in range(300, 310):
            if (il, xl) == (102, 305):
                continue                                  # missing position
            il_v.append(il); xl_v.append(xl)
            data.append(np.full(16, il * 1000 + xl, np.float32))
    il_v.append(103); xl_v.append(304); data.append(np.full(16, -1.0, np.float32))  # duplicate
    seq = np.arange(len(il_v))
    for k, (a, b) in enumerate([(0, 30), (30, len(il_v))]):        # two files, like Spark output
        pq.write_table(pa.table({"trace_seq": pa.array(seq[a:b], pa.int64()),
                                 "il": pa.array(il_v[a:b], pa.int32()),
                                 "xl": pa.array(xl_v[a:b], pa.int32()),
                                 "samples": pa.array([d.tolist() for d in data[a:b]],
                                                     pa.list_(pa.float32()))}),
                       part / f"part-{k:05d}.parquet")
    cube, info = coh.load_cube(root, "T", (101, 105), (302, 308), (2, 10))
    check("5 loader: cube shape (4, 6, 8)", 1.0, float(cube.shape == (4, 6, 8)))
    check("5 loader: value at il 104, xl 303", 104303, cube[3, 1, 0])
    check("5 loader: missing position (102, 305) is zero", 0.0, cube[1, 3].max())
    check("5 loader: duplicate keeps first copy (103, 304)", 103304, cube[2, 2, 0])
    check("5 loader: reported duplicates", 1, info["duplicates"])
    check("5 loader: reported missing", 1, info["missing"])

# 6. gate: a model compared with itself must FAIL (difference exactly 0); a clearly
#    worse reference must make it PASS. Synthetic F3-layout files in a temp folder.
with tempfile.TemporaryDirectory() as tmp:
    sim = make_synthetic_f3.make(Path(tmp) / "f3sim")
    worse = sim / "pred_worse"
    worse.mkdir()
    p_good = np.load(sim / "pred" / "prob_crop.npy")
    rng = np.random.default_rng(5)
    np.save(worse / "prob_crop.npy",
            np.clip(0.6 * p_good + 0.5 * rng.random(p_good.shape), 0, 1).astype(np.float32))
    (worse / "crop.json").write_text((sim / "pred" / "crop.json").read_text())
    for name, b, expect in [("self", sim / "pred", False), ("worse", worse, True)]:
        out = sim / f"gate_{name}"
        subprocess.run([sys.executable, str(REPO / "evaluation" / "compare_models.py"),
                        "--labels-dir", str(sim / "inlines"), "--pred-a", str(sim / "pred"),
                        "--pred-b", str(b), "--block", "2", "--n-boot", "1000", "--out", str(out)],
                       check=True, capture_output=True)
        g = json.loads((out / "gate.json").read_text())
        check(f"6 gate vs {name}: passed == {expect}", float(expect), float(g["passed"]))
        if name == "self":
            check("6 gate vs self: difference", 0.0, g["diff"])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
