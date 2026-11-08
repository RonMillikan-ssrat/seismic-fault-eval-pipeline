#!/usr/bin/env python3
"""Gate: does model A beat model B on held-out inlines, with a CI that excludes zero?

Both models are scored on the same labeled F3 inlines (score_f3.load), in the
corridor around the picks, at the same tolerance.

  1. Split the labeled inlines into calibration and test sets by alternating
     contiguous blocks (fault_metrics.alternate_blocks).
  2. Choose each model's best-F1 threshold on the calibration blocks only.
  3. On the test blocks, paired block bootstrap of F1(A) - F1(B): every bootstrap
     replicate draws the same blocks for both models, so difficulty they share
     cancels out of the difference.
  4. Gate passes if the lower end of the CI is above --min-diff (default 0).

Choosing thresholds on separate inlines matters for a fair comparison: picking
each model's threshold on the test inlines would favor whichever model has the
noisier curve.

Writes gate.json for the pipeline's conditional registration step.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fault_metrics as fm  # noqa: E402
from score_f3 import load  # noqa: E402


def score(P, label, region, tol, calib, test):
    c = fm.curve_counts(P, label, tol, region=region)
    i = fm.pick_threshold(c, calib)
    p, r, f1 = fm.prf(*[x[i] if np.ndim(x) else x for x in c.totals(test)])
    return c, i, dict(threshold=float(c.thresholds[i]), precision=float(p), recall=float(r), f1=float(f1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--pred-a", required=True, help="candidate model (e.g. FaultSeg3D pred dir)")
    ap.add_argument("--pred-b", required=True, help="reference (e.g. coherence baseline dir)")
    ap.add_argument("--name-a", default="FaultSeg3D")
    ap.add_argument("--name-b", default="coherence")
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--tol-sweep", type=float, nargs="*", default=None,
                    help="also report the paired difference at these tolerances")
    ap.add_argument("--region-radius", type=float, default=40.0)
    ap.add_argument("--thin", action="store_true", help="thin both models' maps before scoring")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--min-diff", type=float, default=0.0, help="required lower CI bound on F1(A)-F1(B)")
    ap.add_argument("--out", default=None, help="write gate.json here")
    args = ap.parse_args()

    Pa, label, valid, il, crop_a = load(args.labels_dir, args.pred_a)
    Pb, label_b, _, il_b, crop_b = load(args.labels_dir, args.pred_b)
    if il != il_b or crop_a["lo"] != crop_b["lo"] or crop_a["hi"] != crop_b["hi"]:
        raise SystemExit("the two prediction folders must cover the same crop")
    if args.thin:
        Pa, Pb = fm.thin_faults(Pa), fm.thin_faults(Pb)
    region = valid & fm.label_region(label, args.region_radius)
    calib, test = fm.alternate_blocks(len(il), args.block)
    print(f"{len(il)} labeled inlines: {len(calib)} calibrate thresholds, {len(test)} test; "
          f"corridor {args.region_radius:g}, thinning {'on' if args.thin else 'off'}")

    rows = []
    for tol in (args.tol_sweep or [args.tol]):
        ca, ia, ra = score(Pa, label, region, tol, calib, test)
        cb, ib, rb = score(Pb, label, region, tol, calib, test)
        d = fm.paired_bootstrap(ca.subset(test), ia, cb.subset(test), ib,
                                block=args.block, n_boot=args.n_boot)
        rows.append(dict(tol=tol, a=ra, b=rb, **d))

    print(f"\n{'tol':>5} | {args.name_a + ' F1 (t)':>18} | {args.name_b + ' F1 (t)':>18} | "
          f"{'F1 diff':>8} {'95% CI':>18} {'P(diff<=0)':>10}")
    for r in rows:
        print(f"{r['tol']:5g} | {r['a']['f1']:10.3f} ({r['a']['threshold']:.3f}) | "
              f"{r['b']['f1']:10.3f} ({r['b']['threshold']:.3f}) | {r['diff']:8.3f} "
              f"[{r['diff_ci'][0]:7.3f}, {r['diff_ci'][1]:7.3f}] {r['frac_le_0']:10.4f}")

    main_row = next(r for r in rows if r["tol"] == args.tol) if args.tol in [r["tol"] for r in rows] else rows[0]
    passed = main_row["diff_ci"][0] > args.min_diff
    print(f"\nGATE at tol {main_row['tol']:g}: {'PASS' if passed else 'FAIL'}  "
          f"(lower CI bound {main_row['diff_ci'][0]:.3f} {'>' if passed else '<='} {args.min_diff:g})")

    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        gate = dict(passed=bool(passed), tol=main_row["tol"], min_diff=args.min_diff,
                    diff=main_row["diff"], diff_ci=main_row["diff_ci"], candidate=args.name_a,
                    reference=args.name_b, block=args.block, n_test_inlines=int(len(test)),
                    thin=args.thin, region_radius=args.region_radius, rows=rows)
        (out / "gate.json").write_text(json.dumps(gate, indent=1))
        print(f"wrote {out / 'gate.json'}")
    # Exit 0 either way: a failed gate is a result, not an error. The pipeline reads
    # gate.json and skips registration; a non-zero exit would fail the whole step.
    return 0


if __name__ == "__main__":
    sys.exit(main())
