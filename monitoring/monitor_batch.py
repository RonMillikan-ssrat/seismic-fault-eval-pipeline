#!/usr/bin/env python3
"""Label-free monitoring of a new batch against a reference survey.

Two commands:

  calibrate   Derive alert thresholds from a perturbation study's drift_table.csv
              (scripts/perturbation_study.sh), where true F1 is known.
  check       Compute the same signals for a new batch, with no labels, and report
              OK / WARN / ALERT against those thresholds.

Signals (all label-free)
  frac_pos_pct    % change in the fraction of voxels with fault probability > 0.5,
                  relative to the reference survey's predictions
  d_centroid_hz   change in the median spectral centroid of the traces
  psi_rms_norm    PSI of trace RMS / survey median RMS
  psi_rms_raw     PSI of raw trace RMS
Each is tested in a direction (fall or rise), because direction mattered in the
F3 study: a fall in predicted fraction tracked the F1 drop, a rise much less so.

How calibrate sets a threshold
  For a signal, a direction and a harm level h (an F1 drop, at the deployed
  threshold): split the study's surveys into harmed (drop >= h) and not harmed.
  If every harmed survey has a larger signal than every unharmed one, the
  threshold is the midpoint of the gap. If they interleave, the signal cannot
  separate harm at that level and is reported as context only, never alerted on.
  Two harm levels give the WARN and ALERT thresholds.

Limits (say these when presenting it)
  * Thresholds come from one survey and a handful of synthetic perturbations.
  * The study holds the geology fixed. On a new area the predicted fault fraction
    also changes because the faulting differs, so the reference predictions must
    come from comparable data.
  * A WARN or ALERT is a reason to have an interpreter look (see spot checks), not
    a measurement of accuracy.

Spot checks: `check` also lists inlines for an interpreter to review, the k with
the highest mean prediction entropy (where the model is least sure) plus k chosen
at random (so the review isn't biased toward the hard cases only).

Exit code is 0 whatever the status: a WARN is a result, not a failure.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "monitoring"))
sys.path.insert(0, str(REPO / "baseline"))
import drift_signals as ds  # noqa: E402
from coherence import crop_ranges  # noqa: E402

CANDIDATES = [("frac_pos_pct", "fall"), ("frac_pos_pct", "rise"),
              ("d_centroid_hz", "rise"), ("d_centroid_hz", "fall"),
              ("psi_rms_norm", "rise"), ("psi_rms_raw", "rise")]
LEVELS = {"OK": 0, "WARN": 1, "ALERT": 2}


# --------------------------------------------------------------- calibrate
def magnitude(values, direction):
    """Size of a signed signal in one direction; surveys moving the other way are dropped."""
    v = np.asarray(values, dtype=float)
    keep = v <= 0 if direction == "fall" else v >= 0
    return np.abs(v[keep]), keep


def separate(mag, drop, h):
    """Threshold separating surveys with F1 drop >= h from the rest, or (None, reason)."""
    bad, ok = mag[drop >= h], mag[drop < h]
    if bad.size == 0:
        return None, f"no survey moving this way lost {h:g} F1 or more"
    if ok.size == 0 or ok.max() < bad.min():
        lo = ok.max() if ok.size else 0.0
        return float((lo + bad.min()) / 2), None
    return None, (f"not separable at a drop of {h:g}: an unharmed survey reads {ok.max():.3g}, "
                  f"a harmed one {bad.min():.3g}")


def calibrate(table, reference, warn_drop=0.05, alert_drop=0.15):
    df = pd.read_csv(table) if not isinstance(table, pd.DataFrame) else table.copy()
    drop_col = "f1_fixed_drop" if "f1_fixed_drop" in df.columns else "f1_drop"
    ref = df[df.survey == reference].iloc[0]
    df["frac_pos_pct"] = (df.frac_pos / ref.frac_pos - 1.0) * 100.0
    rules, info = [], []
    for signal, direction in CANDIDATES:
        mag, keep = magnitude(df[signal], direction)
        drop = df.loc[keep, drop_col].to_numpy(float)
        warn, why = separate(mag, drop, warn_drop)
        if warn is None:
            info.append(dict(signal=signal, direction=direction, reason=why))
            continue
        alert, why_a = separate(mag, drop, alert_drop)
        rules.append(dict(signal=signal, direction=direction, warn=round(warn, 4),
                          alert=None if alert is None else round(alert, 4),
                          alert_note=why_a, n_surveys=int(keep.sum())))
    return dict(reference=reference, drop_column=drop_col, warn_drop=warn_drop, alert_drop=alert_drop,
                n_surveys=int(len(df)), surveys=df.survey.tolist(), rules=rules, context_only=info)


# ------------------------------------------------------------------- check
def evaluate(values, thresholds):
    """Compare measured signals with the calibrated rules. Returns (rows, overall status)."""
    rows, overall = [], "OK"
    for r in thresholds["rules"]:
        v = values.get(r["signal"])
        if v is None or not np.isfinite(v):
            continue
        m = -v if r["direction"] == "fall" else v          # size in the rule's direction
        level = "OK"
        if m >= r["warn"]:
            level = "WARN"
        if r["alert"] is not None and m >= r["alert"]:
            level = "ALERT"
        rows.append(dict(signal=r["signal"], direction=r["direction"], value=float(v),
                         warn=r["warn"], alert=r["alert"], level=level))
        if LEVELS[level] > LEVELS[overall]:
            overall = level
    for c in thresholds["context_only"]:
        v = values.get(c["signal"])
        if v is not None and np.isfinite(v):
            rows.append(dict(signal=c["signal"], direction=c["direction"], value=float(v),
                             warn=None, alert=None, level="context"))
    return rows, overall


def spot_check(prob, k, seed=0, margin=0):
    """Inline indices for interpreter review: k highest mean entropy + k random others.
    `margin` inlines at each end of the crop are skipped: on F3 the first and last
    few inlines had the highest entropy on every survey, clean F3 included, which
    points to the edge of the predicted volume rather than to the data."""
    ent = np.full(prob.shape[0], -np.inf)
    for i in range(margin, prob.shape[0] - margin):
        p = np.clip(np.asarray(prob[i], dtype=np.float32), 1e-7, 1 - 1e-7)
        ent[i] = float(-(p * np.log(p) + (1 - p) * np.log1p(-p)).mean())
    order = np.argsort(ent)[::-1]
    top = order[:k]
    rest = np.setdiff1d(np.arange(margin, prob.shape[0] - margin), top)
    rnd = np.random.default_rng(seed).choice(rest, size=min(k, rest.size), replace=False)
    return sorted(int(i) for i in top), sorted(int(i) for i in rnd)


def check(args):
    thresholds = json.loads(pathlib.Path(args.thresholds).read_text())
    il_numbers = None
    if args.crop:
        crop = json.loads(pathlib.Path(args.crop).read_text())
        meta = json.loads(pathlib.Path(args.meta).read_text())
        il_r, xl_r, _ = crop_ranges(crop, meta)
        bounds = ((il_r[0], il_r[1] - 1), (xl_r[0], xl_r[1] - 1))
        il_numbers = list(range(il_r[0], il_r[1]))
    else:
        bounds = (tuple(args.il_range), tuple(args.xl_range))

    if args.engine == "bq":
        inp = ds.input_signals_bq(args.bq_table, args.reference, bounds, args.project, args.location)
    else:
        inp = ds.input_signals(args.root, args.reference, bounds, args.master, args.driver_memory)
    row = inp[inp.survey == args.survey]
    if row.empty:
        raise SystemExit(f"{args.survey} has no trace_stats inside the crop: run spark/trace_attributes.py"
                         + (" and gcp/setup_monitor.sh" if args.engine == "bq" else ""))
    values = {k: float(row[k].iloc[0]) for k in ("psi_rms_raw", "psi_rms_norm", "d_centroid_hz")}

    new = np.load(pathlib.Path(args.pred_dir) / "prob_crop.npy", mmap_mode="r")
    ref = np.load(pathlib.Path(args.ref_pred_dir) / "prob_crop.npy", mmap_mode="r")
    o_new, o_ref = ds.output_signals(new), ds.output_signals(ref)
    values["frac_pos_pct"] = (o_new["frac_pos"] / o_ref["frac_pos"] - 1.0) * 100.0

    rows, status = evaluate(values, thresholds)
    print(f"{args.survey} vs reference {args.reference}  (thresholds from {thresholds['n_surveys']} "
          f"study surveys; WARN = signal seen with an F1 drop >= {thresholds['warn_drop']:g}, "
          f"ALERT >= {thresholds['alert_drop']:g})")
    print(f"{'signal':15s} {'dir':5s} {'value':>9} {'warn':>8} {'alert':>8}  level")
    fmt = lambda x: "       -" if x is None else f"{x:8.3g}"
    for r in rows:
        print(f"{r['signal']:15s} {r['direction']:5s} {r['value']:9.3g} {fmt(r['warn'])} {fmt(r['alert'])}  {r['level']}")
    print(f"STATUS: {status}")

    top, rnd = spot_check(new, args.spot_check, args.seed, args.spot_margin) if args.spot_check else ([], [])
    name = (lambda i: il_numbers[i]) if il_numbers and len(il_numbers) == new.shape[0] else (lambda i: i)
    if top:
        print(f"spot checks for an interpreter: highest entropy {[name(i) for i in top]}, "
              f"random {[name(i) for i in rnd]}")

    report = dict(survey=args.survey, reference=args.reference, status=status, engine=args.engine, signals=rows,
                  values=values, predicted_fraction=dict(new=o_new["frac_pos"], reference=o_ref["frac_pos"]),
                  spot_check=dict(highest_entropy=[name(i) for i in top], random=[name(i) for i in rnd]))
    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "monitor_report.json").write_text(json.dumps(report, indent=1))
        print(f"wrote {out / 'monitor_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="thresholds from a study's drift_table.csv")
    c.add_argument("--drift-table", required=True)
    c.add_argument("--reference", default="F3")
    c.add_argument("--warn-drop", type=float, default=0.05, help="F1 drop that should raise WARN")
    c.add_argument("--alert-drop", type=float, default=0.15, help="F1 drop that should raise ALERT")
    c.add_argument("--out", required=True, help="thresholds JSON to write")

    k = sub.add_parser("check", help="score a new batch against the thresholds")
    k.add_argument("--engine", choices=["spark", "bq"], default="spark",
                   help="where the input signals are computed: Spark SQL on the lake, or BigQuery")
    k.add_argument("--root", help="lake root with trace_stats for both surveys (spark engine)")
    k.add_argument("--bq-table", help="project.dataset.table holding trace_stats (bq engine)")
    k.add_argument("--project", default=None)
    k.add_argument("--location", default=None)
    k.add_argument("--reference", default="F3")
    k.add_argument("--survey", required=True)
    k.add_argument("--pred-dir", required=True, help="new batch's prob_crop.npy folder")
    k.add_argument("--ref-pred-dir", required=True, help="reference survey's prob_crop.npy folder")
    k.add_argument("--thresholds", required=True)
    k.add_argument("--crop"); k.add_argument("--meta")
    k.add_argument("--il-range", type=int, nargs=2); k.add_argument("--xl-range", type=int, nargs=2)
    k.add_argument("--spot-check", type=int, default=5)
    k.add_argument("--spot-margin", type=int, default=8,
                   help="inlines skipped at each end of the crop when choosing spot checks")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--out")
    k.add_argument("--master", default="local[*]")
    k.add_argument("--driver-memory", default="8g")
    args = ap.parse_args()

    if args.cmd == "calibrate":
        t = calibrate(args.drift_table, args.reference, args.warn_drop, args.alert_drop)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(t, indent=1))
        print(f"from {t['n_surveys']} surveys, harm = F1 drop in {t['drop_column']}")
        for r in t["rules"]:
            a = "none" if r["alert"] is None else f"{r['alert']:g}"
            print(f"  RULE  {r['signal']:14s} {r['direction']:5s} warn at {r['warn']:g}, alert at {a}"
                  + (f"   ({r['alert_note']})" if r["alert"] is None else ""))
        for i in t["context_only"]:
            print(f"  context only  {i['signal']:14s} {i['direction']:5s} {i['reason']}")
        print(f"wrote {args.out}")
        return
    if not (args.crop and args.meta) and not (args.il_range and args.xl_range):
        ap.error("check needs --crop and --meta, or --il-range and --xl-range")
    if args.engine == "spark" and not args.root:
        ap.error("--root is required with --engine spark")
    if args.engine == "bq" and not args.bq_table:
        ap.error("--bq-table is required with --engine bq")
    check(args)


if __name__ == "__main__":
    main()
