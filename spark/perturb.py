#!/usr/bin/env python3
"""Controlled perturbations of a survey crop, written back to the lake as new surveys.

Used to measure how fault-detection quality (true F1) changes as the input data
drifts, and which label-free drift signals track that change.

Kinds (one kind per run, several levels):
  noise    add white Gaussian noise with std = level x the crop's RMS amplitude.
           Seeded per trace from (seed, il, xl), so a rerun gives identical noise
           however Spark splits the data into partitions.
  lowpass  zero-phase Butterworth low-pass (order 6, applied forward and backward
           with sosfiltfilt), cutoff = level Hz. Removes high frequencies, like
           stronger attenuation or a different processing flow.
  gain     multiply by a constant (level). FaultSeg3D normalizes each crop to
           zero mean and unit std before inference, so this changes raw amplitudes
           but should leave predictions untouched: a negative control.
  tgain    multiply by (t / t_mid)^level, t = two-way time of each sample and t_mid
           the middle of the trace. Mimics a missing or extra spherical-divergence
           correction. Global normalization does NOT undo it: deep samples are
           boosted relative to shallow ones.

Mechanics
  * The crop's RMS comes from a Spark higher-order function:
      aggregate(samples, 0D, (acc, x) -> acc + x * x)
    sums squares inside each array without exploding it into one row per sample.
  * The perturbation itself is a pandas UDF: each Arrow batch of traces is stacked
    into a 2-D NumPy array and processed in one vectorized call.
  * Output surveys are named <survey>-<kind>-<level>, e.g. F3-noise-0.5, and written
    with dynamic partition overwrite, so other surveys are untouched. Each gets a
    copy of the parent's surveys row (same schema) and a row in a separate
    `perturbations` table (parent, kind, level, seed, noise std).
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F, types as T

KINDS = ("noise", "lowpass", "gain", "tgain")


def perturb_block(x, il, xl, kind, level, dt_s, t0_index, noise_std, seed):
    """Apply one perturbation to a (n_traces, n_samples) float array. Pure NumPy/SciPy,
    so it can be tested without Spark."""
    x = np.asarray(x, dtype=np.float64)
    if kind == "gain":
        return x * level
    if kind == "tgain":
        ns = x.shape[1]
        t = (t0_index + np.arange(ns) + 0.5) * dt_s           # sample-centre times
        t_mid = (t0_index + ns / 2.0) * dt_s
        return x * (t / t_mid) ** level
    if kind == "lowpass":
        from scipy.signal import butter, sosfiltfilt
        sos = butter(6, level, btype="low", fs=1.0 / dt_s, output="sos")
        return sosfiltfilt(sos, x, axis=1)
    if kind == "noise":
        out = np.empty_like(x)
        for k in range(x.shape[0]):
            rng = np.random.default_rng((seed, int(il[k]), int(xl[k])))
            out[k] = x[k] + rng.standard_normal(x.shape[1]) * noise_std
        return out
    raise ValueError(kind)


def level_name(level):
    return f"{level:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="lake root")
    ap.add_argument("--survey", default="F3")
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--levels", type=float, nargs="+", required=True)
    ap.add_argument("--il-range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    help="inclusive inline range of the crop")
    ap.add_argument("--xl-range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    ap.add_argument("--crop", help="alternatively, crop.json + --meta: use FaultSeg3D's crop box")
    ap.add_argument("--meta", help="meta.json (inline numbers, xl_first) for --crop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--master", default=None)
    ap.add_argument("--driver-memory", default="8g")
    args = ap.parse_args()

    if args.crop:
        crop = json.loads(pathlib.Path(args.crop).read_text())
        meta = json.loads(pathlib.Path(args.meta).read_text())
        lo, hi = crop["lo"], crop["hi"]
        il_range = (int(meta["il_numbers"][lo[0]]), int(meta["il_numbers"][hi[0] - 1]))
        xl_range = (int(meta["xl_first"]) + lo[1], int(meta["xl_first"]) + hi[1] - 1)
    elif args.il_range and args.xl_range:
        il_range, xl_range = tuple(args.il_range), tuple(args.xl_range)
    else:
        ap.error("give --crop and --meta, or --il-range and --xl-range")

    b = SparkSession.builder.appName(f"perturb:{args.survey}:{args.kind}")
    if args.master:
        b = b.master(args.master).config("spark.driver.memory", args.driver_memory)
    spark = (b.config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    root = args.root.rstrip("/")
    t0 = time.time()
    # Read the source partition folder directly (basePath keeps the survey column).
    # Reading "traces/" while overwriting into "traces/" makes Spark refuse with
    # "Cannot overwrite a path that is also being read from".
    src = (spark.read.option("basePath", f"{root}/traces").parquet(f"{root}/traces/survey={args.survey}")
           .where(F.col("il").between(*il_range) & F.col("xl").between(*xl_range)))
    srow = (spark.read.parquet(f"{root}/surveys").where(F.col("survey") == args.survey).first())
    if srow is None:
        raise SystemExit(f"survey {args.survey} not in {root}/surveys")
    dt_s = float(srow["dt_ms"]) / 1000.0
    src = src.cache()

    # RMS of the crop with a higher-order function: sum of squares per array, then over traces.
    agg = src.select(F.aggregate("samples", F.lit(0.0), lambda acc, x: acc + x * x).alias("ss"),
                     F.size("samples").alias("n")).agg(F.sum("ss").alias("ss"), F.sum("n").alias("n"),
                                                       F.count("*").alias("traces")).first()
    rms = float(np.sqrt(agg["ss"] / agg["n"]))
    print(f"{args.survey} crop il {il_range[0]}-{il_range[1]} xl {xl_range[0]}-{xl_range[1]}: "
          f"{agg['traces']} traces, RMS {rms:.4g}, dt {dt_s * 1000:g} ms")

    prows = []
    for level in args.levels:
        name = f"{args.survey}-{args.kind}-{level_name(level)}"
        noise_std = level * rms if args.kind == "noise" else 0.0
        kind, seed = args.kind, args.seed

        @F.pandas_udf(T.ArrayType(T.FloatType()))
        def apply(samples: pd.Series, il: pd.Series, xl: pd.Series) -> pd.Series:
            x = np.stack(samples.to_numpy())
            y = perturb_block(x, il.to_numpy(), xl.to_numpy(), kind, level, dt_s, 0, noise_std, seed)
            return pd.Series(list(y.astype(np.float32)))

        out = src.withColumn("samples", apply("samples", "il", "xl")).withColumn("survey", F.lit(name))
        (out.select("trace_seq", "il", "xl", "cdp_x", "cdp_y", "samples", "survey")
            .write.mode("overwrite").partitionBy("survey").parquet(f"{root}/traces"))

        row = srow.asDict()
        row["survey"] = name
        (spark.createDataFrame([row], schema=spark.read.parquet(f"{root}/surveys").schema)
         .write.mode("overwrite").partitionBy("survey").parquet(f"{root}/surveys"))
        prows.append((name, args.survey, args.kind, float(level), args.seed, float(noise_std), rms,
                      il_range[0], il_range[1], xl_range[0], xl_range[1]))
        print(f"wrote {name}  [{time.time() - t0:.1f} s]")

    (spark.createDataFrame(prows, "survey string, parent string, kind string, level double, seed long, "
                                  "noise_std double, parent_rms double, il_first int, il_last int, "
                                  "xl_first int, xl_last int")
     .write.mode("overwrite").partitionBy("survey").parquet(f"{root}/perturbations"))
    spark.stop()


if __name__ == "__main__":
    main()
