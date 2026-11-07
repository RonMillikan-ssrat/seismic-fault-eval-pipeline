#!/usr/bin/env python3
"""Per-trace attributes in PySpark, computed three ways and timed.

Variants (same math, different execution):
  python      plain Python UDF: rows pickled one at a time to the Python worker,
              function called once per trace
  python_arrow  Python UDF with useArrow=True: rows sent in Arrow batches, but the
              function is still called once per trace
  pandas      pandas UDF: Arrow batches arrive as pandas Series; one NumPy call
              handles thousands of traces at once

python vs python_arrow isolates the serialization cost.
python_arrow vs pandas isolates the vectorization gain.

Each variant is forced with the "noop" writer, which computes every column and
then discards the result, so the timing excludes output I/O.

The sample interval dt comes from the surveys table through a broadcast join.
"""
import argparse
import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F, types as T

ATTR_FIELDS = ["rms", "mean", "std", "max_abs", "zero_frac", "n_nan",
               "dom_freq_hz", "spec_centroid_hz"]
ATTR_SCHEMA = T.StructType([T.StructField(c, T.DoubleType()) for c in ATTR_FIELDS])


def attrs_2d(x, dt_ms):
    """Vectorized attributes for a 2-D array: rows = traces, cols = samples."""
    x = np.asarray(x, dtype=np.float64)
    n_nan = np.isnan(x).sum(axis=1)
    x = np.nan_to_num(x)
    ns = x.shape[1]
    absx = np.abs(x)

    # Amplitude spectrum of the demeaned trace.
    spec = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1))
    freqs = np.fft.rfftfreq(ns, d=dt_ms / 1000.0)
    total = spec.sum(axis=1)
    live = total > 0
    centroid = np.full(x.shape[0], np.nan)
    centroid[live] = (spec[live] * freqs).sum(axis=1) / total[live]
    dom = np.where(live, freqs[np.argmax(spec, axis=1)], np.nan)

    return {
        "rms": np.sqrt((x ** 2).mean(axis=1)),
        "mean": x.mean(axis=1),
        "std": x.std(axis=1),
        "max_abs": absx.max(axis=1),
        "zero_frac": (x == 0).mean(axis=1),
        "n_nan": n_nan.astype(np.float64),
        "dom_freq_hz": dom,
        "spec_centroid_hz": centroid,
    }


def attrs_row(samples, dt_ms):
    """One trace at a time (used by the two row-wise Python UDFs)."""
    a = attrs_2d(np.asarray(samples, dtype=np.float64)[None, :], dt_ms)
    return tuple(float(a[c][0]) for c in ATTR_FIELDS)


py_udf = F.udf(attrs_row, ATTR_SCHEMA, useArrow=False)
py_arrow_udf = F.udf(attrs_row, ATTR_SCHEMA, useArrow=True)


@F.pandas_udf(ATTR_SCHEMA)
def pandas_attrs(samples: pd.Series, dt_ms: pd.Series) -> pd.DataFrame:
    # A batch can mix surveys with different ns or dt, so group within the batch.
    out = pd.DataFrame(index=samples.index, columns=ATTR_FIELDS, dtype="float64")
    lengths = samples.map(len)
    for (ns, dt), idx in samples.groupby([lengths, dt_ms]).groups.items():
        block = np.stack(samples.loc[idx].to_numpy())
        a = attrs_2d(block, dt)
        out.loc[idx, ATTR_FIELDS] = np.column_stack([a[c] for c in ATTR_FIELDS])
    return out


VARIANTS = {"python": py_udf, "python_arrow": py_arrow_udf, "pandas": pandas_attrs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="same --out root used by segy_to_parquet.py")
    ap.add_argument("--survey", required=True)
    ap.add_argument("--variants", default="python,python_arrow,pandas")
    ap.add_argument("--repeat", type=int, default=2,
                    help="timed runs per variant; the first run includes Python worker start-up")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N traces (0 = all)")
    ap.add_argument("--master", default=None)
    ap.add_argument("--driver-memory", default="4g")
    ap.add_argument("--no-write", action="store_true", help="time only; skip writing trace_stats")
    args = ap.parse_args()

    b = SparkSession.builder.appName(f"trace_attributes:{args.survey}")
    if args.master:
        b = b.master(args.master).config("spark.driver.memory", args.driver_memory)
    spark = (b.config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    root = args.root.rstrip("/")
    traces = spark.read.parquet(f"{root}/traces").where(F.col("survey") == args.survey)
    if args.limit:
        traces = traces.orderBy("trace_seq").limit(args.limit)
    surveys = spark.read.parquet(f"{root}/surveys").select("survey", "dt_ms")

    # surveys is a few rows, so broadcast it: each executor gets a copy and the
    # large traces table is joined in place, with no shuffle.
    base = traces.join(F.broadcast(surveys), "survey")
    base.cache()
    n = base.count()  # materializes the cache, so every variant reads the same in-memory input
    print(f"{n} traces, {base.rdd.getNumPartitions()} partitions")

    results = {}
    for name in args.variants.split(","):
        fn = VARIANTS[name]
        df = base.select("survey", "trace_seq", "il", "xl", fn("samples", "dt_ms").alias("a"))
        times = []
        for _ in range(args.repeat):
            t0 = time.time()
            df.write.format("noop").mode("overwrite").save()
            times.append(time.time() - t0)
        results[name] = df
        print(f"{name:13s} runs: " + ", ".join(f"{t:6.2f}s" for t in times)
              + f"   best {min(times):6.2f}s   {n / min(times):9.0f} traces/s")

    # Check the variants agree (on a sample) before trusting the timings.
    names = list(results)
    if len(names) > 1:
        ref = results[names[0]].select("trace_seq", "a.*").orderBy("trace_seq").limit(2000).toPandas()
        for other in names[1:]:
            o = results[other].select("trace_seq", "a.*").orderBy("trace_seq").limit(2000).toPandas()
            diff = np.nanmax(np.abs(ref[ATTR_FIELDS].to_numpy() - o[ATTR_FIELDS].to_numpy()))
            print(f"max |{names[0]} - {other}| over 2000 traces: {diff:.3g}")

    if not args.no_write:
        stats = (base.select("survey", "trace_seq", "il", "xl",
                             pandas_attrs("samples", "dt_ms").alias("a"))
                 .select("survey", "trace_seq", "il", "xl", "a.*"))
        stats.write.mode("overwrite").partitionBy("survey").parquet(f"{root}/trace_stats")
        print(f"wrote {root}/trace_stats/survey={args.survey}")
        stats.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
