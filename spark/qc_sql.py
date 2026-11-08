#!/usr/bin/env python3
"""Run the QC queries in sql/*.sql against the trace lakehouse.

Views registered before the queries run:
  traces_hdr   traces table WITHOUT the samples column (header fields only)
  trace_stats  per-trace attributes from trace_attributes.py
  surveys      one row of file metadata per survey

Queries use named parameter markers (:survey, :jump, ...). Spark binds them through
spark.sql(text, args={...}) instead of string formatting, so values are never
pasted into the SQL text. BigQuery uses @survey for the same thing.

Writes <root>/qc_summary/survey=<name>/: one row per query (row count, run time),
which the monitoring step reads later.
"""
import argparse
import os
import sys
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

# Spark workers must run the same Python as this script (the venv). Without this,
# local mode looks for a default "python3"/"python" on PATH, which may not exist
# or may lack the packages. Dataproc sets PYSPARK_PYTHON itself, so keep any value.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def register_views(spark, root):
    traces = spark.read.parquet(f"{root}/traces")
    traces.drop("samples").createOrReplaceTempView("traces_hdr")
    spark.read.parquet(f"{root}/trace_stats").createOrReplaceTempView("trace_stats")
    spark.read.parquet(f"{root}/surveys").createOrReplaceTempView("surveys")


def query_names():
    return sorted(p.stem for p in SQL_DIR.glob("*.sql"))


def run_query(spark, name, params):
    """Run sql/<name>.sql, binding only the parameters the query uses."""
    text = (SQL_DIR / f"{name}.sql").read_text()
    used = set(re.findall(r":([A-Za-z_]\w*)", re.sub(r"--[^\n]*", "", text)))
    return spark.sql(text, args={k: v for k, v in params.items() if k in used})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--survey", required=True)
    ap.add_argument("--queries", default="", help="comma list of prefixes, e.g. 01,04 (default: all)")
    ap.add_argument("--show", type=int, default=10, help="rows to print per query")
    ap.add_argument("--explain", default="", help="query prefix to print the physical plan for")
    ap.add_argument("--jump", type=float, default=2.0, help="neighbor RMS ratio flagged by 04")
    ap.add_argument("--spike", type=float, default=2.0, help="RMS / rolling-average ratio flagged by 05")
    ap.add_argument("--xl-step", type=int, default=1, help="crossline increment used by 06")
    ap.add_argument("--master", default=None)
    ap.add_argument("--driver-memory", default="4g")
    args = ap.parse_args()

    b = SparkSession.builder.appName(f"qc_sql:{args.survey}")
    if args.master:
        b = b.master(args.master).config("spark.driver.memory", args.driver_memory)
    spark = (b.config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    root = args.root.rstrip("/")
    register_views(spark, root)
    params = {"survey": args.survey, "jump": args.jump, "spike": args.spike,
              "xl_step": args.xl_step}

    names = query_names()
    if args.queries:
        wanted = args.queries.split(",")
        names = [n for n in names if any(n.startswith(w) for w in wanted)]

    summary = []
    for name in names:
        df = run_query(spark, name, params).cache()
        t0 = time.time()
        n = df.count()
        dt = time.time() - t0
        print(f"\n=== {name}: {n} rows ({dt:.2f} s)")
        if args.show:
            df.show(args.show, truncate=60)
        if args.explain and name.startswith(args.explain):
            df.explain(mode="formatted")
        summary.append((name, n, round(dt, 3)))
        df.unpersist()

    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (spark.createDataFrame(summary, "query string, n_rows long, seconds double")
     .withColumn("survey", F.lit(args.survey)).withColumn("run_ts", F.lit(run_ts))
     .write.mode("overwrite").partitionBy("survey").parquet(f"{root}/qc_summary"))
    print(f"\nwrote {root}/qc_summary/survey={args.survey}")
    spark.stop()


if __name__ == "__main__":
    main()
