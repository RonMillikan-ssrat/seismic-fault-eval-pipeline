#!/usr/bin/env python3
"""Check the QC SQL results against the synthetic truth file.

Every expected answer is computed here in plain NumPy from the truth file,
independently of the SQL, then compared with what the query returned.

  --engine spark  runs sql/*.sql with Spark on the local lake (--root)
  --engine bq     runs sql/bq/*.sql in BigQuery (--project, --dataset)

Also imported by tests/test_qc.py (qc_checks(), spark_query(), bq_query()).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def params_for(survey):
    return {"survey": survey, "jump": 2.0, "spike": 2.0, "xl_step": 1}


def spark_query(root, survey, master="local[*]"):
    """Return (q, stop): q(name) -> pandas result of sql/<name>.sql in Spark."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession
    sys.path.insert(0, str(REPO / "spark"))
    from qc_sql import register_views, run_query

    spark = (SparkSession.builder.master(master)
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    register_views(spark, str(root).rstrip("/"))
    params = params_for(survey)
    return (lambda name: run_query(spark, name, params).toPandas()), spark.stop


def bq_query(survey, project=None, dataset="seismic", location=None):
    """Return q(name) -> pandas result of sql/bq/<name>.sql in BigQuery."""
    from google.cloud import bigquery
    sys.path.insert(0, str(REPO / "gcp"))
    from qc_bq import run_query_bq

    client = bigquery.Client(project=project, location=location)
    ds = f"{client.project}.{dataset}"
    params = params_for(survey)
    return lambda name: run_query_bq(client, name, ds, params)[0]


def qc_checks(q, truth_path, survey):
    """Run every QC query through q and return [(label, expected, got), ...]."""
    tr = np.load(truth_path)
    il, xl, data = tr["il"], tr["xl"], tr["data"]
    il0, il1 = tr["il_range"]
    xl0, xl1 = tr["xl_range"]
    out = []

    # 01 missing positions
    present = set(zip(il.tolist(), xl.tolist()))
    grid = {(i, x) for i in range(il0, il1 + 1) for x in range(xl0, xl1 + 1)}
    m = q("01_missing_positions")
    out.append(("01 missing positions (count)", len(grid - present), len(m)))
    out.append(("01 missing positions (exact set)", True,
                set(zip(m.il.tolist(), m.xl.tolist())) == grid - present))

    # 02 duplicates
    out.append(("02 duplicate traces", len(tr["dup_src"]), len(q("02_duplicates"))))

    # 03 dead traces (duplicates of a dead trace are dead too)
    dead_truth = int((np.abs(data).max(axis=1) == 0).sum())
    dd = q("03_dead_traces")
    out.append(("03 dead traces", dead_truth, int(dd.n_dead.sum()) if len(dd) else 0))

    # 04 amplitude jumps: one flag where the patch starts and one where it ends, on each
    # inline crossing the patch, unless the patch runs to the edge of the line.
    if str(tr["gain_patch"]):
        pil0, pil1, pxl0, pxl1, _ = str(tr["gain_patch"]).split(",")
        pil0, pil1, pxl0, pxl1 = map(int, (pil0, pil1, pxl0, pxl1))
        live = np.abs(data).max(axis=1) > 0
        expected = 0
        for i in range(pil0, pil1 + 1):
            xs = np.unique(xl[(il == i) & live])
            inside = (xs >= pxl0) & (xs <= pxl1)
            expected += int(np.count_nonzero(np.diff(inside.astype(int))))
        j = q("04_amplitude_jumps")
        at_edges = j[(j.il >= pil0) & (j.il <= pil1)]
        out.append(("04 jumps at gain-patch edges", expected, len(at_edges)))
        out.append(("04 jumps elsewhere (false flags)", 0, len(j) - len(at_edges)))

    # 06 islands
    islands = 0
    for i in range(il0, il1 + 1):
        xs = np.unique(xl[il == i])
        if xs.size:
            islands += 1 + int(np.count_nonzero(np.diff(xs) > 1))
    out.append(("06 crossline islands", islands, len(q("06_xl_islands"))))

    # 08 summary row count matches traces
    s = q("08_survey_summary")
    s = s[s.survey == survey]
    out.append(("08 traces in stats == traces in file", int(s.n_traces_file.iloc[0]),
                int(s.n_traces_stats.iloc[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("--survey", required=True)
    ap.add_argument("--engine", choices=["spark", "bq"], default="spark")
    ap.add_argument("--root", help="lake root (spark engine)")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--project", default=None, help="GCP project (bq engine)")
    ap.add_argument("--dataset", default="seismic")
    ap.add_argument("--location", default=None)
    args = ap.parse_args()

    stop = None
    if args.engine == "spark":
        if not args.root:
            ap.error("--root is required with --engine spark")
        q, stop = spark_query(args.root, args.survey, args.master)
    else:
        q = bq_query(args.survey, args.project, args.dataset, args.location)
    print(f"engine: {args.engine}")

    results = []
    for label, expected, got in qc_checks(q, args.truth, args.survey):
        ok = expected == got
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {label:38s} expected {expected!s:>8}  got {got!s:>8}")
    if stop:
        stop()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
