#!/usr/bin/env python3
"""Run the BigQuery QC queries in sql/bq/*.sql against the seismic dataset.

For each query:
  1. Dry run: BigQuery plans the query and reports how many bytes it would scan,
     without running it or charging for it.
  2. Real run with the query cache off, so the numbers reflect actual work.
  3. Print row count, bytes processed, bytes billed, slot time and the first rows.

Why bytes matter: on-demand pricing charges by bytes scanned, with a minimum of
10 MB per table referenced. BigQuery stores data by column, so a query that never
touches traces.samples scans only a few percent of the table, the same column
pruning the Spark plan showed as ReadSchema.

Parameters are passed as typed query parameters (@survey STRING, @jump FLOAT64,
...). Table names can't be parameters, so {ds} in the SQL is replaced with
project.dataset after checking it is a plain identifier.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

SQL_DIR = Path(__file__).resolve().parent.parent / "sql" / "bq"
PARAM_TYPES = {"survey": "STRING", "jump": "FLOAT64", "spike": "FLOAT64", "xl_step": "INT64"}
DS_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]\.[A-Za-z_][A-Za-z0-9_]{0,1023}$")


def query_names():
    return sorted(p.stem for p in SQL_DIR.glob("*.sql"))


def build_query(name, ds, params, dry_run=False):
    if not DS_PATTERN.match(ds):
        raise ValueError(f"not a valid project.dataset: {ds!r}")
    text = (SQL_DIR / f"{name}.sql").read_text().replace("{ds}", ds)
    used = set(re.findall(r"@([A-Za-z_]\w*)", re.sub(r"--[^\n]*", "", text)))
    qp = [bigquery.ScalarQueryParameter(k, PARAM_TYPES[k], params[k]) for k in sorted(used)]
    cfg = bigquery.QueryJobConfig(query_parameters=qp, dry_run=dry_run, use_query_cache=False)
    return text, cfg


def run_query_bq(client, name, ds, params):
    """Run one query and return (DataFrame, finished job)."""
    text, cfg = build_query(name, ds, params)
    job = client.query(text, job_config=cfg)
    df = job.result().to_dataframe()
    return df, job


def mb(n):
    return f"{(n or 0) / 1e6:8.2f} MB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", default=None, help="default: the client's default project")
    ap.add_argument("--dataset", default="seismic")
    ap.add_argument("--location", default=None, help="dataset location, e.g. US or us-central1")
    ap.add_argument("--survey", required=True)
    ap.add_argument("--queries", default="", help="comma list of prefixes, e.g. 01,04")
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--jump", type=float, default=2.0)
    ap.add_argument("--spike", type=float, default=2.0)
    ap.add_argument("--xl-step", type=int, default=1)
    args = ap.parse_args()

    client = bigquery.Client(project=args.project, location=args.location)
    ds = f"{client.project}.{args.dataset}"
    params = {"survey": args.survey, "jump": args.jump, "spike": args.spike,
              "xl_step": args.xl_step}

    names = query_names()
    if args.queries:
        names = [n for n in names if any(n.startswith(w) for w in args.queries.split(","))]

    rows = []
    for name in names:
        text, cfg = build_query(name, ds, params, dry_run=True)
        est = client.query(text, job_config=cfg).total_bytes_processed
        df, job = run_query_bq(client, name, ds, params)
        secs = (job.ended - job.started).total_seconds()
        print(f"\n=== {name}: {len(df)} rows  dry-run {mb(est)}  processed {mb(job.total_bytes_processed)}"
              f"  billed {mb(job.total_bytes_billed)}  slot {job.slot_millis or 0} ms  {secs:.1f} s")
        if args.show:
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                print(df.head(args.show).to_string(index=False))
        rows.append((name, len(df), job.total_bytes_processed or 0, job.total_bytes_billed or 0))

    print("\nquery                     rows      processed         billed")
    for name, n, p, b in rows:
        print(f"{name:22s} {n:7d}  {mb(p)}  {mb(b)}")
    total = sum(r[3] for r in rows)
    print(f"total billed {mb(total)} (on-demand: the first 1 TiB scanned per month is free)")


if __name__ == "__main__":
    sys.exit(main())
