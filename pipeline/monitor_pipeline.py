#!/usr/bin/env python3
"""KFP v2 monitoring pipeline: check one new batch against the reference, no labels.

    monitor ──► Metrics (each signal, status as 0/1/2), report (JSON), status string

The step calls monitoring/monitor_batch.py check, which needs:
  * trace_stats in the lake for the reference and the new survey
    (spark/trace_attributes.py), and
  * FaultSeg3D predictions for both (infer/faultseg_from_lake.py).

Run locally:
  python pipeline/monitor_pipeline.py local --repo-dir . --lake ~/lake --reference F3 \
      --survey F3-noise-0.5 --pred-dir ~/f3_study/F3-noise-0.5/pred --ref-pred-dir ~/f3_study/F3/pred \
      --thresholds monitoring/thresholds_f3.json \
      --crop ~/devl/seismic-fault-labeling/pred/crop.json --meta ~/devl/seismic-fault-labeling/inlines/meta.json

On GEAP (gcp/setup_monitor.sh, then gcp/submit_monitor.py) the step uses
engine="bq": the input signals come from BigQuery (sql/drift/bq/), so the step
image needs no Spark or Java. Locally the default engine is Spark SQL on the lake.
"""
import argparse
import os

from kfp import compiler, dsl
from kfp.dsl import Artifact, Metrics, Output

IMAGE = os.environ.get("FAULT_EVAL_IMAGE", "python:3.12-slim")


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def monitor(repo_dir: str, lake: str, reference: str, survey: str, pred_dir: str, ref_pred_dir: str,
            thresholds: str, crop: str, meta: str, spot_check: int,
            engine: str, bq_table: str, project: str, location: str,
            metrics: Output[Metrics], report: Output[Artifact]) -> str:
    """Label-free drift check. Returns OK, WARN or ALERT."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    out = tempfile.mkdtemp()
    where = (["--engine", "bq", "--bq-table", bq_table, "--project", project, "--location", location]
             if engine == "bq" else ["--engine", "spark", "--root", lake])
    cmd = [sys.executable, os.path.join(repo_dir, "monitoring", "monitor_batch.py"), "check", *where,
           "--reference", reference, "--survey", survey,
           "--pred-dir", pred_dir, "--ref-pred-dir", ref_pred_dir, "--thresholds", thresholds,
           "--crop", crop, "--meta", meta, "--spot-check", str(spot_check), "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode:
        print(r.stderr[-3000:])
        raise RuntimeError("monitor_batch.py failed")
    rep = json.load(open(os.path.join(out, "monitor_report.json")))
    shutil.copy(os.path.join(out, "monitor_report.json"), report.path)

    for k, v in rep["values"].items():
        metrics.log_metric(k, round(float(v), 4))
    metrics.log_metric("status_level", {"OK": 0, "WARN": 1, "ALERT": 2}[rep["status"]])
    metrics.metadata.update({"survey": survey, "reference": reference, "status": rep["status"],
                             "engine": engine})
    report.metadata.update({"status": rep["status"], "spot_check": rep["spot_check"]})
    return rep["status"]


@dsl.pipeline(name="seismic-fault-monitor",
              description="Label-free drift check of a new batch against the reference survey.")
def monitor_pipeline(repo_dir: str, lake: str, survey: str, pred_dir: str, ref_pred_dir: str,
                     thresholds: str, crop: str, meta: str, reference: str = "F3", spot_check: int = 5,
                     engine: str = "spark", bq_table: str = "", project: str = "",
                     location: str = "us-central1"):
    m = monitor(repo_dir=repo_dir, lake=lake, reference=reference, survey=survey, pred_dir=pred_dir,
                ref_pred_dir=ref_pred_dir, thresholds=thresholds, crop=crop, meta=meta,
                spot_check=spot_check, engine=engine, bq_table=bq_table, project=project,
                location=location).set_display_name("drift check")
    m.set_cpu_limit("4").set_memory_limit("16G")     # two probability volumes in memory


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--out", default="monitor_pipeline.yaml")
    r = sub.add_parser("local")
    r.add_argument("--repo-dir", default=".")
    for a in ("--lake", "--survey", "--pred-dir", "--ref-pred-dir", "--thresholds", "--crop", "--meta"):
        r.add_argument(a, required=True)
    r.add_argument("--reference", default="F3")
    r.add_argument("--spot-check", type=int, default=5)
    r.add_argument("--pipeline-root", default=os.path.expanduser("~/kfp_local_outputs"))
    args = ap.parse_args()

    if args.cmd == "compile":
        compiler.Compiler().compile(monitor_pipeline, args.out)
        print(f"wrote {args.out}")
        return
    from kfp import local
    local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root=args.pipeline_root)
    ab = lambda p: os.path.abspath(os.path.expanduser(p))
    monitor_pipeline(repo_dir=ab(args.repo_dir), lake=ab(args.lake), survey=args.survey,
                     pred_dir=ab(args.pred_dir), ref_pred_dir=ab(args.ref_pred_dir),
                     thresholds=ab(args.thresholds), crop=ab(args.crop), meta=ab(args.meta),
                     reference=args.reference, spot_check=args.spot_check)
    print(f"\noutputs under {args.pipeline_root}")


if __name__ == "__main__":
    main()
