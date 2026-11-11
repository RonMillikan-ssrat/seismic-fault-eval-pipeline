#!/usr/bin/env python3
"""Submit the monitoring pipeline to GEAP (Vertex AI Pipelines), once or on a schedule.

One run:
  python gcp/submit_monitor.py --image <image> --survey F3-noise-0.5 --wait
Scheduled (weekly, at most 3 runs, then the schedule completes on its own):
  python gcp/submit_monitor.py --image <image> --survey F3-noise-0.5 \
      --schedule "TZ=America/Chicago 0 6 * * 1" --max-runs 3
List, pause or delete schedules:
  python gcp/submit_monitor.py --list-schedules
  python gcp/submit_monitor.py --delete-schedule <schedule resource name>

What a schedule is: Vertex stores the compiled pipeline and its parameter values and
starts a new PipelineJob at each cron time. --max-runs caps the total so a forgotten
schedule cannot keep spending; each run costs about $0.03 plus a few minutes of one VM.

In this project the scheduled run re-checks the same batch each time, which shows the
mechanism. In production the survey parameter would point at whatever batch arrived.

Inputs on GCS (gcp/setup_monitor.sh): predictions under f3/study/<survey>/pred, the
crop and labels metadata under f3/, trace_stats in BigQuery. Thresholds come from the
image (/app/monitoring/thresholds_f3.json).
"""
import argparse
import importlib.util
import os
import pathlib
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", default="wx-modernization-0")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", default="gs://wx-modernization-0-seismic-lake")
    ap.add_argument("--dataset", default="seismic")
    ap.add_argument("--image")
    ap.add_argument("--service-account",
                    default="seismic-pipeline@wx-modernization-0.iam.gserviceaccount.com")
    ap.add_argument("--reference", default="F3")
    ap.add_argument("--survey", help="batch to check, e.g. F3-noise-0.5")
    ap.add_argument("--schedule", help='cron, e.g. "TZ=America/Chicago 0 6 * * 1"')
    ap.add_argument("--max-runs", type=int, default=3, help="total runs a schedule may start")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--list-schedules", action="store_true")
    ap.add_argument("--delete-schedule", help="schedule resource name to delete")
    args = ap.parse_args()

    from google.cloud import aiplatform
    bucket = args.bucket.rstrip("/")
    aiplatform.init(project=args.project, location=args.region, staging_bucket=bucket)

    if args.list_schedules:
        for s in aiplatform.PipelineJobSchedule.list():
            print(f"{s.display_name}  state={s.state.name}  cron={s.cron!r}  max_runs={s.max_run_count}  {s.resource_name}")
        return
    if args.delete_schedule:
        aiplatform.PipelineJobSchedule.get(args.delete_schedule).delete()
        print(f"deleted {args.delete_schedule}")
        return
    if not (args.image and args.survey):
        ap.error("--image and --survey are required to submit")

    os.environ["FAULT_EVAL_IMAGE"] = args.image       # read when the pipeline module is imported
    spec = importlib.util.spec_from_file_location("mp", REPO / "pipeline" / "monitor_pipeline.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    from kfp import compiler
    yaml_path = os.path.join(tempfile.mkdtemp(), "monitor_pipeline.yaml")
    compiler.Compiler().compile(mp.monitor_pipeline, yaml_path)

    gcs = lambda p: f"/gcs/{bucket.removeprefix('gs://')}/f3/{p}"
    params = dict(repo_dir="/app", lake="", reference=args.reference, survey=args.survey,
                  pred_dir=gcs(f"study/{args.survey}/pred"), ref_pred_dir=gcs(f"study/{args.reference}/pred"),
                  thresholds="/app/monitoring/thresholds_f3.json",
                  crop=gcs("pred/crop.json"), meta=gcs("inlines/meta.json"), spot_check=5,
                  engine="bq", bq_table=f"{args.project}.{args.dataset}.study_trace_stats",
                  project=args.project, location=args.region)
    job = aiplatform.PipelineJob(display_name=f"seismic-fault-monitor-{args.survey.lower()}",
                                 template_path=yaml_path, pipeline_root=f"{bucket}/pipeline_root",
                                 parameter_values=params, enable_caching=False)

    if args.schedule:
        sched = job.create_schedule(cron=args.schedule, display_name=f"monitor {args.survey}",
                                    max_run_count=args.max_runs, max_concurrent_run_count=1,
                                    service_account=args.service_account)
        print(f"created schedule {sched.resource_name}")
        print(f"cron {args.schedule!r}, at most {args.max_runs} runs; delete with "
              f"--delete-schedule {sched.resource_name}")
        return
    job.submit(service_account=args.service_account)
    print(f"submitted {job.resource_name}")
    print(f"console: https://console.cloud.google.com/vertex-ai/locations/{args.region}/pipelines/runs/"
          f"{job.name}?project={args.project}")
    if args.wait:
        job.wait()
        print(f"final state: {job.state.name}")


if __name__ == "__main__":
    main()
