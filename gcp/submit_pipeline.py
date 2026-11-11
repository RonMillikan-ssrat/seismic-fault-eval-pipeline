#!/usr/bin/env python3
"""Compile the pipeline against the step image and submit it to GEAP (Vertex AI Pipelines).

What happens on submit
----------------------
  * The pipeline is compiled to a YAML spec with every component's base_image set
    to --image (the Dockerfile image pushed by gcp/setup_vertex.sh).
  * PipelineJob uploads the spec and starts a run. Each step runs as its own
    container on a VM sized by its CPU/memory requests, as --service-account.
  * pipeline_root (gs://<bucket>/pipeline_root) is where every output artifact
    lands; ML Metadata records which step produced and consumed each one.
  * Inputs are passed as /gcs/<bucket>/... paths. On Vertex, Cloud Storage is
    mounted at /gcs/ inside step containers, so the scripts read them as files.
  * Caching: a step whose inputs and code are unchanged reuses its earlier outputs
    (--no-cache turns that off).

Run gcp/setup_vertex.sh first. Needs Application Default Credentials
(gcloud auth application-default login) and google-cloud-aiplatform.
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
    ap.add_argument("--data-prefix", default="f3", help="folder in the bucket written by setup_vertex.sh")
    ap.add_argument("--image", required=True, help="e.g. us-central1-docker.pkg.dev/<project>/seismic/fault-eval:v1")
    ap.add_argument("--service-account", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-diff", type=float, default=0.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--wait", action="store_true", help="block until the run finishes")
    args = ap.parse_args()

    # The image is read when the pipeline module is imported, so set it first.
    os.environ["FAULT_EVAL_IMAGE"] = args.image
    spec = importlib.util.spec_from_file_location("fep", REPO / "pipeline" / "fault_eval_pipeline.py")
    fep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fep)

    from google.cloud import aiplatform
    from kfp import compiler

    yaml_path = os.path.join(tempfile.mkdtemp(), "fault_eval_pipeline.yaml")
    compiler.Compiler().compile(fep.fault_eval_pipeline, yaml_path)

    bucket = args.bucket.rstrip("/")
    mount = lambda p: f"/gcs/{bucket.removeprefix('gs://')}/{args.data_prefix}/{p}"
    params = dict(repo_dir="/app", lake=mount("lake"), labels_dir=mount("inlines"),
                  faultseg_pred_dir=mount("pred"), weights_uri=mount("models/fseg-70.hdf5"),
                  survey="F3", n_boot=args.n_boot, min_diff=args.min_diff,
                  project=args.project, location=args.region, use_vertex=True,
                  serving_image=args.image)

    aiplatform.init(project=args.project, location=args.region, staging_bucket=bucket)
    job = aiplatform.PipelineJob(display_name="seismic-fault-eval", template_path=yaml_path,
                                 pipeline_root=f"{bucket}/pipeline_root", parameter_values=params,
                                 enable_caching=not args.no_cache)
    job.submit(service_account=args.service_account)
    print(f"submitted {job.resource_name}")
    print(f"console: https://console.cloud.google.com/vertex-ai/locations/{args.region}/pipelines/runs/"
          f"{job.name}?project={args.project}")
    if args.wait:
        job.wait()
        print(f"final state: {job.state.name}")


if __name__ == "__main__":
    main()
