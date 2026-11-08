#!/usr/bin/env python3
"""KFP v2 pipeline: evaluate a fault model against a baseline and register it if it wins.

    faultseg_predictions ──┬──────────────► score (FaultSeg3D) ──► Metrics, HTML report
                           │                                         │
                           └► coherence_baseline ► score (coherence) │
                                       │                             │
                           both ─────► gate ──► passed? ──► register (Model + model card)

Each component calls the repo's tested scripts as a subprocess (score_f3.py,
coherence.py, compare_models.py), so the pipeline adds orchestration, artifact
passing and lineage, not new logic.

Artifacts and parameters
------------------------
  * Large existing inputs (the Parquet lake, the label folder) are passed as path
    strings. On GEAP these become gs:// locations read through the /gcs/ mount.
  * Everything a step PRODUCES is an output artifact (Dataset, Metrics, HTML,
    Model). KFP gives each one a path, records it in ML Metadata, and passes it to
    the next step, which is what makes lineage queryable.
  * The gate returns `passed` as an output PARAMETER, because dsl.If branches on
    parameters, not on artifact contents.

Why no ClassificationMetrics: KFP's ClassificationMetrics logs ROC curves and
confusion matrices, both of which need true negatives. With tolerance matching
there is no well-defined true-negative count, and with 0.14% fault pixels an ROC
curve would look excellent for almost any model. The PR curve goes in the HTML
report instead.

Run locally (no Docker, no cluster):
  python pipeline/fault_eval_pipeline.py local --repo-dir . --lake ~/lake \
      --labels-dir ~/devl/seismic-fault-labeling/inlines \
      --faultseg-pred-dir ~/devl/seismic-fault-labeling/pred \
      --weights-uri ~/devl/seismic-fault-labeling/models/fseg-70.hdf5
Compile for GEAP:
  python pipeline/fault_eval_pipeline.py compile --out fault_eval_pipeline.yaml
"""
import argparse
import os

from kfp import compiler, dsl
from kfp.dsl import HTML, Artifact, Dataset, Input, Metrics, Model, Output

# Container image for GEAP runs (built on Day 4 with the repo and its dependencies).
# The local SubprocessRunner ignores it and uses the current Python environment.
IMAGE = os.environ.get("FAULT_EVAL_IMAGE", "python:3.12-slim")


@dsl.component(base_image=IMAGE)
def faultseg_predictions(precomputed_dir: str, predictions: Output[Dataset]):
    """FaultSeg3D fault probabilities as a pipeline artifact.

    On GEAP this is where inference would run (predict_faultseg.py on a CPU or GPU
    worker). Locally it stages the existing output (prob_crop.npy + crop.json) so
    the evaluation steps get a real, lineage-tracked input without hours of CPU
    inference.
    """
    import json
    import os
    import shutil

    os.makedirs(predictions.path, exist_ok=True)
    for name in ("prob_crop.npy", "crop.json"):
        shutil.copy(os.path.join(precomputed_dir, name), os.path.join(predictions.path, name))
    crop = json.load(open(os.path.join(predictions.path, "crop.json")))
    predictions.metadata.update({"model": "FaultSeg3D (Wu et al. 2019), pretrained fseg-70",
                                 "source": precomputed_dir, "crop_lo": crop["lo"], "crop_hi": crop["hi"]})


@dsl.component(base_image=IMAGE)
def coherence_baseline(repo_dir: str, lake: str, survey: str, labels_dir: str,
                       crop_from: Input[Dataset], baseline: Output[Dataset]):
    """1 - semblance on the SAME crop as the candidate model, read from the Parquet lake."""
    import os
    import subprocess
    import sys

    cmd = [sys.executable, os.path.join(repo_dir, "baseline", "coherence.py"),
           "--lake", lake, "--survey", survey,
           "--crop", os.path.join(crop_from.path, "crop.json"),
           "--meta", os.path.join(labels_dir, "meta.json"),
           "--out", baseline.path]
    print(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    baseline.metadata.update({"model": "coherence: 1 - semblance, 3x3x9 window, no dip steering",
                              "lake": lake, "survey": survey})


@dsl.component(base_image=IMAGE)
def score(repo_dir: str, labels_dir: str, predictions: Input[Dataset], model_name: str,
          tol: float, tol_sweep: list, region_radius: float, n_boot: int,
          metrics: Output[Metrics], report: Output[HTML], scores: Output[Dataset]):
    """Score one model on the labeled inlines; log scalars and write an HTML report."""
    import base64
    import io
    import json
    import os
    import subprocess
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cmd = [sys.executable, os.path.join(repo_dir, "evaluation", "score_f3.py"),
           "--labels-dir", labels_dir, "--pred-dir", predictions.path,
           "--tol", str(tol), "--region-radius", str(region_radius), "--n-boot", str(n_boot),
           "--tol-sweep", *[str(t) for t in tol_sweep], "--out", scores.path]
    log = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    print(log)
    m = json.load(open(os.path.join(scores.path, "metrics.json")))

    for k in ("ap_all", "ap_corridor", "f1", "precision", "recall", "heldout_f1",
              "best_threshold", "dice_pixel", "dice_tol"):
        metrics.log_metric(k, round(float(m[k]), 4))
    metrics.log_metric("f1_ci_lo", round(m["f1_ci"][0], 4))
    metrics.log_metric("f1_ci_hi", round(m["f1_ci"][1], 4))
    for r in m["tol_curve"]:
        metrics.log_metric(f"f1_tol{r['tol']:g}", round(r["f1"], 4))
    metrics.metadata["model"] = model_name

    # Report: PR curves (all and corridor) and F1 against tolerance with CIs.
    pr = np.genfromtxt(os.path.join(scores.path, "pr_curve.csv"), delimiter=",", names=True)
    tc = m["tol_curve"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(pr["recall"], pr["precision_corridor"], label=f"corridor (AP {m['ap_corridor']:.3f})")
    ax[0].plot(pr["recall"], pr["precision_all"], "--", label=f"whole section (AP {m['ap_all']:.3f})")
    ax[0].set(xlabel=f"recall (tol {tol:g})", ylabel="precision", xlim=(0, 1), ylim=(0, 1),
              title=f"{model_name}: precision-recall")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    f1 = np.array([r["f1"] for r in tc])
    lo = np.array([r["f1_ci"][0] for r in tc]); hi = np.array([r["f1_ci"][1] for r in tc])
    ax[1].errorbar([r["tol"] for r in tc], f1, yerr=[f1 - lo, hi - f1], fmt="o-", capsize=3)
    ax[1].set(xlabel="tolerance (samples)", ylabel="best F1 (corridor), 95% CI", ylim=(0, 1),
              title="F1 against matching tolerance")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode()

    rows = "".join(f"<tr><td>{r['tol']:g}</td><td>{r['f1']:.3f}</td>"
                   f"<td>[{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}]</td>"
                   f"<td>{r['heldout_f1']:.3f}</td><td>{r['best_threshold']:.3f}</td></tr>" for r in tc)
    html = f"""<html><body style="font-family:sans-serif">
<h2>{model_name}: fault detection on {m['n_inlines']} labeled inlines</h2>
<p>Tolerance {tol:g} samples, corridor {region_radius:g} samples, block bootstrap
({m['n_blocks']} blocks of {m['block']} inlines).</p>
<img src="data:image/png;base64,{img}"/>
<table border="1" cellpadding="4" style="border-collapse:collapse">
<tr><th>tol</th><th>best F1</th><th>95% CI</th><th>held-out F1</th><th>threshold</th></tr>{rows}</table>
<pre>{log}</pre></body></html>"""
    with open(report.path, "w") as f:
        f.write(html)


@dsl.component(base_image=IMAGE)
def gate(repo_dir: str, labels_dir: str, candidate: Input[Dataset], reference: Input[Dataset],
         tol: float, tol_sweep: list, region_radius: float, min_diff: float, n_boot: int,
         metrics: Output[Metrics], decision: Output[Artifact]) -> bool:
    """Paired block bootstrap of F1(candidate) - F1(reference) on held-out inlines.

    Returns `passed` as the step's single output parameter (KFP names it "Output").
    A single return value rather than a NamedTuple: kfp.local (2.18) resolves a
    dsl.If condition on a task output only when the channel is "<task>-Output",
    and the CI bounds are in the Metrics artifact anyway.
    """
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    out = tempfile.mkdtemp()
    cmd = [sys.executable, os.path.join(repo_dir, "evaluation", "compare_models.py"),
           "--labels-dir", labels_dir, "--pred-a", candidate.path, "--pred-b", reference.path,
           "--tol", str(tol), "--tol-sweep", *[str(t) for t in tol_sweep],
           "--region-radius", str(region_radius), "--min-diff", str(min_diff),
           "--n-boot", str(n_boot), "--out", out]
    print(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    g = json.load(open(os.path.join(out, "gate.json")))
    shutil.copy(os.path.join(out, "gate.json"), decision.path)

    metrics.log_metric("f1_diff", round(g["diff"], 4))
    metrics.log_metric("f1_diff_ci_lo", round(g["diff_ci"][0], 4))
    metrics.log_metric("f1_diff_ci_hi", round(g["diff_ci"][1], 4))
    metrics.log_metric("passed", int(g["passed"]))
    decision.metadata.update({"passed": g["passed"], "tol": g["tol"], "min_diff": g["min_diff"]})
    return bool(g["passed"])


@dsl.component(base_image=IMAGE)
def register(weights_uri: str, candidate: Input[Dataset], candidate_scores: Input[Dataset],
             decision: Input[Artifact], model: Output[Model]):
    """Record the approved model version: a model card with its metrics and gate result.

    Locally this is a Model artifact in ML Metadata. On GEAP (Day 4) this step
    uploads to the Model Registry instead, with the same card as metadata.
    """
    import json
    import os

    m = json.load(open(os.path.join(candidate_scores.path, "metrics.json")))
    g = json.load(open(decision.path))
    card = {"name": "faultseg3d-f3", "weights_uri": weights_uri,
            "evaluated_on": f"F3, {m['n_inlines']} labeled inlines ({m['inlines'][0]}-{m['inlines'][-1]})",
            "metrics": {k: m[k] for k in ("f1", "f1_ci", "precision", "recall", "heldout_f1",
                                          "ap_corridor", "best_threshold", "tol", "region_radius")},
            "gate": {k: g[k] for k in ("passed", "reference", "tol", "diff", "diff_ci", "min_diff")},
            "predictions_source": candidate.metadata.get("source")}
    os.makedirs(model.path, exist_ok=True)
    with open(os.path.join(model.path, "model_card.json"), "w") as f:
        json.dump(card, f, indent=1)
    model.metadata.update({"weights_uri": weights_uri, "f1": m["f1"], "gate_diff_lo": g["diff_ci"][0],
                           "threshold": m["best_threshold"]})


@dsl.pipeline(name="seismic-fault-eval",
              description="Score FaultSeg3D and a coherence baseline on F3; gate; register.")
def fault_eval_pipeline(repo_dir: str, lake: str, labels_dir: str, faultseg_pred_dir: str,
                        weights_uri: str, survey: str = "F3", tol: float = 3.0,
                        tol_sweep: list = [1.0, 2.0, 3.0, 5.0, 8.0], region_radius: float = 40.0,
                        min_diff: float = 0.0, n_boot: int = 2000):
    pred = faultseg_predictions(precomputed_dir=faultseg_pred_dir).set_display_name("FaultSeg3D predictions")
    base = coherence_baseline(repo_dir=repo_dir, lake=lake, survey=survey, labels_dir=labels_dir,
                              crop_from=pred.outputs["predictions"]).set_display_name("coherence baseline")
    s_model = score(repo_dir=repo_dir, labels_dir=labels_dir, predictions=pred.outputs["predictions"],
                    model_name="FaultSeg3D", tol=tol, tol_sweep=tol_sweep,
                    region_radius=region_radius, n_boot=n_boot).set_display_name("score FaultSeg3D")
    score(repo_dir=repo_dir, labels_dir=labels_dir, predictions=base.outputs["baseline"],
          model_name="coherence", tol=tol, tol_sweep=tol_sweep,
          region_radius=region_radius, n_boot=n_boot).set_display_name("score coherence")
    g = gate(repo_dir=repo_dir, labels_dir=labels_dir, candidate=pred.outputs["predictions"],
             reference=base.outputs["baseline"], tol=tol, tol_sweep=tol_sweep,
             region_radius=region_radius, min_diff=min_diff, n_boot=n_boot).set_display_name("gate")
    with dsl.If(g.outputs["Output"] == True):  # noqa: E712 (KFP builds the condition from ==, not "is")
        register(weights_uri=weights_uri, candidate=pred.outputs["predictions"],
                 candidate_scores=s_model.outputs["scores"],
                 decision=g.outputs["decision"]).set_display_name("register model")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="write the pipeline spec (YAML) for GEAP")
    c.add_argument("--out", default="fault_eval_pipeline.yaml")
    r = sub.add_parser("local", help="run every step locally with kfp.local")
    r.add_argument("--repo-dir", default=".")
    r.add_argument("--lake", required=True)
    r.add_argument("--labels-dir", required=True)
    r.add_argument("--faultseg-pred-dir", required=True)
    r.add_argument("--weights-uri", required=True)
    r.add_argument("--survey", default="F3")
    r.add_argument("--min-diff", type=float, default=0.0)
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--pipeline-root", default=os.path.expanduser("~/kfp_local_outputs"))
    args = ap.parse_args()

    if args.cmd == "compile":
        compiler.Compiler().compile(fault_eval_pipeline, args.out)
        print(f"wrote {args.out}")
        return

    from kfp import local
    # SubprocessRunner runs each component as a Python subprocess. use_venv=False
    # reuses THIS environment (which already has numpy, scipy, pyarrow, matplotlib)
    # instead of building a fresh venv per step.
    local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root=args.pipeline_root)
    ab = lambda p: os.path.abspath(os.path.expanduser(p))
    fault_eval_pipeline(repo_dir=ab(args.repo_dir), lake=ab(args.lake), labels_dir=ab(args.labels_dir),
                        faultseg_pred_dir=ab(args.faultseg_pred_dir), weights_uri=ab(args.weights_uri),
                        survey=args.survey, min_diff=args.min_diff, n_boot=args.n_boot)
    print(f"\noutputs under {args.pipeline_root}")


if __name__ == "__main__":
    main()
