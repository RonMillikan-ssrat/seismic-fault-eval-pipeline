#!/usr/bin/env python3
"""Label-free drift signals, and how well each one tracks the true F1 drop.

Input-drift signals (from trace_stats in the lake, Spark SQL in sql/drift/):
  psi_rms_raw     PSI of trace RMS vs the reference survey, as recorded
  psi_rms_norm    the same after dividing by each survey's median RMS
  d_centroid_hz   change in median spectral centroid
  d_domfreq_hz    change in median dominant frequency

Output-drift signals (from the model's probability volume, NumPy):
  d_mean_prob     change in mean fault probability
  d_frac_pos      change in the fraction of voxels above 0.5
  d_entropy       change in mean binary entropy, -p ln p - (1-p) ln(1-p): how unsure
                  the model is on average (0 when every voxel is 0 or 1)
  d_frag          change in fragmentation: connected fault pieces (8-connected, per
                  inline section) per 1,000 fault pixels. Noise tends to break faults
                  into short pieces, so this rises even when the total fault area doesn't.

None of these needs labels, so all of them can run on new production data. The
study measures true F1 alongside them on perturbed copies of labeled F3, and the
analysis asks which signals rise when F1 falls (Spearman rank correlation of the
signal's size with the F1 drop) and which rise when it doesn't (false alarms).

  python monitoring/drift_signals.py --root ~/lake --study ~/f3_study --reference F3 \\
      --crop <crop.json> --meta <meta.json>
writes <study>/drift_table.csv and <study>/drift_vs_f1.png.
"""
import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

REPO = pathlib.Path(__file__).resolve().parent.parent
SQL_DIR = REPO / "sql" / "drift"
sys.path.insert(0, str(REPO / "baseline"))
from coherence import crop_ranges  # noqa: E402

SIGNALS = ["psi_rms_raw", "psi_rms_norm", "d_centroid_hz", "d_domfreq_hz",
           "d_mean_prob", "d_frac_pos", "d_entropy", "d_frag"]


# ------------------------------------------------------------------ input
def input_signals(root, reference, bounds, master="local[*]", driver_memory="8g"):
    """Run sql/drift/*.sql on trace_stats; return one row per survey (pandas)."""
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master(master).config("spark.driver.memory", driver_memory)
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    spark.read.parquet(f"{str(root).rstrip('/')}/trace_stats").createOrReplaceTempView("trace_stats")
    (il0, il1), (xl0, xl1) = bounds
    params = {"reference": reference, "il0": il0, "il1": il1, "xl0": xl0, "xl1": xl1}
    psi = spark.sql((SQL_DIR / "01_psi_rms.sql").read_text(), args=params).toPandas()
    spec = spark.sql((SQL_DIR / "02_spectral.sql").read_text(),
                     args={k: v for k, v in params.items() if k != "reference"}).toPandas()
    spark.stop()
    return _shifts(psi.merge(spec, on="survey", how="outer"), reference)


def _shifts(df, reference):
    ref = df[df.survey == reference]
    if ref.empty:
        raise SystemExit(f"reference survey {reference} has no trace_stats inside the crop")
    for col, src in [("d_centroid_hz", "centroid_med_hz"), ("d_domfreq_hz", "domfreq_med_hz")]:
        df[col] = df[src] - float(ref[src].iloc[0])
    return df


def input_signals_bq(table, reference, bounds, project=None, location=None):
    """Same signals as input_signals(), computed in BigQuery from a trace_stats table
    (sql/drift/bq/). No Spark or Java needed, so this is what runs in a pipeline step."""
    import re
    from google.cloud import bigquery
    if not re.match(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$", table):
        raise ValueError(f"not a valid project.dataset.table: {table!r}")
    client = bigquery.Client(project=project, location=location)
    (il0, il1), (xl0, xl1) = bounds
    ints = [bigquery.ScalarQueryParameter(k, "INT64", int(v))
            for k, v in (("il0", il0), ("il1", il1), ("xl0", xl0), ("xl1", xl1))]
    ref_p = [bigquery.ScalarQueryParameter("reference", "STRING", reference)]

    def run(name, params):
        text = (SQL_DIR / "bq" / name).read_text().replace("{table}", table)
        job = client.query(text, job_config=bigquery.QueryJobConfig(query_parameters=params))
        df = job.result().to_dataframe()
        print(f"BigQuery {name}: {len(df)} rows, {(job.total_bytes_processed or 0) / 1e6:.1f} MB scanned")
        return df

    psi = run("01_psi_rms.sql", ints + ref_p)
    spec = run("02_spectral.sql", ints)
    return _shifts(psi.merge(spec, on="survey", how="outer"), reference)


# ----------------------------------------------------------------- output
def output_signals(prob, threshold=0.5, step=1):
    """Signals from a probability volume (il, xl, t), using every `step`-th inline.
    All inlines by default: on F3 every 4th inline gave a predicted fraction about
    14% higher than the whole crop, enough to shift a threshold."""
    p = np.asarray(prob[::step], dtype=np.float32)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    entropy = -(pc * np.log(pc) + (1 - pc) * np.log1p(-pc))
    b = p > threshold
    n_px = int(b.sum())
    n_comp = sum(ndi.label(sec, structure=np.ones((3, 3), bool))[1] for sec in b)
    return dict(mean_prob=float(p.mean()), frac_pos=float(b.mean()),
                mean_entropy=float(entropy.mean()),
                frag_per_kpx=1000.0 * n_comp / max(n_px, 1),
                mean_comp_px=n_px / max(n_comp, 1), n_components=int(n_comp))


# --------------------------------------------------------------- analysis
def analyze(study, inputs, reference):
    study = pathlib.Path(study)
    summ = pd.read_csv(study / "summary.csv")
    rows = []
    for s in summ.survey:
        o = output_signals(np.load(study / s / "pred" / "prob_crop.npy", mmap_mode="r"))
        rows.append(dict(survey=s, **o))
    out = pd.DataFrame(rows)
    ref_o = out[out.survey == reference].iloc[0]
    out["d_mean_prob"] = out.mean_prob - ref_o.mean_prob
    out["d_frac_pos"] = out.frac_pos - ref_o.frac_pos
    out["d_entropy"] = out.mean_entropy - ref_o.mean_entropy
    out["d_frag"] = out.frag_per_kpx - ref_o.frag_per_kpx

    df = summ.merge(inputs, on="survey", how="left").merge(out, on="survey", how="left")
    f1_ref = float(df.loc[df.survey == reference, "f1"].iloc[0])
    df["f1_drop"] = f1_ref - df.f1
    if "f1_fixed" in df.columns:        # F1 at the reference survey's threshold (deployed setting)
        df["f1_fixed_drop"] = float(df.loc[df.survey == reference, "f1_fixed"].iloc[0]) - df.f1_fixed
    return df


def spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def report(df, study):
    fixed = "f1_fixed" in df.columns
    ycol = "f1_fixed" if fixed else "f1"
    head = f"{'survey':16s} {'F1':>6} {'drop':>6} " + (f"{'F1@ref t':>8} {'drop':>6} " if fixed else "")
    print("\n" + head + " ".join(f"{s:>13s}" for s in SIGNALS))
    for _, r in df.iterrows():
        mid = f"{r.f1_fixed:8.3f} {r.f1_fixed_drop:6.3f} " if fixed else ""
        print(f"{r.survey:16s} {r.f1:6.3f} {r.f1_drop:6.3f} " + mid + " ".join(f"{r[s]:13.4f}" for s in SIGNALS))

    # Signals are signed (a centroid can move up or down); the correlation uses their
    # size. With ~10 surveys these are rough rankings, not precise estimates.
    print(f"\nSpearman rank correlation of |signal| with the F1 drop ({len(df)} surveys):")
    print(f"  {'signal':14s} {'best-threshold F1':>18}" + (f" {'F1 at reference threshold':>26}" if fixed else ""))
    rho = {}
    for s in SIGNALS:
        x = np.abs(df[s].to_numpy(float))
        best = spearman(x, df.f1_drop.to_numpy(float))
        rho[s] = spearman(x, df.f1_fixed_drop.to_numpy(float)) if fixed else best
        print(f"  {s:14s} {best:+18.2f}" + (f" {rho[s]:+26.2f}" if fixed else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = df.kind.fillna("none").to_numpy()
    colors = {"none": "black", "noise": "tab:red", "lowpass": "tab:blue", "gain": "tab:green",
              "tgain": "tab:orange"}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for ax, s in zip(axes.ravel(), SIGNALS):
        for k in colors:
            m = kinds == k
            if m.any():
                ax.scatter(df.loc[m, s], df.loc[m, ycol], c=colors[k], label=k, s=40)
        for _, r in df.iterrows():
            ax.annotate(str(r.level) if r.kind != "none" else "ref", (r[s], r[ycol]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set(title=f"{s}  (rho {rho[s]:+.2f})", xlabel=s)
        ax.grid(alpha=0.3)
    ylab = "true F1 at the reference threshold" if fixed else "true F1 (best threshold)"
    axes[0, 0].set_ylabel(ylab); axes[1, 0].set_ylabel(ylab)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Label-free drift signals vs. true F1 on perturbed F3")
    fig.tight_layout()
    fig.savefig(pathlib.Path(study) / "drift_vs_f1.png", dpi=110)
    df.to_csv(pathlib.Path(study) / "drift_table.csv", index=False)
    print(f"\nwrote {study}/drift_table.csv and drift_vs_f1.png")
    return rho


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="lake root (needs trace_stats for every survey)")
    ap.add_argument("--study", required=True, help="folder from scripts/perturbation_study.sh")
    ap.add_argument("--reference", default="F3")
    ap.add_argument("--crop", required=True, help="crop.json: compare only traces inside this box")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--driver-memory", default="8g")
    args = ap.parse_args()

    il_r, xl_r, _ = crop_ranges(json.loads(pathlib.Path(args.crop).read_text()),
                                json.loads(pathlib.Path(args.meta).read_text()))
    bounds = ((il_r[0], il_r[1] - 1), (xl_r[0], xl_r[1] - 1))
    inputs = input_signals(args.root, args.reference, bounds, args.master, args.driver_memory)
    report(analyze(args.study, inputs, args.reference), args.study)


if __name__ == "__main__":
    main()
