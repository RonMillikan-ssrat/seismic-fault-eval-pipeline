#!/usr/bin/env bash
# Set up a second machine for the F3 perturbation study, from the copies in GCS
# (uploaded by gcp/setup_vertex.sh) and GitHub.
#
#   1. Clone seismic-fault-labeling (for predict_faultseg.py) if it isn't there.
#   2. Download labels, the original FaultSeg3D output and the weights into it.
#   3. Download the F3 traces into the lake.
#   4. Write F3's row in the lake's surveys table. Only the traces were uploaded,
#      and spark/perturb.py reads the sample interval from surveys. The row is
#      written the same way spark/segy_to_parquet.py writes it, with the values
#      that script printed for F3.sgy (text_header is left empty).
#   5. Create a venv in the labeling repo with TensorFlow for inference.
#
# Safe to re-run: rsync skips files already present; each step checks first.
# Run from the eval-pipeline venv (needs pyspark). Settings:
#   BUCKET  default gs://<gcloud project>-seismic-lake
#   L       default ~/devl/seismic-fault-labeling      LAKE  default ~/lake
#   STEPS   default "repo data lake survey tf"
set -euo pipefail

PROJECT=$(gcloud config get-value project 2>/dev/null)
BUCKET=${BUCKET:-gs://${PROJECT}-seismic-lake}
L=${L:-$HOME/devl/seismic-fault-labeling}
LAKE=${LAKE:-$HOME/lake}
STEPS=${STEPS:-"repo data lake survey tf"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
echo "bucket=$BUCKET labeling repo=$L lake=$LAKE"
df -h "$HOME" | tail -1

if has repo; then
  echo "--- 1. labeling repo"
  [[ -f "$L/predict_faultseg.py" ]] || git clone https://github.com/RonMillikan-ssrat/seismic-fault-labeling.git "$L"
  git -C "$L" pull --ff-only || echo "(pull skipped: local changes or no network)"
fi

if has data; then
  echo "--- 2. labels, original predictions, weights"
  mkdir -p "$L/inlines" "$L/pred" "$L/models"
  gcloud storage rsync "$BUCKET/f3/inlines" "$L/inlines"
  gcloud storage rsync "$BUCKET/f3/pred"    "$L/pred"
  gcloud storage rsync "$BUCKET/f3/models"  "$L/models"
  ls -la "$L/inlines/labels_sparse.npy" "$L/inlines/meta.json" "$L/pred/prob_crop.npy" \
         "$L/pred/crop.json" "$L/models/fseg-70.hdf5"
fi

if has lake; then
  echo "--- 3. F3 traces"
  mkdir -p "$LAKE/traces/survey=F3"
  gcloud storage rsync --recursive "$BUCKET/f3/lake/traces/survey=F3" "$LAKE/traces/survey=F3"
  du -sh "$LAKE/traces/survey=F3"
fi

if has survey; then
  echo "--- 4. surveys row for F3"
  python - "$LAKE" <<'EOF'
import os, sys
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
from pyspark.sql import SparkSession, functions as F
lake = sys.argv[1].rstrip("/")
spark = (SparkSession.builder.master("local[2]")
         .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
# Values printed by segy_to_parquet.py for F3.sgy on 2026-11-08.
meta = dict(path="F3.sgy (row rebuilt by fetch_f3_from_gcs.sh)", format_code=1, ns=463, dt_ms=4.0,
            revision=256, n_traces=600515, file_bytes=1256280980, text_header="",
            byteorder="big", survey="F3", il_byte=189, xl_byte=193)
spark.createDataFrame([meta]).write.mode("overwrite").partitionBy("survey").parquet(f"{lake}/surveys")
t = spark.read.parquet(f"{lake}/traces").where(F.col("survey") == "F3")
n, ns = t.count(), t.select(F.size("samples").alias("n")).first()["n"]
print(f"F3 in the lake: {n} traces (expect 600515), {ns} samples per trace (expect 463)")
spark.stop()
EOF
fi

if has tf; then
  echo "--- 5. TensorFlow venv in the labeling repo"
  [[ -x "$L/venv/bin/python" ]] || python3 -m venv "$L/venv"
  "$L/venv/bin/pip" install -q --upgrade pip
  "$L/venv/bin/pip" install -q --pre tensorflow numpy scipy matplotlib pyarrow
  "$L/venv/bin/python" -c "import tensorflow as tf, pyarrow, scipy, matplotlib; print('TF', tf.__version__)" 2>&1 | tail -1
fi
echo "done"
