# seismic-fault-eval-pipeline

Evaluation and monitoring pipeline for seismic fault-detection models, on GCP GEAP (Gemini Enterprise Agent Platform, formerly Vertex AI) Pipelines with Kubeflow (KFP v2).

- **Part A, seismic trace lakehouse (PySpark + SQL):** SEG-Y to Parquet with one row per trace, per-trace attributes, geometry/amplitude QC in SQL, drift statistics in BigQuery.
- **Part B, fault-model evaluation (KFP):** pretrained FaultSeg3D inference, a coherence baseline, tolerance-based metrics with bootstrap confidence intervals, a gated registration step and label-free drift monitoring.

Work in progress. The full README comes at the end of the build.

## Status

| Part | Component | State |
|---|---|---|
| A1 | `spark/segy_to_parquet.py`: byte-range SEG-Y reader (`mapInPandas`), IBM/IEEE floats | Working; tested on synthetic SEG-Y |
| A2 | `spark/trace_attributes.py`: per-trace attributes; Python UDF vs Arrow UDF vs pandas UDF timing | Working; tested on synthetic SEG-Y |
| A5 | `sql/01–08` + `spark/qc_sql.py`: missing positions, duplicates, dead traces, amplitude jumps, rolling RMS, crossline islands, percentiles, survey summary | Working; 8/8 checks against synthetic truth (`tests/check_qc.py`) |
| A4 | `gcp/load_bq.sh`: lake → GCS → BigQuery (Hive partitioning, list inference, clustering by survey, il, xl) | Working on GCP (`us-central1`): 57,051 traces loaded, `samples` as REPEATED FLOAT64 |
| B3 | `evaluation/fault_metrics.py`: tolerance P/R/F1 (distance transform + max filter), PR curve, AP, best-F1 threshold, Dice/IoU with dilation, label-region precision, block and paired bootstrap | Working; 26/26 known-answer checks (`tests/check_metrics.py`) |
| A5b | `sql/bq/01–08` + `gcp/qc_bq.py`: same QC checks in BigQuery SQL, with dry-run byte estimates | Working; 8/8 truth checks on BigQuery; header-only queries scan 1.1–2.1 MB of a ~210 MB table |

## Quick start (local, no cluster)

```bash
# PySpark 4 needs Java 17 or 21
sudo apt install openjdk-21-jre-headless
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-spark.txt
export PYSPARK_PYTHON=$(which python)

# Self-test on a synthetic SEG-Y with known contents
python tests/make_synthetic_segy.py /tmp/syn.sgy
python spark/segy_to_parquet.py /tmp/syn.sgy --survey SYN --out /tmp/lake --master 'local[*]'
python tests/check_roundtrip.py /tmp/syn.sgy.truth.npz /tmp/lake/traces/survey=SYN
python spark/trace_attributes.py --root /tmp/lake --survey SYN --master 'local[*]'

# QC SQL on a survey with known problems: gaps, dead traces, duplicates, a 3x gain patch
python tests/make_synthetic_segy.py /tmp/qc.sgy --il 100,299 --xl 300,599 --ns 462 \
    --dups 25 --gain-patch 150,169,400,439,3
python spark/segy_to_parquet.py /tmp/qc.sgy --survey QC --out /tmp/lake --master 'local[*]'
python spark/trace_attributes.py --root /tmp/lake --survey QC --master 'local[*]' --variants pandas --repeat 1
python tests/check_qc.py /tmp/qc.sgy.truth.npz --root /tmp/lake --survey QC
python spark/qc_sql.py --root /tmp/lake --survey QC --master 'local[*]' --explain 08

# BigQuery (needs gcloud auth and Application Default Credentials)
pip install -r requirements-gcp.txt
bash gcp/load_bq.sh                       # LAKE=/tmp/lake by default
python gcp/qc_bq.py --survey QC
python tests/check_qc.py /tmp/qc.sgy.truth.npz --engine bq --survey QC

# Fault metrics (no Spark needed)
python tests/check_metrics.py
python evaluation/fault_metrics.py --prob prob.npy --label label.npy --tol 2 --block 10
```
