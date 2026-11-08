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
| B3 | `evaluation/fault_metrics.py`: tolerance P/R/F1 (distance transform + max filter), PR curve, AP, best-F1 threshold, Dice/IoU with dilation, label-region precision, block and paired bootstrap, NMS thinning | Working; 34/34 known-answer checks (`tests/check_metrics.py`) |
| B3b | `evaluation/score_f3.py`: scores FaultSeg3D's `prob_crop.npy` on the labeled F3 inlines (all and corridor precision, AP, block-bootstrap CIs, held-out threshold) | Run on F3: matches `score_faults.py` at every threshold (see below). `--thin` added. |
| B2 | `baseline/coherence.py`: 1 − semblance from the Parquet lake (pyarrow filter pushdown), same output layout as FaultSeg3D | Working; 16/16 checks incl. gate (`tests/check_baseline.py`); F3 run pending |
| B4 | `evaluation/compare_models.py`: gate = paired block bootstrap of F1(model) − F1(baseline) on held-out inlines | Working on synthetic; F3 run pending |
| A5b | `sql/bq/01–08` + `gcp/qc_bq.py`: same QC checks in BigQuery SQL, with dry-run byte estimates | Working; 8/8 truth checks on BigQuery; header-only queries scan 1.1–2.1 MB of a ~210 MB table |

## Results so far: pretrained FaultSeg3D on F3

Scored against one interpreted fault (`fault_0`) picked on 41 inlines (every 10th, 100–500).
Tolerance 3 samples; corridor = within 40 samples of a pick. No thinning.

| Threshold | Recall | Precision (all) | Precision (corridor) | Predicted px |
|---|---|---|---|---|
| 0.50 | 0.740 | 0.103 | 0.409 | 242,675 |
| 0.70 | 0.713 | 0.113 | 0.437 | 190,535 |
| 0.90 | 0.653 | 0.129 | 0.489 | 118,613 |

| Summary | Value |
|---|---|
| AP, all / corridor | 0.111 / 0.505 |
| Best F1 (corridor), t = 0.97 | 0.567, 95% CI [0.457, 0.652] (block bootstrap, 11 blocks of 4 inlines) |
| Held-out F1 (threshold chosen on alternate blocks) | 0.564 |
| Dice, pixel / dilated by 3, at t = 0.97 | 0.251 / 0.482 |

Reading:
- **Precision (all) is mostly a labeling-coverage number.** Only one of F3's many faults was picked, so other real faults count as false positives.
- **The best threshold is 0.97** because FaultSeg3D predicts bands several samples wide; a high threshold keeps only the band center. Thinning (`--thin`) does that explicitly.
- **Pixel Dice is half the tolerance-aware Dice** for the same prediction, so the metric definition has to be stated with any number.
- The table matches the earlier `score_faults.py` in seismic-fault-labeling exactly, as an independent check of `fault_metrics.py` on real data.

### Tolerance and thinning (same 41 inlines, corridor precision)

| Tolerance (samples) | Best F1, unthinned | Best F1, thinned (NMS) |
|---|---|---|
| 1 | 0.411 | 0.362 |
| 2 | 0.501 | 0.481 |
| 3 | 0.567 | 0.559 |
| 5 | 0.651 | 0.651 |
| 8 | 0.715 | 0.717 |

- **F1 rises steadily with tolerance, with no single jump**, which fits a spread of 1–8 sample offsets between the model's fault centers and the hand-drawn picks rather than one constant shift. Separating pick error from model error needs inter-interpreter agreement (e.g. CRACKS' multiple annotators).
- **Thinning cut predicted pixels 4.7×** but did not change F1 at practical tolerances; thinned F1 catches up by tolerance 5, so NMS is not breaking ridges. It costs a little F1 at tolerance 1–2, where the wide band covers a pick better than its center line does.
- **AP is not comparable between unthinned and thinned** (0.111 vs 0.058, all-section): thinning caps recall near 0.61, so the curves span different recall ranges. Compare F1 at an operating point instead.
- One tolerance hides all of this, so `score_f3.py --tol-sweep` reports F1 against tolerance.

## Quick start (local, no cluster)

```bash
# PySpark 4 needs Java 17 or 21
sudo apt install openjdk-21-jre-headless
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-spark.txt

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
pip install -r requirements-eval.txt
python tests/check_metrics.py
python evaluation/fault_metrics.py --prob prob.npy --label label.npy --tol 2 --block 10
python tests/check_baseline.py

# F3: lake -> coherence baseline -> score both -> gate
# (F3 files come from seismic-fault-labeling: F3.sgy, inlines/, pred/)
L=~/devl/seismic-fault-labeling
python spark/segy_to_parquet.py $L/F3.sgy --survey F3 --out ~/lake --master 'local[*]' --driver-memory 16g
python baseline/coherence.py --lake ~/lake --survey F3 --crop $L/pred/crop.json --meta $L/inlines/meta.json --out ~/f3_coherence
python evaluation/score_f3.py --labels-dir $L/inlines --pred-dir ~/f3_coherence --tol 3 --tol-sweep 1 2 3 5 8
python evaluation/compare_models.py --labels-dir $L/inlines --pred-a $L/pred --pred-b ~/f3_coherence --tol 3 --tol-sweep 1 2 3 5 8 --out ~/f3_gate
```
