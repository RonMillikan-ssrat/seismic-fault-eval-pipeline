#!/usr/bin/env bash
# Prepare GEAP for the monitoring pipeline (run gcp/setup_vertex.sh once before this).
# Safe to re-run.
#
#   roles   the pipeline's service account may run BigQuery jobs (project level) and
#           read the seismic dataset (dataset level, granted with SQL GRANT)
#   stats   upload the lake's trace_stats (small: one row per trace, no samples) and
#           load it into BigQuery as <dataset>.study_trace_stats
#   preds   upload FaultSeg3D predictions for the reference and the batches to check
#   image   rebuild the step image (now includes the BigQuery client and the
#           thresholds file) under a new tag
#
# Settings: PROJECT, REGION (us-central1), BUCKET (gs://<project>-seismic-lake),
#   DATASET (seismic), LAKE (~/lake), STUDY (~/f3_study), TAG (v2),
#   SURVEYS ("F3 F3-gain-4 F3-noise-0.5 F3-lowpass-40"), STEPS ("roles stats preds image")
set -euo pipefail

PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
REGION=${REGION:-us-central1}
BUCKET=${BUCKET:-gs://${PROJECT}-seismic-lake}
DATASET=${DATASET:-seismic}
LAKE=${LAKE:-$HOME/lake}
STUDY=${STUDY:-$HOME/f3_study}
TAG=${TAG:-v2}
SURVEYS=${SURVEYS:-"F3 F3-gain-4 F3-noise-0.5 F3-lowpass-40"}
STEPS=${STEPS:-"roles stats preds image"}
SA="seismic-pipeline@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/seismic/fault-eval:${TAG}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
has() { [[ " $STEPS " == *" $1 "* ]]; }
echo "project=$PROJECT bucket=$BUCKET dataset=$DATASET image=$IMAGE"

if has roles; then
  echo "--- roles for $SA"
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
      --role=roles/bigquery.jobUser --condition=None --quiet >/dev/null
  bq query --use_legacy_sql=false --location="$REGION" \
      "GRANT \`roles/bigquery.dataViewer\` ON SCHEMA \`$PROJECT.$DATASET\` TO \"serviceAccount:$SA\"" >/dev/null
  echo "granted bigquery.jobUser (project) and bigquery.dataViewer (dataset $DATASET)"
fi

if has stats; then
  echo "--- trace_stats to BigQuery"
  [[ -d "$LAKE/trace_stats" ]] || { echo "no $LAKE/trace_stats: run the study's attrs step first"; exit 1; }
  gcloud storage rsync --recursive "$LAKE/trace_stats" "$BUCKET/f3/trace_stats" \
      --exclude='.*\.crc$|.*_SUCCESS$' --delete-unmatched-destination-objects
  bq --location="$REGION" load --replace --source_format=PARQUET \
      --hive_partitioning_mode=STRINGS --hive_partitioning_source_uri_prefix="$BUCKET/f3/trace_stats/" \
      --clustering_fields=survey,il,xl \
      "$PROJECT:$DATASET.study_trace_stats" "$BUCKET/f3/trace_stats/*"
  bq query --use_legacy_sql=false --location="$REGION" \
      "SELECT survey, COUNT(*) AS traces FROM \`$PROJECT.$DATASET.study_trace_stats\` GROUP BY survey ORDER BY survey"
fi

if has preds; then
  echo "--- predictions"
  for s in $SURVEYS; do
    gcloud storage cp "$STUDY/$s/pred/prob_crop.npy" "$STUDY/$s/pred/crop.json" "$BUCKET/f3/study/$s/pred/"
  done
  gcloud storage du --summarize --readable-sizes "$BUCKET/f3/study"
fi

if has image; then
  echo "--- build $IMAGE"
  gcloud builds submit "$HERE" --tag="$IMAGE" --region="$REGION" --project="$PROJECT"
fi
echo "done. Submit with: python gcp/submit_monitor.py --image $IMAGE --survey F3-noise-0.5"
