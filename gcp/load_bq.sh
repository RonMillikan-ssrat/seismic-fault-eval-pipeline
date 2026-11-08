#!/usr/bin/env bash
# Upload the local Parquet lake to GCS, then load it into BigQuery.
#
# Steps
#   1. Bucket: created if missing, in the same location as the dataset
#      (a BigQuery load reads from GCS and needs compatible locations).
#   2. rsync the lake to gs://.../lake, skipping Spark's .crc checksum files and
#      _SUCCESS markers. --delete-unmatched-destination-objects removes objects that
#      no longer exist locally: when Spark rewrites a partition the part-file names
#      change, so without this the old files would stay and load as duplicates.
#   3. Dataset: created if missing.
#   4. bq load for each table:
#      --hive_partitioning_mode=STRINGS  turns the folder name survey=QC/ into a
#          STRING column "survey" (STRINGS rather than AUTO, so a survey named
#          "3" would never be inferred as an integer)
#      --parquet_enable_list_inference   loads Spark's array<float> as
#          REPEATED FLOAT64 instead of a nested list.element record
#      --clustering_fields               BigQuery has no partitioning on a STRING
#          column, so cluster by survey, il, xl: storage is sorted into blocks by
#          those columns and a filter on them reads fewer bytes
#      --replace                         overwrite the table on each run
#
# Settings (override with environment variables):
#   PROJECT  default: gcloud's current project
#   LAKE     default: /tmp/lake
#   DATASET  default: seismic
#   BUCKET   default: gs://<project>-seismic-lake
#   LOCATION default: the location of the existing "wx" dataset, else US
set -euo pipefail

PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
LAKE=${LAKE:-/tmp/lake}
DATASET=${DATASET:-seismic}
BUCKET=${BUCKET:-gs://${PROJECT}-seismic-lake}
if [[ -z "${LOCATION:-}" ]]; then
  LOCATION=$(bq show --format=json "${PROJECT}:wx" 2>/dev/null \
             | python3 -c 'import sys, json; print(json.load(sys.stdin)["location"])' 2>/dev/null \
             || echo US)
fi

echo "project=$PROJECT location=$LOCATION dataset=$DATASET bucket=$BUCKET lake=$LAKE"
for t in traces trace_stats surveys; do
  [[ -d "$LAKE/$t" ]] || { echo "missing $LAKE/$t: run the Spark steps first"; exit 1; }
done

echo "--- 1. bucket"
if ! gcloud storage buckets describe "$BUCKET" >/dev/null 2>&1; then
  gcloud storage buckets create "$BUCKET" --project="$PROJECT" \
      --location="$LOCATION" --uniform-bucket-level-access
fi

echo "--- 2. upload"
gcloud storage rsync --recursive "$LAKE" "$BUCKET/lake" \
    --exclude='.*\.crc$|.*_SUCCESS$' --delete-unmatched-destination-objects
gcloud storage du --summarize --readable-sizes "$BUCKET/lake"

echo "--- 3. dataset"
if ! bq --location="$LOCATION" show "$PROJECT:$DATASET" >/dev/null 2>&1; then
  bq --location="$LOCATION" mk --dataset --description "Seismic trace lakehouse" "$PROJECT:$DATASET"
fi

echo "--- 4. load"
load() {
  local table=$1 cluster=$2
  local extra=()
  [[ -n "$cluster" ]] && extra+=(--clustering_fields="$cluster")
  bq --location="$LOCATION" load --replace --source_format=PARQUET \
      --parquet_enable_list_inference \
      --hive_partitioning_mode=STRINGS \
      --hive_partitioning_source_uri_prefix="$BUCKET/lake/$table/" \
      "${extra[@]}" \
      "$PROJECT:$DATASET.$table" "$BUCKET/lake/$table/*"
}
load traces      survey,il,xl
load trace_stats survey,il,xl
load surveys     ""

echo "--- schema of traces"
bq show --schema --format=prettyjson "$PROJECT:$DATASET.traces"

echo "--- rows per survey"
bq query --use_legacy_sql=false --location="$LOCATION" \
  "SELECT 'traces' AS tbl, survey, COUNT(*) AS n FROM \`$PROJECT.$DATASET.traces\` GROUP BY survey
   UNION ALL
   SELECT 'trace_stats', survey, COUNT(*) FROM \`$PROJECT.$DATASET.trace_stats\` GROUP BY survey
   ORDER BY tbl, survey"
