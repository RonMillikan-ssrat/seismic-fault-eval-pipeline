#!/usr/bin/env bash
# One-time setup for running the pipeline on GEAP (Vertex AI Pipelines).
# Safe to re-run: each step checks before creating.
#
#   1. Enable the Vertex AI API.
#   2. Artifact Registry Docker repo "seismic" (us-central1).
#   3. Service account seismic-pipeline, with only what the pipeline needs:
#        roles/aiplatform.user          run pipeline steps, upload to Model Registry
#        roles/logging.logWriter        write step logs
#        roles/storage.objectAdmin      ON THE BUCKET ONLY: read F3 data, write pipeline_root
#        roles/artifactregistry.reader  ON THE REPO ONLY: pull the step image
#   4. Build the step image with Cloud Build (no local Docker) and push it.
#   5. Upload the F3 inputs to gs://<bucket>/f3/ (lake, labels, FaultSeg3D output, weights).
#      A separate prefix from lake/, so load_bq.sh's rsync --delete never touches it.
#
# Settings (environment variables):
#   PROJECT  default: gcloud's current project      REGION   default: us-central1
#   BUCKET   default: gs://<project>-seismic-lake   TAG      default: v1
#   LAKE     default: ~/lake                         LABELS   default: ~/devl/seismic-fault-labeling
#   STEPS    default: "api repo sa image data" (run a subset, e.g. STEPS="image")
set -euo pipefail

PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
REGION=${REGION:-us-central1}
BUCKET=${BUCKET:-gs://${PROJECT}-seismic-lake}
TAG=${TAG:-v1}
LAKE=${LAKE:-$HOME/lake}
LABELS=${LABELS:-$HOME/devl/seismic-fault-labeling}
STEPS=${STEPS:-"api repo sa image data"}
REPO=seismic
SA_NAME=seismic-pipeline
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/fault-eval:${TAG}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
has() { [[ " $STEPS " == *" $1 "* ]]; }

echo "project=$PROJECT region=$REGION bucket=$BUCKET image=$IMAGE sa=$SA"

if has api; then
  echo "--- 1. Vertex AI API"
  gcloud services enable aiplatform.googleapis.com --project="$PROJECT"
fi

if has repo; then
  echo "--- 2. Artifact Registry repo"
  gcloud artifacts repositories describe "$REPO" --location="$REGION" --project="$PROJECT" >/dev/null 2>&1 \
    || gcloud artifacts repositories create "$REPO" --repository-format=docker \
         --location="$REGION" --project="$PROJECT" --description="seismic-fault-eval pipeline images"
fi

if has sa; then
  echo "--- 3. service account and roles"
  gcloud iam service-accounts describe "$SA" --project="$PROJECT" >/dev/null 2>&1 \
    || gcloud iam service-accounts create "$SA_NAME" --project="$PROJECT" \
         --display-name="seismic-fault-eval pipeline runner"
  for role in roles/aiplatform.user roles/logging.logWriter; do
    gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
        --role="$role" --condition=None --quiet >/dev/null
  done
  gcloud storage buckets add-iam-policy-binding "$BUCKET" --member="serviceAccount:$SA" \
      --role=roles/storage.objectAdmin --quiet >/dev/null
  gcloud artifacts repositories add-iam-policy-binding "$REPO" --location="$REGION" \
      --project="$PROJECT" --member="serviceAccount:$SA" --role=roles/artifactregistry.reader --quiet >/dev/null
  echo "roles granted to $SA"
fi

if has image; then
  echo "--- 4. build and push $IMAGE (Cloud Build)"
  gcloud builds submit "$HERE" --tag="$IMAGE" --region="$REGION" --project="$PROJECT"
fi

if has data; then
  echo "--- 5. upload F3 inputs to $BUCKET/f3/"
  [[ -d "$LAKE/traces/survey=F3" ]] || { echo "no F3 in $LAKE: run spark/segy_to_parquet.py first"; exit 1; }
  gcloud storage rsync --recursive "$LAKE/traces/survey=F3" "$BUCKET/f3/lake/traces/survey=F3" \
      --exclude='.*\.crc$|.*_SUCCESS$' --delete-unmatched-destination-objects
  gcloud storage cp "$LABELS/inlines/labels_sparse.npy" "$LABELS/inlines/meta.json" "$BUCKET/f3/inlines/"
  gcloud storage cp "$LABELS/pred/prob_crop.npy" "$LABELS/pred/crop.json" "$BUCKET/f3/pred/"
  gcloud storage cp "$LABELS/models/fseg-70.hdf5" "$BUCKET/f3/models/"
  gcloud storage du --summarize --readable-sizes "$BUCKET/f3"
fi

echo "done. Submit with: python gcp/submit_pipeline.py --image $IMAGE --service-account $SA"
