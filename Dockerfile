# Image for the KFP pipeline steps on GEAP (Vertex AI Pipelines).
# Contains the repo at /app and the evaluation dependencies. No Spark or Java:
# the Parquet lake is built beforehand (locally, or by Dataproc Serverless later).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements-image.txt .
RUN pip install --no-cache-dir -r requirements-image.txt

# Copied after the pip install, so code changes rebuild only this layer.
COPY . /app
