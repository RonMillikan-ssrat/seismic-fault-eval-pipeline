#!/usr/bin/env python3
"""SEG-Y -> Parquet with PySpark: one row per trace.

How it works
------------
1. The driver reads only the 3600-byte file header (3200 textual + 400 binary)
   and works out the byte layout: samples per trace, sample format, trace
   length in bytes and number of traces.
2. The driver builds a small "plan" DataFrame: one row per chunk of traces
   (chunk_id, first_trace, n_traces). No seismic data moves through the driver.
3. mapInPandas runs read_chunks() on the executors. Each task opens the file,
   seeks straight to its byte range, reads it in one call and decodes headers
   and samples with NumPy (vectorized, no per-trace Python loop).
4. Output: <out>/traces/survey=<name>/ (one row per trace, samples as
   array<float>) and <out>/surveys/survey=<name>/ (one row of survey metadata).

Assumes fixed-length traces (true for nearly all post-stack volumes).
Works on local paths and gs:// paths (gs:// needs `pip install gcsfs`).
"""
import argparse
import os
import sys
import json
import struct
import time

import fsspec
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F, types as T

# Spark workers must run the same Python as this script (the venv). Without this,
# local mode looks for a default "python3"/"python" on PATH, which may not exist
# or may lack the packages. Dataproc sets PYSPARK_PYTHON itself, so keep any value.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# SEG-Y data sample format code -> (numpy kind, bytes per sample)
FORMATS = {1: ("ibm", 4), 2: ("i4", 4), 3: ("i2", 2), 5: ("f4", 4), 8: ("i1", 1)}


# ---------------------------------------------------------------- driver side
def read_file_header(path):
    """Read the 3600-byte header and derive the trace layout."""
    with fsspec.open(path, "rb") as f:
        head = f.read(3600)
        first_th = f.read(240)
    fs, p = fsspec.core.url_to_fs(path)
    size = fs.size(p)

    # SEG-Y is big-endian by standard, but some writers use little-endian.
    # The format code (bytes 3225-3226) must be a known value, so use it to detect byte order.
    for bo in (">", "<"):
        fmt = struct.unpack(bo + "h", head[3224:3226])[0]
        if fmt in FORMATS:
            break
    else:
        raise ValueError(f"Unrecognized sample format code in {path}")

    u16 = lambda off: struct.unpack(bo + "H", head[off:off + 2])[0]
    i16 = lambda off: struct.unpack(bo + "h", head[off:off + 2])[0]
    dt_us = u16(3216)            # bytes 3217-3218: sample interval (microseconds)
    ns = u16(3220)               # bytes 3221-3222: samples per trace
    revision = u16(3500)         # bytes 3501-3502
    n_ext = i16(3504)            # bytes 3505-3506: extended textual headers
    if n_ext < 0:
        raise ValueError("Variable number of extended textual headers is not supported")

    # The trace header also stores ns (bytes 115-116). Some writers leave the
    # binary-header copy at 0, so fall back to the trace header.
    ns_trace = struct.unpack(bo + "H", first_th[114:116])[0]
    if ns == 0:
        ns = ns_trace
    if ns_trace not in (0, ns):
        raise ValueError(f"ns mismatch: binary header {ns}, first trace header {ns_trace}")

    kind, bps = FORMATS[fmt]
    data_start = 3600 + 3200 * n_ext
    trace_len = 240 + ns * bps
    n_bytes = size - data_start
    if n_bytes % trace_len:
        raise ValueError(
            f"File size is not a whole number of {trace_len}-byte traces "
            f"(remainder {n_bytes % trace_len}); variable-length traces or trailing data?")

    text = head[:3200].decode("cp037", errors="replace")   # EBCDIC textual header
    if not text.startswith("C"):                             # some writers use ASCII
        text = head[:3200].decode("ascii", errors="replace")

    return dict(path=path, byteorder=bo, format_code=fmt, kind=kind, bps=bps,
                ns=ns, dt_ms=dt_us / 1000.0, revision=revision, n_ext=n_ext,
                data_start=data_start, trace_len=trace_len,
                n_traces=n_bytes // trace_len, file_bytes=size,
                text_header="\n".join(text[i:i + 80] for i in range(0, 3200, 80)))


# -------------------------------------------------------------- executor side
def ibm_to_ieee(u):
    """IBM System/360 float (uint32 array) -> float32.

    value = (-1)^sign * 0.mantissa(24 bits) * 16^(exponent - 64)
          = (-1)^sign * mantissa * 2^(4*(exponent - 64) - 24)
    """
    u = u.astype(np.uint32)
    sign = np.where(u >> 31, -1.0, 1.0)
    expo = ((u >> 24) & 0x7F).astype(np.int64)
    mant = (u & 0x00FFFFFF).astype(np.float64)
    return (sign * np.ldexp(mant, 4 * (expo - 64) - 24)).astype(np.float32)


def header_int(hdr, byte1, nbytes, bo):
    """Pull one integer field (1-based SEG-Y byte position) from all trace headers at once."""
    off = byte1 - 1
    cols = np.ascontiguousarray(hdr[:, off:off + nbytes])
    return cols.view(f"{bo}i{nbytes}").ravel().astype(np.int64)


def make_reader(info, survey, il_byte, xl_byte, x_byte, y_byte):
    bo, ns, tl = info["byteorder"], info["ns"], info["trace_len"]

    def read_chunks(batches):
        with fsspec.open(info["path"], "rb") as f:          # one open per task, not per chunk
            for plan in batches:
                for first, n in zip(plan["first_trace"], plan["n_traces"]):
                    f.seek(info["data_start"] + int(first) * tl)
                    raw = np.frombuffer(f.read(int(n) * tl), dtype=np.uint8).reshape(int(n), tl)
                    hdr, body = raw[:, :240], np.ascontiguousarray(raw[:, 240:])

                    if info["kind"] == "ibm":
                        samples = ibm_to_ieee(body.view(f"{bo}u4"))
                    else:
                        samples = body.view(f"{bo}{info['kind']}").astype(np.float32)
                    samples = samples.reshape(int(n), ns)

                    # Coordinate scalar (bytes 71-72): negative = divide, positive = multiply.
                    scal = header_int(hdr, 71, 2, bo).astype(np.float64)
                    factor = np.where(scal < 0, -1.0 / np.where(scal == 0, 1, scal),
                                      np.where(scal == 0, 1.0, scal))

                    yield pd.DataFrame({
                        "survey": survey,
                        "trace_seq": np.arange(first, first + n, dtype=np.int64),
                        "il": header_int(hdr, il_byte, 4, bo).astype(np.int32),
                        "xl": header_int(hdr, xl_byte, 4, bo).astype(np.int32),
                        "cdp_x": header_int(hdr, x_byte, 4, bo) * factor,
                        "cdp_y": header_int(hdr, y_byte, 4, bo) * factor,
                        "samples": list(samples),     # Arrow converts to array<float>
                    })

    return read_chunks


TRACE_SCHEMA = T.StructType([
    T.StructField("survey", T.StringType()),
    T.StructField("trace_seq", T.LongType()),
    T.StructField("il", T.IntegerType()),
    T.StructField("xl", T.IntegerType()),
    T.StructField("cdp_x", T.DoubleType()),
    T.StructField("cdp_y", T.DoubleType()),
    T.StructField("samples", T.ArrayType(T.FloatType())),
])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("segy", help="local path or gs:// URI")
    ap.add_argument("--survey", required=True, help="survey name, e.g. F3")
    ap.add_argument("--out", required=True, help="output root (local dir or gs://bucket/prefix)")
    ap.add_argument("--traces-per-chunk", type=int, default=2000)
    ap.add_argument("--il-byte", type=int, default=189, help="1-based header byte for inline")
    ap.add_argument("--xl-byte", type=int, default=193, help="1-based header byte for crossline")
    ap.add_argument("--x-byte", type=int, default=181)
    ap.add_argument("--y-byte", type=int, default=185)
    ap.add_argument("--master", default=None, help="e.g. local[*]; omit on Dataproc")
    ap.add_argument("--driver-memory", default="4g")
    args = ap.parse_args()

    t0 = time.time()
    info = read_file_header(args.segy)
    print(json.dumps({k: v for k, v in info.items() if k != "text_header"}, indent=2))

    b = SparkSession.builder.appName(f"segy_to_parquet:{args.survey}")
    if args.master:
        b = b.master(args.master).config("spark.driver.memory", args.driver_memory)
    # Overwrite only the partitions being written, so other surveys survive a rerun.
    spark = (b.config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # Plan: one row per chunk, one Spark partition per chunk.
    n, k = info["n_traces"], args.traces_per_chunk
    plan_rows = [(i, s, min(k, n - s)) for i, s in enumerate(range(0, n, k))]
    plan = (spark.sparkContext.parallelize(plan_rows, numSlices=len(plan_rows))
            .toDF("chunk_id long, first_trace long, n_traces long"))
    print(f"{n} traces -> {len(plan_rows)} chunks of up to {k} traces "
          f"(~{k * info['trace_len'] / 1e6:.1f} MB each)")

    traces = plan.mapInPandas(
        make_reader(info, args.survey, args.il_byte, args.xl_byte, args.x_byte, args.y_byte),
        schema=TRACE_SCHEMA)

    out = args.out.rstrip("/")
    traces.write.mode("overwrite").partitionBy("survey").parquet(f"{out}/traces")

    meta = {k: info[k] for k in ("path", "format_code", "ns", "dt_ms", "revision",
                                  "n_traces", "file_bytes", "text_header")}
    meta["byteorder"] = "big" if info["byteorder"] == ">" else "little"
    meta.update(survey=args.survey, il_byte=args.il_byte, xl_byte=args.xl_byte)
    spark.createDataFrame([meta]).write.mode("overwrite").partitionBy("survey") \
        .parquet(f"{out}/surveys")

    # Quick geometry sanity check from what was written (full QC is Day 2, in SQL).
    t = spark.read.parquet(f"{out}/traces").where(F.col("survey") == args.survey)
    s = t.agg(F.count("*").alias("n"),
              F.countDistinct("il", "xl").alias("n_positions"),
              F.min("il").alias("il_min"), F.max("il").alias("il_max"),
              F.min("xl").alias("xl_min"), F.max("xl").alias("xl_max")).first()
    full = (s.il_max - s.il_min + 1) * (s.xl_max - s.xl_min + 1)
    print(f"traces={s.n} distinct positions={s.n_positions} "
          f"il {s.il_min}-{s.il_max} xl {s.xl_min}-{s.xl_max} "
          f"full grid (step 1)={full} missing={full - s.n_positions} "
          f"duplicates={s.n - s.n_positions}")
    print(f"elapsed {time.time() - t0:.1f} s")
    spark.stop()


if __name__ == "__main__":
    main()
