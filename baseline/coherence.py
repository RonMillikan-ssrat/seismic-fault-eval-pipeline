#!/usr/bin/env python3
"""Classical fault baseline: semblance coherence, computed from the Parquet trace lake.

Semblance (Marfurt et al. style, without dip steering)
------------------------------------------------------
For each sample, take a small window of neighboring traces (n_il x n_xl) and a
short time window (2k+1 samples). Semblance compares the energy of the stacked
traces with the total energy of the traces:

    semblance = sum_t (mean_i a_i(t))^2 / sum_t mean_i (a_i(t)^2)       in [0, 1]

If every trace in the window looks the same, stacking loses nothing and semblance
is 1. Across a fault the reflections are offset, stacking partly cancels, and
semblance drops. The fault attribute is 1 - semblance, also in [0, 1], so it can
be thresholded and scored exactly like a fault probability.

Both sums are box filters (scipy.ndimage.uniform_filter), so the whole volume is a
few array passes, with no Python loop over samples.

Known limitation: without dip steering, steeply DIPPING reflectors also lower
semblance (the traces are shifted copies), so dip shows up as false faults. F3 is
mostly gently dipping. Dip-steered semblance would align traces along the local
dip first.

Reading from the lake
---------------------
The crop box from FaultSeg3D's crop.json (index ranges into the label volume)
is turned into inline/crossline number ranges using meta.json, then read with a
pyarrow dataset filter on survey, il and xl. The traces table is written in trace
order, so each part file covers a narrow inline range; its footer min/max lets
pyarrow skip files outside the crop (predicate pushdown). Missing positions are
left as zeros; duplicates keep the first copy (lowest trace_seq).

Output: <out>/prob_crop.npy (il, xl, t) and <out>/crop.json, the same layout as
predict_faultseg.py, so evaluation/score_f3.py scores the baseline unchanged.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
from scipy import ndimage as ndi


def load_cube(lake_root, survey, il_range, xl_range, t_range=None):
    """Traces for il in [il0, il1), xl in [xl0, xl1) as an (il, xl, t) float32 cube."""
    (il0, il1), (xl0, xl1) = il_range, xl_range
    dset = ds.dataset(f"{str(lake_root).rstrip('/')}/traces", format="parquet", partitioning="hive")
    flt = ((pc.field("survey") == survey) & (pc.field("il") >= il0) & (pc.field("il") < il1)
           & (pc.field("xl") >= xl0) & (pc.field("xl") < xl1))
    tbl = dset.to_table(columns=["trace_seq", "il", "xl", "samples"], filter=flt)
    if tbl.num_rows == 0:
        raise SystemExit(f"no traces for {survey} il {il0}-{il1 - 1} xl {xl0}-{xl1 - 1} in {lake_root}")

    seq = tbl.column("trace_seq").to_numpy()
    il = tbl.column("il").to_numpy()
    xl = tbl.column("xl").to_numpy()
    flat = tbl.column("samples").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    ns = flat.size // tbl.num_rows
    traces = flat.reshape(tbl.num_rows, ns).astype(np.float32)
    if t_range is not None:
        traces = traces[:, t_range[0]:t_range[1]]

    cube = np.zeros((il1 - il0, xl1 - xl0, traces.shape[1]), np.float32)
    order = np.argsort(seq)[::-1]          # assign last-to-first so the first copy wins
    cube[il[order] - il0, xl[order] - xl0] = traces[order]
    n_pos = len(set(zip(il.tolist(), xl.tolist())))
    return cube, dict(rows_read=int(tbl.num_rows), positions=n_pos,
                      missing=int(cube.shape[0] * cube.shape[1] - n_pos),
                      duplicates=int(tbl.num_rows - n_pos))


def semblance(cube, n_il=3, n_xl=3, half_t=4):
    """Semblance of an (il, xl, t) cube over an n_il x n_xl x (2*half_t + 1) window."""
    a = np.asarray(cube, np.float64)
    lateral = (n_il, n_xl, 1)
    stack_sq = ndi.uniform_filter(a, size=lateral, mode="reflect") ** 2      # (mean_i a)^2
    energy = ndi.uniform_filter(a * a, size=lateral, mode="reflect")         # mean_i a^2
    w = 2 * half_t + 1
    num = ndi.uniform_filter1d(stack_sq, size=w, axis=2, mode="reflect")
    den = ndi.uniform_filter1d(energy, size=w, axis=2, mode="reflect")
    s = np.where(den > 1e-12 * max(den.max(), 1e-30), num / np.maximum(den, 1e-30), 1.0)
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def fault_attribute(cube, **kw):
    """1 - semblance: near 0 where traces agree, higher across discontinuities."""
    return (1.0 - semblance(cube, **kw)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lake", required=True, help="lake root written by spark/segy_to_parquet.py")
    ap.add_argument("--survey", default="F3")
    ap.add_argument("--crop", required=True, help="crop.json from predict_faultseg.py")
    ap.add_argument("--meta", required=True, help="meta.json from extract_inlines.py (inline numbers)")
    ap.add_argument("--n-il", type=int, default=3)
    ap.add_argument("--n-xl", type=int, default=3)
    ap.add_argument("--half-t", type=int, default=4, help="time half-window in samples")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    crop = json.loads(pathlib.Path(args.crop).read_text())
    meta = json.loads(pathlib.Path(args.meta).read_text())
    lo, hi = crop["lo"], crop["hi"]
    il_numbers = meta["il_numbers"]
    il_range = (int(il_numbers[lo[0]]), int(il_numbers[hi[0] - 1]) + 1)
    xl_range = (int(meta["xl_first"]) + lo[1], int(meta["xl_first"]) + hi[1])
    if il_range[1] - il_range[0] != hi[0] - lo[0]:
        raise SystemExit("inline numbers in the crop are not consecutive; this loader assumes step 1")

    cube, info = load_cube(args.lake, args.survey, il_range, xl_range, (lo[2], hi[2]))
    print(f"read il {il_range[0]}-{il_range[1] - 1}, xl {xl_range[0]}-{xl_range[1] - 1}, "
          f"t {lo[2]}-{hi[2] - 1} -> cube {cube.shape} ({cube.nbytes / 1e6:.0f} MB); {info}  "
          f"[{time.time() - t0:.1f} s]")

    t1 = time.time()
    attr = fault_attribute(cube, n_il=args.n_il, n_xl=args.n_xl, half_t=args.half_t)
    print(f"semblance window {args.n_il} il x {args.n_xl} xl x {2 * args.half_t + 1} t  "
          f"[{time.time() - t1:.1f} s];  attribute p50 {np.percentile(attr, 50):.3f}  "
          f"p99 {np.percentile(attr, 99):.3f}  max {attr.max():.3f}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "prob_crop.npy", attr)
    (out / "crop.json").write_text(json.dumps(
        dict(lo=lo, hi=hi, source="semblance coherence (1 - semblance)", lake=str(args.lake),
             survey=args.survey, window=[args.n_il, args.n_xl, 2 * args.half_t + 1]), indent=1))
    print(f"wrote {out}/prob_crop.npy and crop.json")


if __name__ == "__main__":
    main()
