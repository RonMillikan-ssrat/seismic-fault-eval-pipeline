#!/usr/bin/env python3
"""Run pretrained FaultSeg3D on one survey's crop, read from the Parquet lake.

Uses the functions in seismic-fault-labeling/predict_faultseg.py unchanged
(build_unet, load_legacy_weights, normalize, predict_tiled), so inference is
identical to the original SEG-Y run; only the source of the amplitudes differs:

    lake (il, xl, t) crop -> transpose to (t, xl, il), time first as the model expects
    -> normalize the whole crop to zero mean, unit std -> 128^3 tiles, 12-sample
    overlap, Gaussian edge blending -> transpose back -> prob_crop.npy

Output matches predict_faultseg.py (prob_crop.npy + crop.json), so score_f3.py,
compare_models.py and the pipeline read it unchanged.

Check: run on the unperturbed survey with --compare <original prob_crop.npy>.
The lake reader and the SEG-Y reader should give the same amplitudes, so the
probabilities should match to float rounding.

Needs TensorFlow: run it from the seismic-fault-labeling venv (plus pyarrow).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "baseline"))
from coherence import crop_ranges, load_cube  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labeling-repo", required=True, help="folder containing predict_faultseg.py")
    ap.add_argument("--weights", required=True, help="fseg-70.hdf5")
    ap.add_argument("--lake", required=True)
    ap.add_argument("--survey", required=True, help="e.g. F3 or F3-noise-0.5")
    ap.add_argument("--crop", required=True, help="crop.json defining the box (from the original run)")
    ap.add_argument("--meta", required=True, help="meta.json from extract_inlines.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="optional prob_crop.npy to compare against")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(args.labeling_repo).expanduser().resolve()))
    import predict_faultseg as pf

    crop = json.loads(pathlib.Path(args.crop).read_text())
    meta = json.loads(pathlib.Path(args.meta).read_text())
    il_range, xl_range, t_range = crop_ranges(crop, meta)
    t0 = time.time()
    cube, info = load_cube(args.lake, args.survey, il_range, xl_range, t_range)
    print(f"{args.survey}: cube (il, xl, t) {cube.shape}, {info}  [{time.time() - t0:.1f} s]")

    model = pf.build_unet()
    pf.load_legacy_weights(model, args.weights)
    t1 = time.time()
    gx = pf.normalize(np.transpose(cube, (2, 1, 0)))            # (t, xl, il), time first
    fp = np.transpose(pf.predict_tiled(model, gx), (2, 1, 0)).astype(np.float32)
    print(f"\ninference {time.time() - t1:.0f} s;  prob p50 {np.percentile(fp, 50):.3f}  "
          f"p99 {np.percentile(fp, 99):.3f}  frac > 0.5 {np.mean(fp > 0.5):.4f}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "prob_crop.npy", fp)
    (out / "crop.json").write_text(json.dumps(
        dict(lo=crop["lo"], hi=crop["hi"], survey=args.survey, lake=str(args.lake),
             source="FaultSeg3D fseg-70 via infer/faultseg_from_lake.py"), indent=1))
    print(f"wrote {out}/prob_crop.npy and crop.json")

    if args.compare:
        ref = np.load(args.compare, mmap_mode="r")
        d = np.abs(fp - ref)
        r = np.corrcoef(fp[::4, ::4, ::4].ravel(), np.asarray(ref[::4, ::4, ::4]).ravel())[0, 1]
        print(f"vs {args.compare}: max |diff| {d.max():.2e}, mean |diff| {d.mean():.2e}, "
              f"corr {r:.6f}, pixels changing side of 0.5: {np.mean((fp > 0.5) != (ref > 0.5)):.2e}")


if __name__ == "__main__":
    main()
