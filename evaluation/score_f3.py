#!/usr/bin/env python3
"""Score FaultSeg3D on F3 against sparse interpreter picks, with fault_metrics.py.

Inputs (from the seismic-fault-labeling repo):
  <labels-dir>/labels_sparse.npy  uint8 (il, xl, t): 1 = fault, 0 = labeled
                                  background, 255 = unlabeled inline
  <labels-dir>/meta.json          inline numbers
  <pred-dir>/prob_crop.npy        float (il, xl, t), FaultSeg3D probability on a crop
  <pred-dir>/crop.json            crop box {"lo": [il, xl, t], "hi": [...]} (index ranges)

Only labeled inlines are scored (any value != 255). Each becomes one 2-D section
with rows = time and cols = crossline, the layout fault_metrics.py expects.

Precision is reported two ways, as in score_faults.py:
  all       predictions anywhere on the labeled inline. Pessimistic: only one
            fault was labeled, so other real faults count as false positives.
  corridor  predictions within --region-radius samples of a pick
            (fault_metrics.label_region). Estimates precision where labels exist.

Uncertainty: block bootstrap over contiguous blocks of labeled inlines. Picks are
every 10th inline, so a block of 4 labeled inlines spans about 40 inlines.

Thinning (--thin): FaultSeg3D predicts bands several samples wide, so most
predicted pixels sit beside the fault even when the band is in the right place.
Non-maximum suppression (fault_metrics.thin_faults) keeps the ridge of each band.

Threshold: the best-F1 threshold on all inlines is optimistic, so it is also
chosen on every other block of inlines and reported on the remaining blocks.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fault_metrics as fm  # noqa: E402


def load(labels_dir, pred_dir):
    labels_dir, pred_dir = pathlib.Path(labels_dir), pathlib.Path(pred_dir)
    lab = np.load(labels_dir / "labels_sparse.npy", mmap_mode="r")
    prob = np.load(pred_dir / "prob_crop.npy", mmap_mode="r")
    crop = json.loads((pred_dir / "crop.json").read_text())
    lo, hi = crop["lo"], crop["hi"]
    lab = np.asarray(lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    if lab.shape != prob.shape:
        raise SystemExit(f"label crop {lab.shape} != prob_crop {prob.shape}")

    rows = [k for k in range(lab.shape[0]) if (lab[k] != 255).any()]
    if not rows:
        raise SystemExit("no labeled inlines inside the crop")
    il_numbers = json.loads((labels_dir / "meta.json").read_text())["il_numbers"]
    il = [int(il_numbers[lo[0] + k]) for k in rows]

    L = lab[rows].transpose(0, 2, 1)                       # (section, t, xl)
    P = np.asarray(prob[rows], dtype=np.float32).transpose(0, 2, 1)
    return P, L == 1, L != 255, il, crop


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels-dir", required=True, help="folder with labels_sparse.npy + meta.json")
    ap.add_argument("--pred-dir", required=True, help="folder with prob_crop.npy + crop.json")
    ap.add_argument("--tol", type=float, default=3.0, help="match tolerance in samples")
    ap.add_argument("--region-radius", type=float, default=40.0, help="corridor half-width")
    ap.add_argument("--block", type=int, default=4, help="labeled inlines per bootstrap block")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--compare", type=float, nargs="*",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    help="thresholds for a table in score_faults.py's format")
    ap.add_argument("--thin", action="store_true",
                    help="thin probability bands to 1-pixel ridges (non-maximum suppression) first")
    ap.add_argument("--thin-sigma", type=float, default=1.0, help="smoothing before thinning")
    ap.add_argument("--tol-sweep", type=float, nargs="*", default=None,
                    help="also report best F1 at each of these tolerances, e.g. 1 2 3 5 8")
    ap.add_argument("--out", default=None, help="write metrics.json and pr_curve.csv here")
    args = ap.parse_args()

    P, label, valid, il, crop = load(args.labels_dir, args.pred_dir)
    if args.thin:
        P = fm.thin_faults(P, sigma=args.thin_sigma)
    corridor = valid & fm.label_region(label, args.region_radius)
    print(f"{len(il)} labeled inlines ({il[0]}..{il[-1]}), section {P.shape[1]} t x {P.shape[2]} xl, "
          f"{int(label.sum())} fault px, tol {args.tol:g}, corridor {args.region_radius:g}, "
          f"thinning {'on (sigma ' + format(args.thin_sigma, 'g') + ')' if args.thin else 'off'}")

    c_all = fm.curve_counts(P, label, args.tol, region=valid)
    c_cor = fm.curve_counts(P, label, args.tol, region=corridor)
    s_all, s_cor = fm.summarize(c_all), fm.summarize(c_cor)

    # Same columns as score_faults.py, for a direct cross-check.
    print(f"\n{'thresh':>6} {'recall':>7} {'prec(all)':>10} {'prec(corridor)':>15} {'pred px':>9}")
    for t in args.compare:
        i = int(np.argmin(np.abs(c_all.thresholds - t)))
        print(f"{c_all.thresholds[i]:6.2f} {s_all['recall'][i]:7.3f} {s_all['precision'][i]:10.3f} "
              f"{s_cor['precision'][i]:15.3f} {int(c_all.n_pred[:, i].sum()):9d}")

    # Optimistic: threshold chosen on everything. Honest: chosen on alternate blocks.
    calib, test = fm.alternate_blocks(len(il), args.block)
    i_cal = fm.pick_threshold(c_cor, calib)
    p_t, r_t, f_t = fm.prf(*[x[i_cal] if np.ndim(x) else x for x in c_cor.totals(test)])

    b = fm.block_bootstrap(c_cor, s_cor["best_index"], block=args.block, n_boot=args.n_boot)
    thr = s_cor["best_threshold"]
    d_px = fm.dice_iou((P >= thr) & corridor, label, 0)
    d_tol = fm.dice_iou((P >= thr) & corridor, label, args.tol)

    print(f"\nAP (PR-AUC)       all {s_all['ap']:.3f}   corridor {s_cor['ap']:.3f}")
    print(f"best-F1 (corridor, threshold chosen on all inlines: optimistic)")
    for k in ("precision", "recall", "f1"):
        lo, hi = b[f"{k}_ci"]
        print(f"  {k:9s} {b[k]:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   at t={thr:.2f}, "
              f"{b['n_blocks']} blocks of {args.block}")
    print(f"held-out (threshold {c_cor.thresholds[i_cal]:.2f} chosen on {len(calib)} inlines, "
          f"reported on {len(test)}):  P {p_t:.3f}  R {r_t:.3f}  F1 {f_t:.3f}")
    print(f"Dice/IoU in corridor at t={thr:.2f}: pixel {d_px[0]:.3f}/{d_px[1]:.3f}   "
          f"dilated {args.tol:g}: {d_tol[0]:.3f}/{d_tol[1]:.3f}")

    # F1 against tolerance: one number at one tolerance hides how fast the score
    # improves as matching gets looser (see the README's F3 results).
    tol_curve = []
    if args.tol_sweep:
        print(f"\n{'tol':>5} {'best t':>7} {'F1':>6} {'95% CI':>17} {'P':>6} {'R':>6} {'held-out F1':>12}")
        for tol in args.tol_sweep:
            c = fm.curve_counts(P, label, tol, region=corridor)
            sm = fm.summarize(c)
            bb = fm.block_bootstrap(c, sm["best_index"], block=args.block, n_boot=args.n_boot)
            ic = fm.pick_threshold(c, calib)
            hf = fm.prf(*[x[ic] if np.ndim(x) else x for x in c.totals(test)])[2]
            tol_curve.append(dict(tol=tol, best_threshold=sm["best_threshold"], f1=bb["f1"],
                                  f1_ci=bb["f1_ci"], precision=bb["precision"], recall=bb["recall"],
                                  heldout_f1=float(hf)))
            print(f"{tol:5g} {sm['best_threshold']:7.3f} {bb['f1']:6.3f} "
                  f"[{bb['f1_ci'][0]:.3f}, {bb['f1_ci'][1]:.3f}] {bb['precision']:6.3f} "
                  f"{bb['recall']:6.3f} {hf:12.3f}")

    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        metrics = dict(n_inlines=len(il), inlines=il, tol=args.tol, region_radius=args.region_radius,
                       thin=args.thin, thin_sigma=args.thin_sigma if args.thin else None,
                       ap_all=s_all["ap"], ap_corridor=s_cor["ap"], best_threshold=thr,
                       precision=b["precision"], recall=b["recall"], f1=b["f1"],
                       precision_ci=b["precision_ci"], recall_ci=b["recall_ci"], f1_ci=b["f1_ci"],
                       block=args.block, n_blocks=b["n_blocks"],
                       heldout_threshold=float(c_cor.thresholds[i_cal]),
                       heldout_precision=float(p_t), heldout_recall=float(r_t), heldout_f1=float(f_t),
                       dice_pixel=d_px[0], iou_pixel=d_px[1], dice_tol=d_tol[0], iou_tol=d_tol[1],
                       tol_curve=tol_curve, crop=crop)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        with open(out / "pr_curve.csv", "w") as f:
            f.write("threshold,recall,precision_all,precision_corridor,f1_corridor\n")
            for i, t in enumerate(c_all.thresholds):
                f.write(f"{t:.2f},{s_all['recall'][i]:.6f},{s_all['precision'][i]:.6f},"
                        f"{s_cor['precision'][i]:.6f},{s_cor['f1'][i]:.6f}\n")
        if tol_curve:
            with open(out / "tol_curve.csv", "w") as f:
                f.write("tol,best_threshold,f1,f1_lo,f1_hi,precision,recall,heldout_f1\n")
                for r in tol_curve:
                    f.write(f"{r['tol']:g},{r['best_threshold']:.3f},{r['f1']:.6f},{r['f1_ci'][0]:.6f},"
                            f"{r['f1_ci'][1]:.6f},{r['precision']:.6f},{r['recall']:.6f},"
                            f"{r['heldout_f1']:.6f}\n")
        print(f"\nwrote {out}/metrics.json, pr_curve.csv" + (", tol_curve.csv" if tol_curve else ""))


if __name__ == "__main__":
    main()
