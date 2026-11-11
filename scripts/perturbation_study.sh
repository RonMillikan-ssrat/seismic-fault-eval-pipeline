#!/usr/bin/env bash
# Perturbation study on F3: perturb the crop (Spark), run FaultSeg3D on each version,
# score each against the picks, and collect a summary table.
#
#   STEPS="perturb infer score summary attrs drift" (default; run a subset, e.g. STEPS="attrs drift")
#   Inference skips surveys that already have prob_crop.npy, so the script can be
#   stopped and restarted.
#
# Two Pythons:
#   PY    the eval-pipeline venv (Spark, scoring)          default: python on PATH
#   TFPY  a Python with TensorFlow (the labeling repo venv) default: $L/venv/bin/python
#         It also needs pyarrow and scipy (pip install pyarrow scipy).
#
# Disk: each perturbed crop adds ~0.35 GB to the lake and each prediction ~0.33 GB,
# so the default 9 versions + F3 need ~6.5 GB on the WSL disk. Check C: first.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
L=${L:-$HOME/devl/seismic-fault-labeling}
LAKE=${LAKE:-$HOME/lake}
STUDY=${STUDY:-$HOME/f3_study}
PY=${PY:-$(command -v python)}
TFPY=${TFPY:-$L/venv/bin/python}
STEPS=${STEPS:-"perturb infer score summary attrs drift"}
DRIVER_MEM=${DRIVER_MEM:-16g}          # Spark driver memory; use 6g on a 16 GB machine
CROP="$L/pred/crop.json"
META="$L/inlines/meta.json"
SURVEYS=${SURVEYS:-"F3 F3-noise-0.25 F3-noise-0.5 F3-noise-1 F3-lowpass-40 F3-lowpass-25 F3-lowpass-15 F3-gain-4 F3-tgain-1 F3-tgain-2"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
mkdir -p "$STUDY"

if has perturb; then
  for spec in "noise 0.25 0.5 1" "lowpass 40 25 15" "gain 4" "tgain 1 2"; do
    set -- $spec
    kind=$1; shift
    "$PY" "$HERE/spark/perturb.py" --root "$LAKE" --survey F3 --kind "$kind" --levels "$@" \
        --crop "$CROP" --meta "$META" --master 'local[*]' --driver-memory "$DRIVER_MEM"
  done
fi

if has infer; then
  for s in $SURVEYS; do
    out="$STUDY/$s/pred"
    if [[ -f "$out/prob_crop.npy" ]]; then echo "skip infer $s (exists)"; continue; fi
    mkdir -p "$STUDY/$s"                     # tee opens the log before Python creates the folder
    extra=()
    [[ "$s" == "F3" ]] && extra=(--compare "$L/pred/prob_crop.npy")   # must reproduce the Oct 31 run
    echo "=== infer $s"
    "$TFPY" "$HERE/infer/faultseg_from_lake.py" --labeling-repo "$L" --weights "$L/models/fseg-70.hdf5" \
        --lake "$LAKE" --survey "$s" --crop "$CROP" --meta "$META" --out "$out" "${extra[@]}" \
        2>&1 | grep -vE "^\s*tile [0-9]+/" | tee "$STUDY/$s/infer.log"
  done
fi

if has score; then
  for s in $SURVEYS; do
    echo "=== score $s"
    "$PY" "$HERE/evaluation/score_f3.py" --labels-dir "$L/inlines" --pred-dir "$STUDY/$s/pred" \
        --tol 3 --tol-sweep 1 3 8 --out "$STUDY/$s/score" > "$STUDY/$s/score.log"
    grep -E "^  f1 |held-out" "$STUDY/$s/score.log"
  done
fi

if has summary; then
  # f1       = best F1 with the threshold re-chosen on each survey (an upper bound).
  # f1_fixed = F1 at the threshold chosen on unperturbed F3: what a deployed model,
  #            whose threshold was set once, would score on the drifted data.
  "$PY" - "$STUDY" $SURVEYS <<'EOF'
import csv, json, sys, pathlib
study, surveys = pathlib.Path(sys.argv[1]), sys.argv[2:]
ref_t = json.loads((study / surveys[0] / "score" / "metrics.json").read_text())["best_threshold"]
def fixed(s):
    rows = list(csv.DictReader(open(study / s / "score" / "pr_curve.csv")))
    r = min(rows, key=lambda r: abs(float(r["threshold"]) - ref_t))
    return float(r["f1_corridor"]), float(r["precision_corridor"]), float(r["recall"])
rows = ["survey,kind,level,f1,f1_lo,f1_hi,heldout_f1,precision,recall,best_threshold,f1_tol1,f1_tol8,"
        "f1_fixed,precision_fixed,recall_fixed,fixed_threshold"]
print(f"{'survey':16s} {'best F1':>7} {'95% CI':>16} {'best t':>6} | at t={ref_t:.2f}: {'F1':>6} {'P':>6} {'R':>6}")
for s in surveys:
    m = json.loads((study / s / "score" / "metrics.json").read_text())
    kind, level = (s.split("-")[1], s.split("-")[2]) if "-" in s else ("none", "0")
    tc = {r["tol"]: r["f1"] for r in m["tol_curve"]}
    ff, pf, rf = fixed(s)
    rows.append(f"{s},{kind},{level},{m['f1']:.4f},{m['f1_ci'][0]:.4f},{m['f1_ci'][1]:.4f},"
                f"{m['heldout_f1']:.4f},{m['precision']:.4f},{m['recall']:.4f},{m['best_threshold']:.3f},"
                f"{tc.get(1.0, float('nan')):.4f},{tc.get(8.0, float('nan')):.4f},"
                f"{ff:.4f},{pf:.4f},{rf:.4f},{ref_t:.3f}")
    print(f"{s:16s} {m['f1']:7.3f} [{m['f1_ci'][0]:.3f}, {m['f1_ci'][1]:.3f}] {m['best_threshold']:6.3f} | "
          f"{'':11s}{ff:6.3f} {pf:6.3f} {rf:6.3f}")
(study / "summary.csv").write_text("\n".join(rows) + "\n")
print(f"wrote {study / 'summary.csv'}")
EOF
fi

if has attrs; then
  # Per-trace attributes (RMS, spectral centroid, ...) for each survey -> trace_stats
  for s in $SURVEYS; do
    echo "=== attributes $s"
    "$PY" "$HERE/spark/trace_attributes.py" --root "$LAKE" --survey "$s" --master 'local[*]' \
        --driver-memory "$DRIVER_MEM" --variants pandas --repeat 1 2>&1 | grep -E "^pandas|^wrote"
  done
fi

if has drift; then
  # Label-free drift signals vs true F1 -> drift_table.csv, drift_vs_f1.png
  "$PY" "$HERE/monitoring/drift_signals.py" --root "$LAKE" --study "$STUDY" --reference F3 \
      --crop "$CROP" --meta "$META" --driver-memory "$DRIVER_MEM"
fi
