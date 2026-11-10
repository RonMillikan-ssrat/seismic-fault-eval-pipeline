#!/usr/bin/env bash
# Perturbation study on F3: perturb the crop (Spark), run FaultSeg3D on each version,
# score each against the picks, and collect a summary table.
#
#   STEPS="perturb infer score summary" (default; run a subset, e.g. STEPS="score summary")
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
STEPS=${STEPS:-"perturb infer score summary"}
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
        --crop "$CROP" --meta "$META" --master 'local[*]' --driver-memory 16g
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
  "$PY" - "$STUDY" $SURVEYS <<'EOF'
import json, sys, pathlib
study, surveys = pathlib.Path(sys.argv[1]), sys.argv[2:]
rows = ["survey,kind,level,f1,f1_lo,f1_hi,heldout_f1,precision,recall,best_threshold,f1_tol1,f1_tol8"]
print(f"{'survey':16s} {'F1':>6} {'95% CI':>16} {'held-out':>9} {'P':>6} {'R':>6} {'t':>6}")
for s in surveys:
    m = json.loads((study / s / "score" / "metrics.json").read_text())
    kind, level = (s.split("-")[1], s.split("-")[2]) if "-" in s else ("none", "0")
    tc = {r["tol"]: r["f1"] for r in m["tol_curve"]}
    rows.append(f"{s},{kind},{level},{m['f1']:.4f},{m['f1_ci'][0]:.4f},{m['f1_ci'][1]:.4f},"
                f"{m['heldout_f1']:.4f},{m['precision']:.4f},{m['recall']:.4f},{m['best_threshold']:.3f},"
                f"{tc.get(1.0, float('nan')):.4f},{tc.get(8.0, float('nan')):.4f}")
    print(f"{s:16s} {m['f1']:6.3f} [{m['f1_ci'][0]:.3f}, {m['f1_ci'][1]:.3f}] {m['heldout_f1']:9.3f} "
          f"{m['precision']:6.3f} {m['recall']:6.3f} {m['best_threshold']:6.3f}")
(study / "summary.csv").write_text("\n".join(rows) + "\n")
print(f"wrote {study / 'summary.csv'}")
EOF
fi
