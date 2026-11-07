#!/usr/bin/env python3
"""Compare segy_to_parquet.py output with the synthetic truth file."""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("truth", help="<segy>.truth.npz from make_synthetic_segy.py")
ap.add_argument("traces_dir", help="<out>/traces/survey=<name>")
args = ap.parse_args()

tr = np.load(args.truth)
df = pd.read_parquet(args.traces_dir).sort_values("trace_seq").reset_index(drop=True)

assert len(df) == len(tr["il"]), f"trace count {len(df)} != {len(tr['il'])}"
assert (df["il"].to_numpy() == tr["il"]).all(), "inline mismatch"
assert (df["xl"].to_numpy() == tr["xl"]).all(), "crossline mismatch"
assert np.allclose(df["cdp_x"], tr["x"]) and np.allclose(df["cdp_y"], tr["y"]), "coordinate mismatch"

got = np.stack(df["samples"].to_numpy())
err = np.abs(got - tr["data"])
rel = err.max() / np.abs(tr["data"]).max()
print(f"traces {len(df)}  headers OK  max abs err {err.max():.3g}  rel {rel:.2e}")
assert rel < 1e-6, "sample mismatch"
print("PASS")
