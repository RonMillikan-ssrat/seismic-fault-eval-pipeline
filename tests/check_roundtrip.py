#!/usr/bin/env python3
"""Compare segy_to_parquet.py output with the synthetic truth file.

Also imported by tests/test_roundtrip.py (compare()).
"""
import argparse

import numpy as np
import pandas as pd


def compare(truth_path, traces_dir):
    """Raise AssertionError on any mismatch; return (n_traces, max abs err, rel err)."""
    tr = np.load(truth_path)
    df = pd.read_parquet(traces_dir).sort_values("trace_seq").reset_index(drop=True)

    assert len(df) == len(tr["il"]), f"trace count {len(df)} != {len(tr['il'])}"
    assert (df["il"].to_numpy() == tr["il"]).all(), "inline mismatch"
    assert (df["xl"].to_numpy() == tr["xl"]).all(), "crossline mismatch"
    assert np.allclose(df["cdp_x"], tr["x"]) and np.allclose(df["cdp_y"], tr["y"]), "coordinate mismatch"

    got = np.stack(df["samples"].to_numpy())
    err = np.abs(got - tr["data"])
    rel = err.max() / np.abs(tr["data"]).max()
    assert rel < 1e-6, f"sample mismatch (rel err {rel:.2e})"
    return len(df), float(err.max()), float(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth", help="<segy>.truth.npz from make_synthetic_segy.py")
    ap.add_argument("traces_dir", help="<out>/traces/survey=<name>")
    args = ap.parse_args()
    n, err, rel = compare(args.truth, args.traces_dir)
    print(f"traces {n}  headers OK  max abs err {err:.3g}  rel {rel:.2e}")
    print("PASS")


if __name__ == "__main__":
    main()
