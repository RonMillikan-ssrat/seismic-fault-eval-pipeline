"""Small helpers shared by the tests and conftest.py."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MASTER = "local[2]"
CROP = ["--il-range", "105", "134", "--xl-range", "310", "339"]
CROP_BOUNDS = ((105, 134), (310, 339))


def run(*args):
    """Run one of the repo's scripts with this interpreter; fail the test with its output."""
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        pytest.fail(f"command failed: {' '.join(map(str, args[:3]))}\n"
                    f"--- stdout ---\n{r.stdout[-2000:]}\n--- stderr ---\n{r.stderr[-3000:]}")
    return r.stdout


def at(counts, thr, prf):
    """(precision, recall, F1) from a Counts object at the threshold nearest thr."""
    import numpy as np
    i = int(np.argmin(np.abs(counts.thresholds - thr)))
    return prf(*[x[i] if np.ndim(x) else x for x in counts.totals()])
