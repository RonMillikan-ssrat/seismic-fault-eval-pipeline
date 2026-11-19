"""SEG-Y -> Parquet round trip on synthetic IBM and IEEE files with known contents:
trace count, inline/crossline headers, coordinates and samples (rel err < 1e-6)."""
import pytest

from check_roundtrip import compare

pytestmark = pytest.mark.spark


@pytest.mark.parametrize("fmt", [1, 5], ids=["ibm", "ieee"])
def test_segy_to_parquet_roundtrip(roundtrip_lakes, fmt):
    truth, traces = roundtrip_lakes[fmt]
    n, _, rel = compare(truth, traces)
    assert n > 0
    assert rel < 1e-6
