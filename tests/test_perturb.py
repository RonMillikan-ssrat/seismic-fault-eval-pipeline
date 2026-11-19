"""Known-answer tests for spark/perturb.py, on a small synthetic lake.

  gain 4       output == 4 x input exactly (float32 rounding)
  tgain p      output / input == (t / t_mid)^p sample by sample
  noise L      std(output - input) == L x crop RMS (within 2%); rerun gives identical noise
  lowpass fc   energy above 2 x fc removed (< 0.1% of the input's energy there, Hann-tapered FFT)
  all          same traces (il, xl) as the crop; untouched survey still present
(Converted from tests/check_perturb.py; same cases, same expected values.)
"""
import numpy as np
import pandas as pd
import pytest

from helpers import run

DT = 0.004


def read(root, survey):
    df = pd.read_parquet(f"{root}/traces/survey={survey}").sort_values(["il", "xl"]).reset_index(drop=True)
    return df, np.stack(df["samples"].to_numpy()).astype(np.float64)


@pytest.fixture(scope="module")
def source(perturb_lake):
    lake, _ = perturb_lake
    src, x = read(lake, "S")
    inside = src.il.between(105, 134) & src.xl.between(310, 339)
    return src[inside].reset_index(drop=True), x[inside.to_numpy()]


@pytest.mark.spark
def test_same_positions_as_crop(perturb_lake, source):
    src, _ = source
    g, _ = read(perturb_lake[0], "S-gain-4")
    assert (g.il.to_numpy() == src.il.to_numpy()).all()
    assert (g.xl.to_numpy() == src.xl.to_numpy()).all()


@pytest.mark.spark
def test_gain_4(perturb_lake, source):
    _, x = source
    _, y = read(perturb_lake[0], "S-gain-4")
    np.testing.assert_allclose(y, 4 * x, rtol=1e-6, atol=1e-3)


@pytest.mark.spark
def test_tgain_1_5(perturb_lake, source):
    _, x = source
    _, y = read(perturb_lake[0], "S-tgain-1.5")
    ns = x.shape[1]
    t = (np.arange(ns) + 0.5) * DT
    np.testing.assert_allclose(y, x * (t / (ns / 2 * DT)) ** 1.5, rtol=1e-5, atol=1e-3)


@pytest.mark.spark
def test_noise_level_and_rerun_identical(perturb_lake, source):
    lake, base = perturb_lake
    _, x = source
    _, y = read(lake, "S-noise-0.5")
    ratio = (y - x).std() / np.sqrt((x ** 2).mean())
    assert ratio == pytest.approx(0.5, abs=0.01)
    run(*base, "--kind", "noise", "--levels", "0.5")             # rerun overwrites the survey
    _, y2 = read(lake, "S-noise-0.5")
    assert np.array_equal(y, y2)


@pytest.mark.spark
def test_lowpass_20hz(perturb_lake, source):
    _, x = source
    _, y = read(perturb_lake[0], "S-lowpass-20")
    ns = x.shape[1]
    f = np.fft.rfftfreq(ns, DT)
    # Hann taper before the FFT: the FFT treats each trace as periodic, so the jump
    # between its last and first sample leaks energy into every frequency and would
    # hide what the filter did (0.6% "left" untapered vs ~1e-7 tapered).
    w = np.hanning(ns)
    e_x = np.abs(np.fft.rfft(x * w, axis=1)) ** 2
    e_y = np.abs(np.fft.rfft(y * w, axis=1)) ** 2
    assert e_y[:, f > 40].sum() / e_x[:, f > 40].sum() < 1e-3      # removed above 40 Hz
    assert e_y[:, f < 10].sum() / e_x[:, f < 10].sum() > 0.95      # kept below 10 Hz


@pytest.mark.spark
def test_catalog_tables(perturb_lake):
    lake, _ = perturb_lake
    assert len(pd.read_parquet(lake / "perturbations")) == 4
    assert len(pd.read_parquet(lake / "surveys")) == 5          # parent + 4 perturbed


@pytest.mark.spark          # no Spark job, but spark/perturb.py imports pyspark
def test_noise_is_per_trace_independent_of_batch_order():
    """Noise depends only on (seed, il, xl), not on the order traces arrive in."""
    from perturb import perturb_block
    blk = np.ones((3, 50))
    a = perturb_block(blk, [1, 2, 3], [7, 8, 9], "noise", 1.0, DT, 0, 1.0, 0)
    b = perturb_block(blk[::-1], [3, 2, 1], [9, 8, 7], "noise", 1.0, DT, 0, 1.0, 0)[::-1]
    assert np.array_equal(a, b)
