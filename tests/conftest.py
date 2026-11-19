"""Shared pytest setup.

Imports: the repo's code lives in plain folders (evaluation/, baseline/, ...), not a
package, so they are put on sys.path here, the same way the old check_*.py scripts did.

Fixtures: each synthetic lake is built once per test session and shared by every test
that needs it. The lakes use the same parameters the original check scripts used, so
the expected values carry over unchanged.

Markers (see pytest.ini):
  spark   skipped automatically if PySpark or Java is missing;  -m "not spark" for the fast set
  bq      skipped unless --bq-project is given
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for sub in ["evaluation", "baseline", "monitoring", "spark", "gcp", "tests"]:
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# Spark workers must use this interpreter, not whatever "python" is on PATH.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from helpers import CROP, MASTER, run  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--bq-project", default=None, help="GCP project for the BigQuery QC tests")
    parser.addoption("--bq-dataset", default="seismic")
    parser.addoption("--bq-location", default=None)


def _spark_available():
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return "PySpark not installed (pip install -r requirements-spark.txt)"
    if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
        return "Java not found (PySpark 4 needs Java 17 or 21)"
    return None


def pytest_collection_modifyitems(config, items):
    spark_reason = _spark_available()
    bq = config.getoption("--bq-project")
    for item in items:
        if spark_reason and "spark" in item.keywords:
            item.add_marker(pytest.mark.skip(reason=spark_reason))
        if not bq and "bq" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="BigQuery tests need --bq-project"))


def _segy_to_lake(tmp, name, survey, *segy_args):
    sgy = tmp / f"{name}.sgy"
    run(REPO / "tests/make_synthetic_segy.py", sgy, *segy_args)
    run(REPO / "spark/segy_to_parquet.py", sgy, "--survey", survey, "--out", tmp / "lake",
        "--master", MASTER)
    return sgy


@pytest.fixture(scope="session")
def roundtrip_lakes(tmp_path_factory):
    """One small SEG-Y per sample format (1 = IBM, 5 = IEEE), each converted to a lake."""
    out = {}
    for fmt in (1, 5):
        tmp = tmp_path_factory.mktemp(f"roundtrip_fmt{fmt}")
        sgy = _segy_to_lake(tmp, "syn", "SYN", "--format", str(fmt))
        out[fmt] = (Path(str(sgy) + ".truth.npz"), tmp / "lake" / "traces" / "survey=SYN")
    return out


@pytest.fixture(scope="session")
def qc_lake(tmp_path_factory):
    """Survey with planted problems: gaps, dead traces, duplicates, a 3x gain patch.
    Same parameters as the README's QC self-test."""
    tmp = tmp_path_factory.mktemp("qc")
    sgy = _segy_to_lake(tmp, "qc", "QC", "--il", "100,299", "--xl", "300,599", "--ns", "462",
                        "--dups", "25", "--gain-patch", "150,169,400,439,3")
    run(REPO / "spark/trace_attributes.py", "--root", tmp / "lake", "--survey", "QC",
        "--master", MASTER, "--variants", "pandas", "--repeat", "1")
    return Path(str(sgy) + ".truth.npz"), tmp / "lake"


@pytest.fixture(scope="session")
def perturb_lake(tmp_path_factory):
    """Lake with 2% missing positions, perturbed four ways on the crop."""
    tmp = tmp_path_factory.mktemp("perturb")
    _segy_to_lake(tmp, "s", "S", "--il", "100,139", "--xl", "300,349", "--ns", "200",
                  "--dead", "0", "--missing", "0.02")
    lake = tmp / "lake"
    base = [REPO / "spark/perturb.py", "--root", lake, "--survey", "S", "--master", MASTER, *CROP]
    for kind, level in [("gain", "4"), ("tgain", "1.5"), ("noise", "0.5"), ("lowpass", "20")]:
        run(*base, "--kind", kind, "--levels", level)
    return lake, base


@pytest.fixture(scope="session")
def drift_lake(tmp_path_factory):
    """Lake with no gaps, three perturbations, and trace_stats for all four surveys.
    Shared by the drift-signal and monitoring tests (both used exactly this setup)."""
    tmp = tmp_path_factory.mktemp("drift")
    _segy_to_lake(tmp, "s", "S", "--il", "100,139", "--xl", "300,349", "--ns", "200",
                  "--dead", "0", "--missing", "0")
    lake = tmp / "lake"
    for kind, level in [("gain", "4"), ("noise", "0.5"), ("lowpass", "20")]:
        run(REPO / "spark/perturb.py", "--root", lake, "--survey", "S", "--kind", kind,
            "--levels", level, "--master", MASTER, *CROP)
    for s in ["S", "S-gain-4", "S-noise-0.5", "S-lowpass-20"]:
        run(REPO / "spark/trace_attributes.py", "--root", lake, "--survey", s, "--master", MASTER,
            "--variants", "pandas", "--repeat", "1")
    return lake
