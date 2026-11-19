"""QC SQL against a synthetic survey with planted problems (gaps, dead traces,
duplicates, a 3x gain patch). Expected answers come from the truth file in NumPy,
independently of the SQL; see check_qc.qc_checks().

Spark SQL (sql/*.sql) always runs; BigQuery (sql/bq/*.sql) runs with --bq-project,
after the QC survey has been loaded with gcp/load_bq.sh.
"""
import pytest

from check_qc import bq_query, qc_checks, spark_query

LABELS = [
    "01 missing positions (count)",
    "01 missing positions (exact set)",
    "02 duplicate traces",
    "03 dead traces",
    "04 jumps at gain-patch edges",
    "04 jumps elsewhere (false flags)",
    "06 crossline islands",
    "08 traces in stats == traces in file",
]


@pytest.fixture(scope="module")
def spark_results(qc_lake):
    truth, lake = qc_lake
    q, _ = spark_query(lake, "QC", "local[2]")        # session is shared; not stopped here
    return {label: (exp, got) for label, exp, got in qc_checks(q, truth, "QC")}


@pytest.fixture(scope="module")
def bq_results(request, qc_lake):
    truth, _ = qc_lake
    opt = request.config.getoption
    q = bq_query("QC", opt("--bq-project"), opt("--bq-dataset"), opt("--bq-location"))
    return {label: (exp, got) for label, exp, got in qc_checks(q, truth, "QC")}


@pytest.mark.spark
def test_spark_runs_every_check(spark_results):
    assert sorted(spark_results) == sorted(LABELS)


@pytest.mark.spark
@pytest.mark.parametrize("label", LABELS)
def test_qc_spark(spark_results, label):
    expected, got = spark_results[label]
    assert got == expected


@pytest.mark.bq
@pytest.mark.spark          # the truth file is generated together with the Spark lake
@pytest.mark.parametrize("label", LABELS)
def test_qc_bigquery(bq_results, label):
    expected, got = bq_results[label]
    assert got == expected
