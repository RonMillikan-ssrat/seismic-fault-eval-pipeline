"""The KFP evaluation and monitoring pipelines compile to a spec (no cluster, no data needed)."""
import pytest

from helpers import REPO, run

pytest.importorskip("kfp")
import yaml  # noqa: E402  (installed with kfp)


@pytest.mark.parametrize("script", ["fault_eval_pipeline.py", "monitor_pipeline.py"])
def test_pipeline_compiles(tmp_path, script):
    out = tmp_path / "pipeline.yaml"
    run(REPO / "pipeline" / script, "compile", "--out", out)
    spec = yaml.safe_load(out.read_text())
    assert spec["root"]["dag"]["tasks"], "compiled pipeline has no tasks"
