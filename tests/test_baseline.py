"""Known-answer tests for baseline/coherence.py and evaluation/compare_models.py.

Semblance has exact answers on simple synthetic cubes:
  * identical traces                         -> semblance 1, attribute 0
  * reflectors flipped in polarity across a vertical fault plane, 3-trace window:
    a window holding 2 traces of one polarity and 1 of the other has
    mean = a/3, so semblance = (a/3)^2 / a^2 = 1/9 and attribute = 8/9 exactly,
    on the two crosslines next to the plane, and 0 everywhere else
  * independent noise traces                 -> semblance near 1/N (N = 9 traces)
(Converted from tests/check_baseline.py; same cases, same expected values.)
"""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import coherence as coh
import fault_metrics as fm
import make_synthetic_f3
from helpers import REPO, at, run

N_IL, N_XL, N_T = 12, 40, 120
T = np.arange(N_T)
WAVE = np.sin(2 * np.pi * T / 20.0).astype(np.float32)
C = 20                                       # fault plane between xl 19 and 20


def flat_cube():
    return np.broadcast_to(WAVE, (N_IL, N_XL, N_T)).copy()


def test_identical_traces_attribute_zero():
    assert coh.fault_attribute(flat_cube()).max() == pytest.approx(0.0, abs=1e-6)


@pytest.fixture(scope="module")
def fault_attr():
    cube = flat_cube()
    cube[:, C:, :] *= -1
    return coh.fault_attribute(cube, n_il=3, n_xl=3, half_t=4)


@pytest.mark.parametrize("xl", [C - 1, C])
def test_fault_plane_attribute_is_8_9(fault_attr, xl):
    assert fault_attr[:, xl, :].mean() == pytest.approx(8 / 9, abs=1e-6)


def test_fault_plane_attribute_zero_elsewhere(fault_attr):
    assert np.delete(fault_attr, [C - 1, C], axis=1).max() == pytest.approx(0.0, abs=1e-6)


def test_fault_plane_scores_f1_1(fault_attr):
    """Scored as a fault map: label on xl 20, rows = time, so sections are (il, t, xl)."""
    label = np.zeros((N_IL, N_T, N_XL), bool)
    label[:, :, C] = True
    cc = fm.curve_counts(fault_attr.transpose(0, 2, 1), label, tol=1)
    assert at(cc, 0.5, fm.prf)[2] == pytest.approx(1.0, abs=1e-6)


def test_white_noise_semblance_near_one_ninth():
    noise = np.random.default_rng(0).standard_normal((N_IL, N_XL, N_T)).astype(np.float32)
    assert coh.semblance(noise, n_il=3, n_xl=3, half_t=4).mean() == pytest.approx(1 / 9, abs=0.03)


def test_attribute_rises_with_dip():
    """No dip steering: a dipping reflector with no fault still raises the attribute."""
    means = []
    for dip in [0.0, 0.5, 1.0, 2.0]:                    # samples of shift per crossline
        shifted = np.stack([np.sin(2 * np.pi * (T - dip * x) / 20.0) for x in range(N_XL)])
        cube = np.broadcast_to(shifted, (N_IL, N_XL, N_T)).astype(np.float32)
        means.append(float(coh.fault_attribute(cube)[:, 2:-2, 5:-5].mean()))
    assert np.all(np.diff(means) > 0), means


@pytest.fixture(scope="module")
def tiny_lake(tmp_path_factory):
    """A traces table in the lake layout, with a missing position and a duplicate whose
    later copy must be ignored, split over two files like Spark output."""
    root = tmp_path_factory.mktemp("tiny_lake")
    part = root / "traces" / "survey=T"
    part.mkdir(parents=True)
    il_v, xl_v, data = [], [], []
    for il in range(100, 106):
        for xl in range(300, 310):
            if (il, xl) == (102, 305):
                continue                                  # missing position
            il_v.append(il); xl_v.append(xl)
            data.append(np.full(16, il * 1000 + xl, np.float32))
    il_v.append(103); xl_v.append(304); data.append(np.full(16, -1.0, np.float32))  # duplicate
    seq = np.arange(len(il_v))
    for k, (a, b) in enumerate([(0, 30), (30, len(il_v))]):
        pq.write_table(pa.table({"trace_seq": pa.array(seq[a:b], pa.int64()),
                                 "il": pa.array(il_v[a:b], pa.int32()),
                                 "xl": pa.array(xl_v[a:b], pa.int32()),
                                 "samples": pa.array([d.tolist() for d in data[a:b]],
                                                     pa.list_(pa.float32()))}),
                       part / f"part-{k:05d}.parquet")
    return coh.load_cube(root, "T", (101, 105), (302, 308), (2, 10))


def test_loader_shape_and_values(tiny_lake):
    cube, _ = tiny_lake
    assert cube.shape == (4, 6, 8)
    assert cube[3, 1, 0] == 104303                      # il 104, xl 303


def test_loader_missing_position_is_zero(tiny_lake):
    cube, info = tiny_lake
    assert cube[1, 3].max() == 0.0                      # (102, 305)
    assert info["missing"] == 1


def test_loader_duplicate_keeps_first_copy(tiny_lake):
    cube, info = tiny_lake
    assert cube[2, 2, 0] == 103304                      # (103, 304)
    assert info["duplicates"] == 1


# Gate: a model compared with itself must FAIL (difference exactly 0); a clearly
# worse reference must make it PASS. Synthetic F3-layout files in a temp folder.
@pytest.fixture(scope="module")
def f3sim(tmp_path_factory):
    sim = make_synthetic_f3.make(tmp_path_factory.mktemp("f3") / "f3sim")
    worse = sim / "pred_worse"
    worse.mkdir()
    p_good = np.load(sim / "pred" / "prob_crop.npy")
    rng = np.random.default_rng(5)
    np.save(worse / "prob_crop.npy",
            np.clip(0.6 * p_good + 0.5 * rng.random(p_good.shape), 0, 1).astype(np.float32))
    (worse / "crop.json").write_text((sim / "pred" / "crop.json").read_text())
    return sim


def run_gate(sim, ref, name):
    out = sim / f"gate_{name}"
    run(REPO / "evaluation" / "compare_models.py", "--labels-dir", sim / "inlines",
        "--pred-a", sim / "pred", "--pred-b", ref, "--block", "2", "--n-boot", "1000", "--out", out)
    return json.loads((out / "gate.json").read_text())


def test_gate_fails_against_itself(f3sim):
    g = run_gate(f3sim, f3sim / "pred", "self")
    assert g["passed"] is False
    assert g["diff"] == pytest.approx(0.0, abs=1e-12)


def test_gate_passes_against_worse_model(f3sim):
    assert run_gate(f3sim, f3sim / "pred_worse", "worse")["passed"] is True
