"""experiment.py：Psi 與 LNRM 校準控制器、DFP 試次表、單人分析，全部用模擬受試者跑通。"""
import numpy as np
import pytest

from adaptivesft.ddm import dfp_ddm
from adaptivesft.experiment import (DFP_COLUMNS, LNRMCalibrator, PsiCalibrator, analyze_participant,
                                    dfp_trial_list, read_rows, report, write_rows)
from adaptivesft.models import d_numpy
from adaptivesft.psi import pm_function
from adaptivesft.race import lnrm_random


def test_psi_calibrator_gaussian_observer():
    rng = np.random.default_rng(1)
    cal = PsiCalibrator("colour", p_high=0.90, p_low=0.75)
    for _ in range(144):
        x = cal.next()
        cal.record(x, int(rng.uniform() < pm_function(x, 6.0, 15.0, 0.01)), rt=0.5)
    res = cal.finish()
    assert res.method == "psi" and res.n_trials == 144
    assert abs(res.params["alpha"] - 6) < 3 and abs(res.params["beta"] - 15) < 4
    assert res.low < res.high and res.in_range and not res.warnings
    with pytest.raises(ValueError):
        cal.record(cal.next() + 1.0, 1)


def test_psi_calibrator_custom_grid_warns_out_of_range():
    rng = np.random.default_rng(2)
    cal = PsiCalibrator("hue", x=(0, 10, 0.5), a=(0, 10, 0.5), b=(0.5, 8, 0.5), d=0.02, p_high=0.99, p_low=0.9)
    for _ in range(80):
        x = cal.next()
        cal.record(x, int(rng.uniform() < pm_function(x, 7.0, 6.0, 0.02)))
    res = cal.finish()
    assert res.high > 10 and not res.in_range and res.warnings


def test_lnrm_calibrator_roundtrip():
    rng = np.random.default_rng(3)
    levels = np.linspace(0.2, 3.0, 8)
    cal = LNRMCalibrator("dim", levels, n_per_level=40, link="ogival", h_targ=4.0, l_targ=1.0, seed=3,
                         tune=300, draws=300, chains=2, random_seed=3)
    assert cal.n_trials == 320 and sorted(set(cal.plan)) == sorted(levels.tolist())
    truth = dict(slope=2.0, midpoint=1.5)
    for _ in range(cal.n_trials):
        x = cal.next()
        d = d_numpy("ogival", x, truth, 10.0)
        rt, c = lnrm_random(np.array([d]), 1.5, 0.6, 0.12, rng=rng)
        cal.record(x, int(c[0]), float(rt[0]))
    with pytest.raises(StopIteration):
        cal.next()
    res = cal.finish()
    assert res.method == "lnrm_ogival" and np.isfinite(res.high) and np.isfinite(res.low) and res.low < res.high
    assert "slope" in res.params and res.params["n_used"] <= 320
    with pytest.raises(ValueError):
        LNRMCalibrator("dim", levels, 5)


def test_dfp_trial_list_and_analysis(tmp_path):
    rows = dfp_trial_list("S01", high1=3.0, low1=1.0, high2=3.0, low2=1.0, n_per_cell=60, blocks=2, seed=4)
    assert len(rows) == 480 and set(rows[0]) == set(DFP_COLUMNS)
    counts = {}
    for r in rows:
        counts[(r["channel1"], r["channel2"])] = counts.get((r["channel1"], r["channel2"]), 0) + 1
    assert all(v == 120 for v in counts.values()) and rows[0]["c1_level"] in (3.0, 1.0)
    rng = np.random.default_rng(5)
    for r in rows:                       # 用 PAR-OR 的 DDM 填答案（分離大，SIC 應判 ParallelOR）
        rt, cr = dfp_ddm(1, r["c1_level"], r["c2_level"], 3.0, 0.1, 0.2, "PAR", "OR", rng=rng)
        r["rt"], r["correct"] = float(rt[0]), int(cr[0])
    path = tmp_path / "S01.csv"
    write_rows(rows, path)
    back = read_rows(path)
    assert len(back) == 480 and back[0]["subject"] == "S01"
    res = analyze_participant(back)
    r = res["DFP"]
    assert r["classification"]["Predicted_by"] == "ParallelOR" and r["Dplus"][1] < 0.05
    assert "ParallelOR" in report(res)


def test_analysis_refuses_tiny_cells():
    rows = dfp_trial_list("S02", 3.0, 1.0, 3.0, 1.0, n_per_cell=5, seed=6)
    for r in rows:
        r["rt"], r["correct"] = 0.5, 1
    assert "error" in analyze_participant(rows)["DFP"]
