"""
Parity with the original R code, and the conversions between the two branches' quantities.

Every oracle under tests/data/*_r_oracle.json was produced by running the ORIGINAL code in R
(sft::sic, diffIRT::simdiffT, the Psi loop of psiSimulation_functions.R copied line by line);
see the make_*_oracle.R scripts. These tests never call R.
    scripts/demo_parity.py prints the same comparisons as tables.
"""
import json
import os

import numpy as np
import pytest

from adaptivesft.ddm import ddm_p_correct, simdiffT
from adaptivesft.psi import GRIDS, inv_pm_function, make_psi, pm_function, salience_levels
from adaptivesft.salience import accuracy_to_targ, targ_to_accuracy
from adaptivesft.sic import sic

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} missing: run the matching make_*_oracle.R")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 1. sft::sic
def test_sic_matches_r_bit_for_bit():
    o = _load("sic_r_oracle.json")
    s = sic(**{k: np.asarray(o["cells"][k], float) for k in ("HH", "HL", "LH", "LL")})
    assert np.max(np.abs(np.asarray(o["SIC_values"]) - s["SIC"][1])) < 1e-12
    assert abs(s["SICtest"]["positive"][0] - o["Dplus"]) < 1e-12
    assert abs(s["SICtest"]["positive"][1] - o["p_Dplus"]) < 1e-12
    assert abs(s["MICtest"]["statistic"] - o["MIC"]) < 1e-12
    assert abs(s["MICtest"]["p_value"] - o["p_MIC_art"]) < 1e-8


# ---------------------------------------------------------------- 2. diffIRT::simdiffT
def test_simdiffT_matches_r_distributionally():
    o = _load("ddm_r_oracle.json")
    rng = np.random.default_rng(1)
    for name, c in o.items():
        rt, x = simdiffT(20000, c["a"], c["mv"], c["sv"], c["ter"], rng=rng)
        assert abs(x.mean() - c["p_upper"]) < 0.012, name                       # two MC runs of 20000
        assert abs(rt.mean() - c["rt_mean"]) < 0.03 * c["rt_mean"], name
        assert abs(np.median(rt) - c["rt_q"][2]) < 0.03 * c["rt_q"][2], name
        assert abs(ddm_p_correct(c["a"], c["mv"]) - c["closed_form_p"]) < 1e-12   # simdiffT.r:6 exactly


# ---------------------------------------------------------------- 3. the Psi loop
def _run_python_psi(o):
    u = np.loadtxt(os.path.join(DATA, "psi_oracle_input.csv"), delimiter=",", skiprows=1)[:, 1]
    sim = o["sim"]
    psi = make_psi("colour")
    intens, resp, alpha, beta = [], [], [], []
    for t in range(len(o["intensity"])):
        intens.append(psi.next_intensity)
        r = int(u[t] < pm_function(psi.next_intensity, sim["a"], sim["b"], sim["d"]))
        resp.append(r)
        psi.update(r)
        a_hat, b_hat = psi.estimate()
        alpha.append(a_hat)
        beta.append(b_hat)
    return np.asarray(intens), np.asarray(resp), np.asarray(alpha), np.asarray(beta)


def test_psi_loop_matches_r_trial_by_trial():
    o = _load("psi_r_oracle.json")
    intens, resp, alpha, beta = _run_python_psi(o)
    assert np.array_equal(intens, np.asarray(o["intensity"]))          # same stimulus chosen every trial
    assert np.array_equal(resp, np.asarray(o["response"]))             # hence same responses
    assert np.max(np.abs(alpha - np.asarray(o["alpha"]))) < 1e-10      # same posterior means
    assert np.max(np.abs(beta - np.asarray(o["beta"]))) < 1e-10


def test_psi_salience_levels_match_r_inv_pm_function():
    o = _load("psi_r_oracle.json")
    _, _, alpha, beta = _run_python_psi(o)
    (H, L), _ = salience_levels(alpha[-1], beta[-1], o["sim"]["d"], [0.99, 0.90])
    assert abs(H - o["high"]) < 1e-10 and abs(L - o["low"]) < 1e-10


# ---------------------------------------------------------------- 4. conversions
def test_accuracy_and_separation_targets_are_inverses():
    for varZ in (0.6, 0.95):
        for p in (0.99, 0.90, 0.75):
            assert abs(targ_to_accuracy(accuracy_to_targ(p, varZ), varZ) - p) < 1e-12
    # Houpt's h_targ = 8.0 is ceiling accuracy at any plausible varZ; l_targ = 1.3 is 0.83-0.94
    assert targ_to_accuracy(8.0, 0.95) > 0.9999
    assert 0.83 < targ_to_accuracy(1.3, 0.95) < 0.84 and 0.93 < targ_to_accuracy(1.3, 0.6) < 0.94


def test_two_readings_of_a_reproduce_the_original_numbers():
    a, v, thres50, xmax = 1.45, 1.6, 6.0, 50.0                       # psiSimulation_functions.R:15, 93-97
    sc = lambda x: (x - thres50) / (xmax - thres50)
    # under the 'threshold' reading (26MAR2019.R:117) the range top is the 99% point -> the 2019 script's 50
    assert abs(ddm_p_correct(a, sc(50.0) * v, a_is_separation=False) - 0.9904) < 5e-4
    # under diffIRT's reading the range top is only 91%, and 99% needs x = 93 -> the 2018 script's 101.6
    assert abs(ddm_p_correct(a, sc(50.0) * v, a_is_separation=True) - 0.9105) < 5e-4
    x99_sep = thres50 + np.log(0.99 / 0.01) / a / v * (xmax - thres50)
    assert 92 < x99_sep < 94
    x99_thr = thres50 + np.log(0.99 / 0.01) / (2 * a) / v * (xmax - thres50)
    assert 49 < x99_thr < 50.5


def test_r_beta_grids_contain_the_threshold_reading_only():
    for dim, beta_sep, beta_thr in (("colour", 31.2, 15.8), ("orientation", 18.9, 9.65)):
        lo, hi, _ = GRIDS[dim]["b"]
        assert lo <= beta_thr <= hi
    lo, hi, _ = GRIDS["orientation"]["b"]
    assert not lo <= 18.9 <= hi                                       # separation reading: outside R's grid


def test_psi_estimate_maps_to_ddm_accuracy_under_both_readings():
    o = _load("psi_r_oracle.json")
    a_hat, b_hat, d = o["alpha"][-1], o["beta"][-1], o["sim"]["d"]
    x_H = float(inv_pm_function(0.99, a_hat, b_hat, d))
    sc = (x_H - 6.0) / (50.0 - 6.0)
    p_thr = ddm_p_correct(1.45, sc * 1.6, a_is_separation=False)
    p_sep = ddm_p_correct(1.45, sc * 1.6, a_is_separation=True)
    assert p_thr > p_sep                                              # same x, the two readings disagree
    assert 0 < p_sep < 1 and 0 < p_thr < 1
