"""models_v2：PyMC 內建 LogNormal logp/logcdf 拼出來的賽跑 likelihood，對 v1 與 Stan。"""
import json
import os

import numpy as np
import pytest

from adaptivesft.models import _race_logp_pt, fit_lnrm, make_data
from adaptivesft.models_v2 import build_model, fit_lnrm_v2, race_logp
from adaptivesft.race import lnrm_random
from adaptivesft.salience import find_salience

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ORACLE = os.path.join(DATA, "lnrm_stan_oracle.json")


def _sim(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.choice([0.5, 1.0, 1.5, 2.0, 2.5], n)
    d = 0.8 * x - 0.15 * x ** 2
    rt, correct = lnrm_random(d, 1.5, 0.6, 0.12, rng)
    return make_data(x, rt, correct), d


def test_v2_logp_equals_v1_logp():
    data, d = _sim()
    rt, correct = data[:, 0], data[:, 1]
    for mu, sigma, psi in ((1.5, 0.6, 0.12), (0.5, 0.3, 0.1), (2.0, 1.0, 0.0)):
        v1 = _race_logp_pt(rt, correct.astype("int32"), d, mu, sigma, psi).eval()
        v2 = race_logp(rt, mu, d, sigma, psi, correct).eval()
        assert np.allclose(v1, v2, atol=1e-6, rtol=1e-7)     # erfc 與 log1mexp(logcdf) 兩條路；尾端 logp ≈ −50 時差 1e−6


def test_v2_fit_recovers_and_matches_v1():
    data, _ = _sim(n=600, seed=1)
    tr2 = fit_lnrm_v2(data, tune=400, draws=400, chains=2, random_seed=2)
    assert tr2.attrs["implementation"] == "v2" and "obs" in tr2.log_likelihood
    assert int(tr2.sample_stats["diverging"].sum()) < 10
    tr1 = fit_lnrm(data, tune=400, draws=400, chains=2, random_seed=2)
    for k in ("mu", "alpha", "alpha2", "varZ", "psi"):
        m1, m2, s = tr1.posterior[k].values.mean(), tr2.posterior[k].values.mean(), tr1.posterior[k].values.std()
        assert abs(m1 - m2) < 0.5 * s + 0.02, k
    res = find_salience(tr2, h_targ=1.6, l_targ=0.6)          # salience.py 讀 attrs，v2 的 attrs 與 v1 相同
    assert np.isfinite(res["high"]["intensity"]) and res["low"]["intensity"] < res["high"]["intensity"]


def test_v2_posterior_predictive_runs():
    import pymc as pm
    data, _ = _sim(n=200, seed=3)
    tr = fit_lnrm_v2(data, tune=200, draws=200, chains=1, random_seed=3)
    model, _ = build_model(data)
    with model:
        pp = pm.sample_posterior_predictive(tr, var_names=["obs"], random_seed=3, progressbar=False)
    sim = pp.posterior_predictive["obs"].values
    assert sim.shape[-1] == 200 and np.all(sim > 0)
    assert abs(np.median(sim) - np.median(data[:, 0])) < 0.5


@pytest.mark.skipif(not os.path.exists(ORACLE), reason="no Stan oracle")
def test_v2_posterior_matches_stan():
    with open(ORACLE, encoding="utf-8") as f:
        o = json.load(f)
    d = np.loadtxt(os.path.join(DATA, "lnrm_oracle_input.csv"), delimiter=",", skiprows=1)
    tr = fit_lnrm_v2(make_data(d[:, 0], d[:, 1], d[:, 2]), tune=2000, draws=2000, chains=4, random_seed=20260930)
    for row in o["summary"]:
        v = tr.posterior[row["name"]].values.ravel()
        assert abs(v.mean() - row["mean"]) / row["sd"] < 0.35, row["name"]
        assert abs(v.std() - row["sd"]) / row["sd"] < 0.30, row["name"]
