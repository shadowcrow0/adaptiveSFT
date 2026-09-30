"""
lnrm2.stan fitted by rstan (oracle, made on a machine with Stan: tests/data/make_lnrm_oracle.R)
vs adaptivesft.fit_lnrm(link="quadratic") on the same 1000 trials.

Different samplers, so the comparison is statistical: posterior means within a few posterior SDs
of Monte-Carlo error, quantiles within tolerance, and the R-side salience inversion
(adaptiveSFT_functions.R:229-232, 'all draws' semantics) within 3% of the Python one.
Skipped until tests/data/lnrm_stan_oracle.json exists.
"""
import json
import os

import numpy as np
import pytest

from adaptivesft.models import fit_lnrm, make_data
from adaptivesft.salience import find_salience_polynomial

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ORACLE = os.path.join(DATA, "lnrm_stan_oracle.json")


@pytest.mark.skipif(not os.path.exists(ORACLE), reason="no Stan oracle: run tests/data/make_lnrm_oracle.R on a machine with rstan")
def test_pymc_posterior_matches_stan_posterior():
    with open(ORACLE, encoding="utf-8") as f:
        o = json.load(f)
    d = np.loadtxt(os.path.join(DATA, "lnrm_oracle_input.csv"), delimiter=",", skiprows=1)
    data = make_data(d[:, 0], d[:, 1], d[:, 2])
    tr = fit_lnrm(data, link="quadratic", tune=2000, draws=2000, chains=4, random_seed=20260930)
    print(f"\n{'param':8}{'Stan mean':>12}{'PyMC mean':>12}{'Stan sd':>10}{'PyMC sd':>10}{'|diff|/sd':>11}")
    for row in o["summary"]:
        name = row["name"]
        v = tr.posterior[name].values.ravel()
        diff_sd = abs(v.mean() - row["mean"]) / row["sd"]
        print(f"{name:8}{row['mean']:>12.4f}{v.mean():>12.4f}{row['sd']:>10.4f}{v.std():>10.4f}{diff_sd:>11.2f}")
        assert diff_sd < 0.35, f"{name}: posterior means differ by {diff_sd:.2f} posterior SDs"
        assert abs(v.std() - row["sd"]) / row["sd"] < 0.30, f"{name}: posterior SDs differ by > 30%"
        for q, key in ((0.05, "q05"), (0.5, "q50"), (0.95, "q95")):
            assert abs(np.quantile(v, q) - row[key]) < 0.5 * row["sd"], f"{name}: {key}"
    res = find_salience_polynomial(tr, o["salience"]["h_targ"], o["salience"]["l_targ"], alpha2_rule="all_draws")
    for name in ("high", "low"):
        r_val, py_val = o["salience"][name], res[name]["intensity"]
        print(f"salience {name}: R {r_val:.4f}  Python {py_val:.4f}")
        assert abs(py_val - r_val) / abs(r_val) < 0.03
