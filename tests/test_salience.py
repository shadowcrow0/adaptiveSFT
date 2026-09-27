"""salience.py：反解回代成立；alpha2_rule 五種語意；ogival 反解；正確率介面。用假 trace，不跑 PyMC。"""
import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm

from adaptivesft.models import d_numpy
from adaptivesft.salience import (
    ALPHA2_RULES, accuracy_to_targ, find_salience, find_salience_ogival,
    find_salience_polynomial, summarize, targ_to_accuracy,
)


class FakeTrace:
    def __init__(self, link, L=10.0, fix_params=None, **draws):
        self.attrs = {"link": link, "L": L, "fix_params": fix_params or {}, "min_rt": 0.2}
        self.posterior = xr.Dataset({k: (("chain", "draw"), np.asarray(v, float).reshape(1, -1))
                                     for k, v in draws.items()})


def test_quadratic_inverse_roundtrip():
    tr = FakeTrace("quadratic", alpha=[0.8] * 5, alpha2=[-0.15] * 5, varZ=[0.6] * 5)
    res = find_salience_polynomial(tr, h_targ=1.6, l_targ=0.6)
    for name, targ in (("high", 1.6), ("low", 0.6)):
        x = res[name]["intensity"]
        assert abs(2 * d_numpy("quadratic", x, dict(alpha=0.8, alpha2=-0.15)) - targ) < 1e-9   # z2 − z1 = 2d = targ
        assert res[name]["dropped"] == 0.0


def test_alpha2_rules():
    alpha = np.full(10, 0.8)
    alpha2 = np.array([-0.15] * 7 + [0.02] * 3)          # 三成 draw 是正的
    tr = FakeTrace("quadratic", alpha=alpha, alpha2=alpha2)
    all_draws = find_salience_polynomial(tr, 1.6, 0.6, "all_draws")
    first = find_salience_polynomial(tr, 1.6, 0.6, "first_draw")   # 第一筆 < 0 → 與 all_draws 完全相同
    assert first["high"]["intensity"] == all_draws["high"]["intensity"]
    filt = find_salience_polynomial(tr, 1.6, 0.6, "filter")
    assert filt["high"]["intensity"] != all_draws["high"]["intensity"]
    assert any("filter" in w for w in filt["warnings"])
    with pytest.raises(ValueError):
        find_salience_polynomial(tr, 1.6, 0.6, "all")
    with pytest.raises(ValueError):
        find_salience_polynomial(FakeTrace("quadratic", alpha=alpha, alpha2=alpha2[::-1]), 1.6, 0.6, "first_draw")
    mean_ok = find_salience_polynomial(tr, 1.6, 0.6, "mean")
    assert mean_ok["high"]["intensity"] == all_draws["high"]["intensity"]
    with pytest.raises(ValueError):
        find_salience_polynomial(tr, 1.6, 0.6, "nonsense")
    assert set(ALPHA2_RULES) == {"all_draws", "first_draw", "all", "mean", "filter"}


def test_dropped_draws_are_reported():
    # alpha2 < 0 但目標太高 → 判別式 < 0 → 舊 R 的 na.rm 會靜默丟掉；這裡要回報
    # d(x) = 0.3x − 0.5x² 的最大值 0.045 → 2d 最大 0.09：targ 5.0 不可達、0.05 可達
    tr = FakeTrace("quadratic", alpha=np.full(20, 0.3), alpha2=np.full(20, -0.5))
    res = find_salience_polynomial(tr, h_targ=5.0, l_targ=0.05)
    assert res["high"]["dropped"] == 1.0 and np.isnan(res["high"]["intensity"])
    assert res["low"]["dropped"] == 0.0
    assert any("high" in w for w in res["warnings"])


def test_linear_via_fix_params():
    tr = FakeTrace("quadratic", fix_params={"alpha2": 0.0}, alpha=[0.5] * 4, varZ=[0.6] * 4)
    res = find_salience_polynomial(tr, 1.0, 0.4)
    assert abs(res["high"]["intensity"] - 1.0) < 1e-12 and res["rule"] == "linear"


def test_ogival_inverse_matches_r_formula():
    slope, mid, L = 2.0, 1.5, 10.0
    tr = FakeTrace("ogival", L=L, slope=[slope] * 3, midpoint=[mid] * 3, varZ=[0.6] * 3)
    res = find_salience_ogival(tr, h_targ=8.0, l_targ=1.3)
    from scipy.special import logit
    assert abs(res["high"]["intensity"] - (logit(8.0 / L) / slope + mid)) < 1e-12     # R :195
    assert abs(res["low"]["intensity"] - (logit(1.3 / L) / slope + mid)) < 1e-12      # R :194
    bad = find_salience_ogival(tr, h_targ=12.0, l_targ=1.3)
    assert np.isnan(bad["high"]["intensity"]) and any("到不了" in w for w in bad["warnings"])


def test_accuracy_interface():
    varZ = 0.6
    assert abs(targ_to_accuracy(accuracy_to_targ(0.9, varZ), varZ) - 0.9) < 1e-12
    tr = FakeTrace("quadratic", alpha=[0.8] * 6, alpha2=[-0.15] * 6, varZ=[varZ] * 6)
    res = find_salience(tr, acc_high=0.9, acc_low=0.7)
    x = res["high"]["intensity"]
    acc = norm.cdf(2 * d_numpy("quadratic", x, dict(alpha=0.8, alpha2=-0.15)) / (varZ * np.sqrt(2)))
    assert abs(acc - 0.9) < 1e-9
    res2 = find_salience(tr, h_targ=1.6, l_targ=0.6)
    assert "implied_accuracy" in res2["high"]
    assert "targ" in summarize(res2)
    with pytest.raises(ValueError):
        find_salience(tr)
