"""models.py：Op 輸出 = numba；四種 link 的小樣本擬合都能跑完並回收到合理範圍。"""
import numpy as np
import pytest

from adaptivesft.models import LNRM_PointwiseOp, d_numpy, fit_lnrm, make_data
from adaptivesft.race import lnrm_pointwise_loglik, lnrm_random

TRUE = dict(mu=1.5, varZ=0.6, psi=0.12)


def _simulate(link, n=300, seed=3, **p):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 3.0, n)
    d = d_numpy(link, x, p)
    rt, correct = lnrm_random(d, TRUE["mu"], TRUE["varZ"], TRUE["psi"], rng=rng)
    return make_data(x, rt, correct), d


def test_op_equals_numba():
    data, d = _simulate("quadratic", alpha=0.8, alpha2=-0.15)
    op = LNRM_PointwiseOp()
    got = op(data[:, 0], data[:, 1].astype("int32"), d, 1.5, 0.6, 0.12).eval()
    want = lnrm_pointwise_loglik(data[:, 0], data[:, 1].astype(np.int32), d, 1.5, 0.6, 0.12)
    assert np.allclose(got, want, atol=1e-12, rtol=0)


def test_make_data_rejects_nonpositive_rt():
    with pytest.raises(ValueError):
        make_data([1, 2], [0.5, 0.0], [1, 0])


@pytest.mark.parametrize("link,params", [
    ("quadratic", dict(alpha=0.8, alpha2=-0.15)),
    ("linear", dict(alpha=0.5)),
    ("ogival", dict(slope=2.0, midpoint=1.5)),
    ("none", dict(alpha=0.7)),
])
def test_fit_small_smoke(link, params):
    data, _ = _simulate(link, n=300, **params)
    tr = fit_lnrm(data, link=link, tune=400, draws=400, chains=4, random_seed=1)
    assert tr.attrs["link"] == link
    post = tr.posterior
    for k, v in params.items():
        m = float(post[k].values.mean())
        assert abs(m - v) < 0.6, f"{link}: {k} 後驗平均 {m:.3f} 離真值 {v} 太遠"
    assert 0 < float(post["psi"].values.mean()) < data[:, 0].min()


def test_fix_params_linear_equals_link_linear():
    data, _ = _simulate("linear", n=200, alpha=0.5)
    tr = fit_lnrm(data, link="quadratic", fix_params={"alpha2": 0.0}, tune=200, draws=200, chains=2)
    assert "alpha2" not in tr.posterior
    assert tr.attrs["fix_params"]["alpha2"] == 0.0
