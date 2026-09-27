"""race.py：numba 逐題 logp 對 scipy 封閉式；plognormalrace 對蒙地卡羅；密度總積分 = 1。"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from adaptivesft.race import dlognormalrace, lnrm_pointwise_loglik, lnrm_random, plognormalrace

MU, VARZ, PSI = 1.5, 0.6, 0.12


def test_pointwise_matches_scipy_closed_form():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.0, 2.0, 400)
    rt, correct = lnrm_random(d, MU, VARZ, PSI, rng=rng)
    got = lnrm_pointwise_loglik(rt, correct.astype(np.int32), d, MU, VARZ, PSI)
    want = np.empty_like(got)
    for i in range(len(rt)):
        mu2 = np.array([MU - d[i], MU + d[i]])
        m = 0 if correct[i] == 1 else 1
        want[i] = dlognormalrace(rt[i], m, PSI, mu2, [VARZ, VARZ], log=True)
    assert np.allclose(got, want, atol=1e-10, rtol=0)


def test_density_integrates_to_one_and_matches_closed_form_accuracy():
    d = 0.8
    mu2 = np.array([MU - d, MU + d])
    sd = [VARZ, VARZ]
    total = sum(quad(dlognormalrace, PSI, np.inf, args=(m, PSI, mu2, sd))[0] for m in (0, 1))
    p0 = quad(dlognormalrace, PSI, np.inf, args=(0, PSI, mu2, sd))[0]
    assert abs(total - 1.0) < 1e-6
    assert abs(p0 - norm.cdf(2 * d / (VARZ * np.sqrt(2)))) < 1e-6    # P(correct) = Φ(2d / (σ√2))


def test_plognormalrace_matches_monte_carlo():
    rng = np.random.default_rng(1)
    d = 0.8
    mu2 = np.array([MU - d, MU + d])
    N = 200_000
    t0 = PSI + rng.lognormal(mu2[0], VARZ, N)
    t1 = PSI + rng.lognormal(mu2[1], VARZ, N)
    xs = np.array([0.5, 1.0, 2.0, 4.0])
    mc = np.array([np.mean((t0 < t1) & (t0 <= x)) for x in xs])
    got = plognormalrace(xs, 0, PSI, mu2, [VARZ, VARZ])
    assert np.allclose(got, mc, atol=0.005)
    assert np.all(plognormalrace([PSI - 0.01, PSI], 0, PSI, mu2, [VARZ, VARZ]) == 0.0)
