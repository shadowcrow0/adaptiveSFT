"""psi.py：累積常態受試者的 α/β 回復；salience_levels 與 R 的 inv.pm.function 一致並警告範圍外。"""
import numpy as np
from scipy.stats import norm

from adaptivesft.psi import inv_pm_function, make_psi, pm_function, salience_levels


def test_recovery_gaussian_observer():
    rng = np.random.default_rng(11)
    a_true, b_true, lapse = 6.0, 15.0, 0.02
    est = []
    for _ in range(5):
        psi = make_psi((-55.0, 50.0), 100, lapse)
        for _ in range(144):
            p1 = pm_function(psi.nextIntensity, a_true, b_true, lapse)
            psi.update(int(rng.uniform() < p1))
        est.append(psi.estimateLambda())
    est = np.array(est)
    assert abs(est[:, 0].mean() - a_true) < 2.5
    assert abs(est[:, 1].mean() - b_true) < 3.0


def test_beta_grid_cap_and_override():
    psi = make_psi((-55.0, 50.0), 100, 0.02)
    assert 22 < psi.beta.max() < 23.5                          # AGRT.py:295 的公式
    assert make_psi((-55.0, 50.0), 100, 0.02, beta_max=40.0).beta.max() == 40.0


def test_salience_levels_match_inv_pm_function_and_warn():
    a, b, d = 6.0, 15.0, 0.01
    levels, warns = salience_levels(a, b, d, [0.99, 0.90], x_range=(-55, 50))
    assert abs(levels[0] - inv_pm_function(0.99, a, b, d)) < 1e-12
    assert abs(levels[0] - (a + b * norm.ppf((0.99 - 0.5 * d) / (1 - d)))) < 1e-12
    assert levels[0] < 50 and not warns
    levels2, warns2 = salience_levels(a, 32.0, d, [0.99, 0.90], x_range=(-55, 50))
    assert levels2[0] > 50 and len(warns2) == 1        # psi Simulation_25JUNE2018.R:174 的情況
