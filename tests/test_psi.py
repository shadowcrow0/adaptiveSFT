"""psi.py：網格與 R 的 seq() 一致；累積常態受試者的 α/β 回復；salience_levels = inv.pm.function 並警告範圍外。"""
import numpy as np
from scipy.stats import norm

from adaptivesft.psi import GRIDS, Psi, inv_pm_function, make_psi, pm_function, salience_levels


def test_grids_match_r_seq():
    c = make_psi("colour")                       # psiSimulation_functions.R:15-17
    assert len(c.x) == 106 and c.x[0] == -55 and c.x[-1] == 50
    assert len(c.a) == 71 and c.a[0] == -25 and c.a[-1] == 45
    assert len(c.b) == 50 and c.b[0] == 1 and c.b[-1] == 50
    o = make_psi("orientation")                  # :222-224
    assert len(o.x) == 91 and o.x[-1] == 90
    assert len(o.a) == 60 and o.a[0] == 45.5 and o.a[-1] == 75
    assert len(o.b) == 19 and o.b[0] == 1 and o.b[-1] == 10
    assert c.d == GRIDS["colour"]["d"] == 0.01
    assert c.pR_LX.shape == (2, 71, 50, 106) and abs(c.pL.sum() - 1) < 1e-12


def test_first_trial_and_update_shapes():
    psi = make_psi("colour")
    assert psi.next_intensity in psi.x
    psi.update(1)
    assert abs(psi.pL.sum() - 1) < 1e-12 and psi.next_intensity in psi.x


def test_recovery_gaussian_observer_colour():
    rng = np.random.default_rng(11)
    a_true, b_true, d = 6.0, 15.0, 0.01                 # R :10-12 的 sim.a / sim.b / sim.d
    est = []
    for _ in range(5):
        psi = make_psi("colour")
        for _ in range(144):
            psi.update(int(rng.uniform() < pm_function(psi.next_intensity, a_true, b_true, d)))
        est.append(psi.estimate())
    est = np.array(est)
    assert abs(est[:, 0].mean() - a_true) < 2.5
    assert abs(est[:, 1].mean() - b_true) < 3.0


def test_beta_outside_grid_is_pinned_at_grid_edge():
    # 方位網格 b ∈ [1, 10]（R :224）：真 β = 19 的受試者只能估到 10 附近
    rng = np.random.default_rng(3)
    psi = make_psi("orientation")
    for _ in range(300):
        psi.update(int(rng.uniform() < pm_function(psi.next_intensity, 63.0, 19.0, 0.01)))
    a_hat, b_hat = psi.estimate()
    assert b_hat > 9.0 and b_hat <= 10.0 and abs(a_hat - 63) < 3


def test_custom_grid():
    psi = make_psi(x=(0, 10, 2), a=(0, 10, 5), b=(1, 3, 1), d=0.02)
    assert len(psi.x) == 6 and len(psi.a) == 3 and len(psi.b) == 3 and psi.d == 0.02
    import pytest
    with pytest.raises(ValueError):
        make_psi(x=(0, 1, 1))


def test_salience_levels_match_inv_pm_function_and_warn():
    a, b, d = 6.0, 15.0, 0.01
    levels, warns = salience_levels(a, b, d, [0.99, 0.90], x_range=(-55, 50))
    assert abs(levels[0] - inv_pm_function(0.99, a, b, d)) < 1e-12
    assert abs(levels[0] - (a + b * norm.ppf((0.99 - 0.5 * d) / (1 - d)))) < 1e-12
    assert levels[0] < 50 and not warns
    levels2, warns2 = salience_levels(a, 32.0, d, [0.99, 0.90], x_range=(-55, 50))
    assert levels2[0] > 50 and len(warns2) == 1        # psi Simulation_25JUNE2018.R:174 的情況


def test_asymptotes_and_update_at():
    from adaptivesft.psi import Psi
    psi = make_psi("colour")                                  # 預設 = R：兩端 d/2
    assert psi.lower == psi.upper == 0.005
    p2 = Psi((0, 40, 1), (0, 40, 1), (1, 20, 1), d=0.02, lower=0.15, upper=0.02)
    assert abs(p2.pR_LX[1, 0, 0, 0] - (0.15 + 0.83 * norm.cdf(0, 0, 1))) < 1e-12
    # update_at 在 next_index 上 = update；在別的點上也是合法的貝氏更新
    a = make_psi("colour"); b = make_psi("colour")
    a.update(1); b.update_at(b.next_index, 1)
    assert np.allclose(a.pL, b.pL) and a.next_intensity == b.next_intensity
    idx = b.nearest_index(12.3)
    assert b.x[idx] == 12.0
    b.update_at(idx, 0)
    assert abs(b.pL.sum() - 1) < 1e-12
    lv, _ = salience_levels(5.0, 10.0, 0.02, [0.9], lower=0.15, upper=0.02)
    assert abs(lv[0] - (5.0 + 10.0 * norm.ppf((0.9 - 0.15) / 0.83))) < 1e-12
