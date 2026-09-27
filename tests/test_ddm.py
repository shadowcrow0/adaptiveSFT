"""ddm.py：反應機率對 simdiffT.r:6；sv = 0 時平均 RT 對解析式；dfp_ddm / moc_ddm 形狀。"""
import numpy as np

from adaptivesft.ddm import ddm_mean_dt, ddm_p_correct, dfp_ddm, draw_participant, moc_ddm, simdiffT


def test_response_probability_matches_simdiffT_formula():
    rng = np.random.default_rng(0)
    a, drift = 1.45, 1.2
    _, x = simdiffT(4000, a, drift, 0.0, 0.1, rng=rng)
    assert abs(x.mean() - ddm_p_correct(a, drift)) < 0.02
    # 差 2 倍的那條（psi Simulation_26MAR2019.R:117）不是資料的真相
    assert abs(x.mean() - 1 / (1 + np.exp(-2 * a * drift))) > 0.05


def test_mean_rt_matches_analytic_when_sv_zero():
    rng = np.random.default_rng(1)
    a, drift, ter = 1.45, 1.2, 0.1
    rt, _ = simdiffT(4000, a, drift, 0.0, ter, rng=rng)
    want = ter + ddm_mean_dt(a, drift)
    assert abs(rt.mean() - want) / want < 0.03


def test_dfp_and_moc_shapes():
    rng = np.random.default_rng(2)
    for arch, rule in (("COA", None), ("PAR", "OR"), ("PAR", "AND"), ("SER", "OR"), ("SER", "AND")):
        rt, cr = dfp_ddm(50, 1.5, 0.8, 1.45, 0.1, 0.25, arch, rule, rng=rng)
        assert rt.shape == (50,) and set(np.unique(cr)) <= {0, 1}
    data = moc_ddm(20, 1.45, 1.6, 0.1, 0.25, [0.2, 0.6, 1.0], rng=rng)
    assert data.shape == (60, 3) and set(np.unique(data[:, 2])) == {0.2, 0.6, 1.0}


def test_serial_and_is_sum_and_parallel_or_is_min():
    rng = np.random.default_rng(3)
    rt_and, _ = dfp_ddm(500, 1.5, 1.5, 1.45, 0.1, 0.0, "SER", "AND", rng=rng)
    rt_or, _ = dfp_ddm(500, 1.5, 1.5, 1.45, 0.1, 0.0, "PAR", "OR", rng=rng)
    assert rt_and.mean() > 1.8 * rt_or.mean()


def test_draw_participant_positive():
    rng = np.random.default_rng(4)
    for _ in range(50):
        assert all(v > 0 for v in draw_participant(3.0, 2.0, 0.1, 0.2, rng=rng))


def test_threshold_convention_doubles_the_exponent():
    rng = np.random.default_rng(7)
    a, drift = 1.45, 1.6                                    # psiSimulation_functions.R:96-97，x = 範圍上限
    assert abs(ddm_p_correct(a, drift, a_is_separation=False) - 1 / (1 + np.exp(-2 * a * drift))) < 1e-15
    _, x = simdiffT(3000, a, drift, 0.0, 0.1, rng=rng, a_is_separation=False)
    assert abs(x.mean() - 0.9904) < 0.015                   # decisions_for_author.md A.4：line 117 的世界裡範圍上限 = 99%


def test_euler_fallback_matches_rejection_sampler_and_handles_large_drift():
    import warnings
    from adaptivesft.ddm import _euler_trial
    rng = np.random.default_rng(9)
    a, drift = 1.45, 1.2
    e = np.array([_euler_trial(a, drift, 1.0, rng) for _ in range(1500)])
    assert abs(e[:, 1].mean() - ddm_p_correct(a, drift)) < 0.03
    assert abs(e[:, 0].mean() - ddm_mean_dt(a, drift)) / ddm_mean_dt(a, drift) < 0.06
    rt, x = simdiffT(200, 3.0, 6.4, 0.0, 0.1, rng=rng)             # a·drift ≈ 19 > euler_above：直接 Euler
    assert x.mean() == 1.0 and rt.min() > 0.1
    assert abs(rt.mean() - 0.1 - ddm_mean_dt(3.0, 6.4)) / ddm_mean_dt(3.0, 6.4) < 0.1
    import time
    t0 = time.time()
    simdiffT(300, 6.0, 4.4, 0.0, 0.1, rng=rng)                     # 之前卡住的情況（threshold 慣例的 DFP）
    assert time.time() - t0 < 5.0
