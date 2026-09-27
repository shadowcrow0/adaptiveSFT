"""
DDM 受試者模擬：diffIRT::simdiffT 的逐行移植，加上 adaptiveSFT_functions.R 的 moc_ddm / dfp_ddm。

simdiffT(N, a, mv, sv, ter)：a = 界線距離、起點在 a/2、mv = 平均漂移、sv = 漂移的跨試次 SD、
ter = 非決策時間、擴散係數 vp = 1。反應機率 p = 1/(1 + exp(−a·drift))（simdiffT.r:6）。

⚠ psi Simulation_26MAR2019.R:117 算「真」正確率用 1/(1 + exp(−2·a·s·v))，跟 simdiffT 差一個 2 倍。
  ddm_p_correct() 照 simdiffT；要重現那張圖請自己乘 2。
"""
import numpy as np

__all__ = ["simdiffT", "ddm_p_correct", "ddm_mean_dt", "moc_ddm", "dfp_ddm", "draw_participant"]


def ddm_p_correct(a, drift):
    """simdiffT.r:6：P(上界) = exp(a·drift) / (1 + exp(a·drift))。"""
    return 1.0 / (1.0 + np.exp(-a * np.asarray(drift, dtype=float)))


def ddm_mean_dt(a, drift):
    """sv = 0 時的平均決策時間（對稱起點、s = 1）：(a / 2v) · tanh(a·v / 2)。"""
    drift = np.asarray(drift, dtype=float)
    return np.where(drift == 0, a * a / 4.0, a / (2.0 * drift) * np.tanh(a * drift / 2.0))


def simdiffT(N, a, mv, sv, ter, vp=1.0, max_iter=19999, eps=1e-15, rng=None):
    """diffIRT/R/simdiffT.r 逐行。回傳 (rt, x)，x = 1 表示到上界。"""
    rng = np.random.default_rng() if rng is None else rng
    rt = np.empty(N)
    p = np.empty(N)
    for jj in range(N):
        drift = rng.normal(mv, sv)
        p[jj] = np.exp(a * drift) / (1.0 + np.exp(a * drift))
        lmb = drift ** 2 / (2 * vp ** 2) + np.pi ** 2 * vp ** 2 / (2 * a ** 2)
        FF = np.pi ** 2 * vp ** 4 / (np.pi ** 2 * vp ** 4 + drift ** 2 * a ** 2)
        rej = 0
        while True:
            v = rng.uniform()
            u = rng.uniform()
            sh1, sh2, sh3, i = 1.0, 0.0, 0.0, 0
            while abs(sh1 - sh2) > eps or abs(sh2 - sh3) > eps:
                sh1, sh2 = sh2, sh3
                i += 1
                sh3 = sh2 + (2 * i + 1) * (-1) ** i * (1 - u) ** (FF * (2 * i + 1) ** 2)
            ev = 1 + (1 - u) ** (-FF) * sh3
            if v <= ev:
                rt[jj] = abs(np.log(1 - u)) / lmb + ter
                break
            rej += 1
            if rej == max_iter:
                raise RuntimeError("Rejection algorithm failed. Increase max_iter or change parameters.")
    x = (p > rng.uniform(size=N)).astype(int)
    return rt, x


def moc_ddm(N, a, v, ter, sdv, intensity_levels, rng=None):
    """adaptiveSFT_functions.R:153-165。回傳 (N·levels, 3) [rt, correct, intensity]，models.fit_lnrm 直接吃。"""
    rng = np.random.default_rng() if rng is None else rng
    rows = []
    for i in intensity_levels:
        rt, x = simdiffT(N, a, i * v, sdv, ter, rng=rng)
        rows.append(np.column_stack([rt, x, np.full(N, i, dtype=float)]))
    return np.vstack(rows)


def dfp_ddm(N, drift1, drift2, a, ter, sdv, architecture, stopping_rule=None, pmix=0.5, rng=None):
    """adaptiveSFT_functions.R:115-150。architecture ∈ {COA, PAR, SER}，stopping_rule ∈ {OR, AND}。"""
    rng = np.random.default_rng() if rng is None else rng
    if architecture == "COA":
        rt, x = simdiffT(N, a, drift1 + drift2, sdv, ter, rng=rng)
        return rt, x
    rt1, x1 = simdiffT(N, a, drift1, sdv, ter, rng=rng)
    rt2, x2 = simdiffT(N, a, drift2, sdv, ter, rng=rng)
    if architecture == "PAR":
        if stopping_rule == "OR":
            rt = np.minimum(rt1, rt2)
            cr = np.where(rt1 < rt2, x1, x2)
        elif stopping_rule == "AND":
            rt = np.maximum(rt1, rt2)
            cr = x1 & x2
        else:
            raise ValueError("stopping_rule 必須是 OR / AND")
    elif architecture == "SER":
        if stopping_rule == "OR":
            s = rng.uniform(size=N) < pmix
            rt = np.where(s, rt1, rt2)
            cr = np.where(s, x1, x2)
        elif stopping_rule == "AND":
            rt = rt1 + rt2
            cr = x1 & x2
        else:
            raise ValueError("stopping_rule 必須是 OR / AND")
    else:
        raise ValueError("architecture 必須是 COA / PAR / SER")
    return rt, cr.astype(int)


def draw_participant(a, v, ter, sdv, rng=None):
    """simulateLNRM_ogival.R:496-513：每位模擬受試者的 (a, v, ter, sdv)，全部截到正值。"""
    rng = np.random.default_rng() if rng is None else rng

    def pos(mean, sd):
        val = 0.0
        while val <= 0:
            val = rng.normal(mean, sd)
        return val

    a_p = pos(1.1 * a, a / 6)
    v_p = pos(v / a * a_p, v / 6)
    ter_p = pos(ter, ter / 6)
    return a_p, v_p, ter_p, sdv
