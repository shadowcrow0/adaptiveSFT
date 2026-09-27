"""
DDM 受試者模擬：diffIRT::simdiffT 的逐行移植，加上 adaptiveSFT_functions.R 的 moc_ddm / dfp_ddm。

simdiffT(N, a, mv, sv, ter)：a = 界線距離、起點在 a/2、mv = 平均漂移、sv = 漂移的跨試次 SD、
ter = 非決策時間、擴散係數 vp = 1。反應機率 p = 1/(1 + exp(−a·drift))（simdiffT.r:6）。

⚠ psi Simulation_26MAR2019.R:117 算「真」正確率用 1/(1 + exp(−2·a·s·v))，跟 simdiffT 差一個 2 倍
  （decisions_for_author.md Decision A）。兩種讀法都做成參數 a_is_separation：
    True   a = boundary separation（diffIRT 的定義，simdiffT.Rd:16）        P = 1/(1+exp(−a·drift))
    False  a = 起點到界線的距離（腳本的 `threshold` 命名與 :117 的公式）   P = 1/(1+exp(−2·a·drift))
  預設 True（repo 字面：所有 simdiffT 呼叫都直接傳 threshold）。
"""
import warnings

import numpy as np

__all__ = ["A_CONVENTIONS", "separation", "simdiffT", "ddm_p_correct", "ddm_mean_dt", "moc_ddm", "dfp_ddm",
           "draw_participant"]

A_CONVENTIONS = {"separation": True, "threshold": False}


def separation(a, a_is_separation=True):
    """把腳本裡的 a 換成 diffIRT 要的 boundary separation。"""
    return a if a_is_separation else 2.0 * a


def ddm_p_correct(a, drift, a_is_separation=True):
    """simdiffT.r:6：P(上界) = 1 / (1 + exp(−sep·drift))，sep 依慣例是 a 或 2a。"""
    return 1.0 / (1.0 + np.exp(-separation(a, a_is_separation) * np.asarray(drift, dtype=float)))


def ddm_mean_dt(a, drift, a_is_separation=True):
    """sv = 0 時的平均決策時間（對稱起點、s = 1）：(sep / 2v) · tanh(sep·v / 2)。"""
    sep = separation(a, a_is_separation)
    drift = np.asarray(drift, dtype=float)
    return np.where(drift == 0, sep * sep / 4.0, sep / (2.0 * drift) * np.tanh(sep * drift / 2.0))


def _euler_trial(a, drift, vp, rng, dt=2.5e-4, max_t=60.0):
    """Wiener 過程直接模擬（起點 a/2、界線 0 與 a、擴散 √vp）。只在拒絕抽樣失敗時當備援。
    回傳 (決策時間, 是否到上界)。"""
    x = a / 2.0
    sd = np.sqrt(vp * dt)
    n_max = int(max_t / dt)
    steps = rng.normal(drift * dt, sd, size=4096)
    k = 0
    for i in range(n_max):
        if k == steps.size:
            steps = rng.normal(drift * dt, sd, size=4096)
            k = 0
        x += steps[k]
        k += 1
        if x >= a:
            return (i + 1) * dt, 1
        if x <= 0.0:
            return (i + 1) * dt, 0
    return max_t, int(x > a / 2.0)


def simdiffT(N, a, mv, sv, ter, vp=1.0, max_iter=19999, eps=1e-15, rng=None, a_is_separation=True,
             on_fail="euler", euler_above=10.0):
    """
    diffIRT/R/simdiffT.r 逐行。回傳 (rt, x)，x = 1 表示到上界。a_is_separation=False 時傳 2a 進去。

    a·|drift| 很大時 diffIRT 的拒絕抽樣不可用：級數 (1−u)^(FF·(2i+1)²) 在 FF → 0 時收斂極慢，
    接受率也趨近 0（R 會跑很久然後 stop()）。所以：
      a·|drift| > euler_above          直接用 Wiener 過程模擬那一試（P(correct) 已 > 0.99995）
      拒絕抽樣達 max_iter 仍失敗       on_fail="euler" 同上並警告一次；"raise" 照 R 丟 RuntimeError
    """
    rng = np.random.default_rng() if rng is None else rng
    a = separation(a, a_is_separation)
    rt = np.empty(N)
    p = np.empty(N)
    x = np.full(N, -1)
    warned = False
    for jj in range(N):
        drift = rng.normal(mv, sv)
        p[jj] = np.exp(a * drift) / (1.0 + np.exp(a * drift))
        if abs(a * drift) > euler_above:
            dt_, hit = _euler_trial(a, drift, vp, rng)
            rt[jj] = dt_ + ter
            x[jj] = hit
            continue
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
                if on_fail != "euler":
                    raise RuntimeError("Rejection algorithm failed. Increase max_iter or change parameters.")
                if not warned:
                    warned = True
                    warnings.warn(f"simdiffT: 拒絕抽樣在 a·drift ≈ {a * drift:.1f} 失敗（R 會 stop），改用 Euler 模擬",
                                  RuntimeWarning, stacklevel=2)
                dt_, hit = _euler_trial(a, drift, vp, rng)
                rt[jj] = dt_ + ter
                x[jj] = hit
                break
    u = rng.uniform(size=N)
    x = np.where(x < 0, (p > u).astype(int), x)
    return rt, x


def moc_ddm(N, a, v, ter, sdv, intensity_levels, rng=None, a_is_separation=True):
    """adaptiveSFT_functions.R:153-165。回傳 (N·levels, 3) [rt, correct, intensity]，models.fit_lnrm 直接吃。"""
    rng = np.random.default_rng() if rng is None else rng
    rows = []
    for i in intensity_levels:
        rt, x = simdiffT(N, a, i * v, sdv, ter, rng=rng, a_is_separation=a_is_separation)
        rows.append(np.column_stack([rt, x, np.full(N, i, dtype=float)]))
    return np.vstack(rows)


def dfp_ddm(N, drift1, drift2, a, ter, sdv, architecture, stopping_rule=None, pmix=0.5, rng=None,
            a_is_separation=True):
    """adaptiveSFT_functions.R:115-150。architecture ∈ {COA, PAR, SER}，stopping_rule ∈ {OR, AND}。"""
    rng = np.random.default_rng() if rng is None else rng
    kw = dict(rng=rng, a_is_separation=a_is_separation)
    if architecture == "COA":
        rt, x = simdiffT(N, a, drift1 + drift2, sdv, ter, **kw)
        return rt, x
    rt1, x1 = simdiffT(N, a, drift1, sdv, ter, **kw)
    rt2, x2 = simdiffT(N, a, drift2, sdv, ter, **kw)
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
