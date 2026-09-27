"""
賽跑（lognormal race）的 log-likelihood 積木與密度 / CDF。

numba 部分逐字取自本 repo 的 model_lnrm2.py:37-73（已對過 scipy 到 1e-14），
只做一個改動：`lnrm_pointwise_loglik` 不再自己算 d = α·x + α₂·x²，而是收「每題已算好的 d」。
這樣 lnrm0 / lnrm1 / lnrm2 / lnrm2a 四個模型共用同一個核心，差別全部集中在 models.py 的 d_expr。

    z[1,tr] = mu − d[tr]        答對累積器（lnrm2.stan:23）
    z[2,tr] = mu + d[tr]        答錯累積器（lnrm2.stan:24）
    ln f(rt − psi | 贏家) + ln S(rt − psi | 輸家)      （lnrm2.stan:38-39 / 42-43）

尺度約定（issue.md S3 在這裡定案）：
    varZ 是**對數尺度的標準差**，照 lnrm2.stan:38 的用法。
    R 的 dlognormalrace(…, sigmasq) 收變異數再開根號（adaptiveSFT_functions.R:61-62），
    而 simulateLNRM_ogival.R:204/213/258/281 卻把 varZ 直接塞進 sigmasq —— R 端自己就不一致。
    本檔的 dlognormalrace / plognormalrace 一律收 varZ（SD），不開根號。
"""
import math

import numpy as np
from numba import jit
from scipy.integrate import quad
from scipy.stats import lognorm

__all__ = [
    "log_norm_sf", "lognormal_logpdf", "lognormal_logsf", "lnrm_def_logpdf",
    "lnrm_pointwise_loglik", "lnrm_random", "dlognormalrace", "plognormalrace", "plognormalrace_curve",
]

LOG_SQRT_2PI = 0.9189385332046727      # ln √(2π)
SQRT2 = 1.4142135623730951


# ============================================================================
# 1. numba 純量函式（model_lnrm2.py:37-73 逐字）
# ============================================================================

@jit(nopython=True, fastmath=False)
def log_norm_sf(u):
    """ln P(Z > u)。u ≤ 30 用 erfc；u > 30 用漸近級數，避免 underflow 成 −inf。"""
    if u > 30.0:
        u2 = u * u
        return -0.5 * u2 - math.log(u) - LOG_SQRT_2PI + math.log(1.0 - 1.0 / u2 + 3.0 / (u2 * u2))
    return math.log(0.5 * math.erfc(u / SQRT2))


@jit(nopython=True, fastmath=False)
def lognormal_logpdf(y, m, s):
    """ln f(y; m, s)，對應 Stan lognormal_lpdf(y | m, s)。y ≤ 0 給大懲罰。"""
    if y <= 0.0:
        return -50.0
    z = (math.log(y) - m) / s
    return -math.log(y) - math.log(s) - LOG_SQRT_2PI - 0.5 * z * z


@jit(nopython=True, fastmath=False)
def lognormal_logsf(y, m, s):
    """ln S(y; m, s) = ln P(Y > y)，對應 Stan lognormal_lccdf(y | m, s)。"""
    if y <= 0.0:
        return 0.0
    return log_norm_sf((math.log(y) - m) / s)


@jit(nopython=True, fastmath=False)
def lnrm_def_logpdf(t, m_win, m_lose, s):
    """賽跑的 defective 對數密度：ln f(t; 贏家) + ln S(t; 輸家)。"""
    return lognormal_logpdf(t, m_win, s) + lognormal_logsf(t, m_lose, s)


@jit(nopython=True, fastmath=False)
def lnrm_pointwise_loglik(rt, correct, d, mu, varZ, psi):
    """逐題 log-likelihood。d 是每題的難度向量（由呼叫端依模型算好）。"""
    n = rt.shape[0]
    out = np.empty(n)
    for i in range(n):
        z1 = mu - d[i]                  # 答對累積器
        z2 = mu + d[i]                  # 答錯累積器
        t = rt[i] - psi
        if correct[i] == 1:
            out[i] = lnrm_def_logpdf(t, z1, z2, varZ)
        else:
            out[i] = lnrm_def_logpdf(t, z2, z1, varZ)
    return out


# ============================================================================
# 2. 資料生成：照 lnrm2.stan 的世界觀抽，d 由呼叫端給
# ============================================================================

def lnrm_random(d, mu, varZ, psi, rng=None):
    """
    兩個對數常態累積器賽跑，先到者決定 correct 與 rt。不截尾（lnrm2.stan 沒有截尾）。
    d：每題的難度（長度 N）。回傳 (rt, correct) 兩個長度 N 的陣列。
    """
    rng = np.random.default_rng() if rng is None else rng
    d = np.asarray(d, dtype=float)
    t1 = psi + np.exp(rng.normal(mu - d, varZ))
    t2 = psi + np.exp(rng.normal(mu + d, varZ))
    return np.minimum(t1, t2), (t1 < t2).astype(float)


# ============================================================================
# 3. 密度與 CDF（adaptiveSFT_functions.R:61-111 的 scipy 版）
# ============================================================================

def dlognormalrace(x, m, psi, mu, varZ, log=False):
    """
    累積器 m（0-based）在時間 x 先到的 defective 密度（R :61-77）。
    mu、varZ 皆為長度 = 累積器數的陣列；varZ 是 SD（見檔頭）。x < psi 的密度為 0。
    """
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(varZ, dtype=float)
    x = np.asarray(x, dtype=float)
    t = x - psi
    with np.errstate(divide="ignore", invalid="ignore"):
        g = lognorm.logpdf(t, s=sd[m], scale=np.exp(mu[m]))
        G = np.zeros_like(t)
        for i in range(len(mu)):
            if i != m:
                G = G + lognorm.logsf(t, s=sd[i], scale=np.exp(mu[i]))
        out = np.where(t > 0, g + G, -np.inf)
    return out if log else np.exp(out)


def plognormalrace(x, m, psi, mu, varZ):
    """
    累積器 m 先到且時間 ≤ x 的機率（R :81-111）。
    R 版對 integrate 失敗有三層 tryCatch 補救（且 :98 有 sigmasqx 打錯字）；
    這裡從 psi 起積，quad 不會失敗，不需要補救。
    """
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(xs)
    for k, xi in enumerate(xs):
        if xi > psi:
            out[k] = quad(dlognormalrace, psi, xi, args=(m, psi, mu, varZ), limit=200)[0]
    return out


def plognormalrace_curve(t, m, psi, mu, varZ, n_grid=4000):
    """
    plognormalrace 的向量版：在密網格上算密度再累積梯形積分，一次給整條 CDF 曲線。
    畫後驗預測圖（simulateLNRM_ogival.R:142-300）要對幾百個 t、幾十個 draw 各算一次，
    逐點 quad 會慢到不能用；這裡一條曲線 < 1 ms，對 quad 的誤差約 1e-5（tests/test_race.py）。
    """
    t = np.asarray(t, dtype=float)
    hi = float(t.max()) if t.size else psi
    if hi <= psi:
        return np.zeros_like(t)
    grid = np.linspace(psi, hi, n_grid)
    dens = dlognormalrace(grid, m, psi, mu, varZ)
    dens[0] = 0.0
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    return np.interp(t, grid, cdf, left=0.0)
