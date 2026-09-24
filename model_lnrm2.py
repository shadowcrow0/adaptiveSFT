"""
lnrm2.stan 的 PyMC 版本，用 PyMC package 的標準寫法（同 symmetry repo 的 model_PS_v2.py 那一類）：

    numba 純量 likelihood  →  自訂 pt.Op（逐題 log-likelihood 向量）
                           →  pm.Deterministic("log_likelihood")  供 az.waic / az.loo
                           →  pm.Potential("obs")                 對應 Stan 的 target +=

    Op 沒有梯度，所以採樣用 pm.DEMetropolisZ()。
    fix_params 可把任一參數固定為常數。

資料格式（同 symmetry repo 的慣例，(N, 3) numpy array）：
    observed_data[:, 0] = rt          反應時間（秒）
    observed_data[:, 1] = correct     1 = 答對, 0 = 答錯
    observed_data[:, 2] = intensity   刺激強度（連續值）

對應 lnrm2.stan 的行號寫在各段註解裡。轉換時撞到的問題見 lnrm2_pymc_gaps.md。
與 lnrm2_pymc.py（NUTS 版）的差別：這裡 likelihood 完全在 numba 裡算，
不經過 PyMC 的 logcdf / log1mexp，所以 gaps.md 第 1 節那個 -inf 問題根本不會發生。
"""
import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply
import pymc as pm
import arviz as az
from numba import jit
import math

LOG_SQRT_2PI = 0.9189385332046727      # ln √(2π)
SQRT2 = 1.4142135623730951


# ============================================================================
# 1. NUMBA 純量函式：對數常態的 log-pdf 與 log-存活函數（lnrm2_math.md §1）
# ============================================================================

@jit(nopython=True, fastmath=False)
def log_norm_sf(u):
    """
    ln P(Z > u)，Z 為標準常態。這就是 Stan lognormal_lccdf 底層需要的東西。
    u <= 30：用 erfc，尾端相對精度好，不會像 1 - cdf 那樣在 cdf≈1 時消去成 0。
    u  > 30：erfc 會 underflow 到 0，改用漸近展開  ln sf ≈ -u²/2 - ln u - ln√(2π) + ln(1 - 1/u² + 3/u⁴)
    """
    if u > 30.0:
        u2 = u * u
        return -0.5 * u2 - math.log(u) - LOG_SQRT_2PI + math.log(1.0 - 1.0 / u2 + 3.0 / (u2 * u2))
    return math.log(0.5 * math.erfc(u / SQRT2))


@jit(nopython=True, fastmath=False)
def lognormal_logpdf(y, m, s):
    """ln f(y; m, s)，對應 Stan lognormal_lpdf(y | m, s)。y <= 0 給大懲罰（同 symmetry 的 -50 慣例）。"""
    if y <= 0.0:
        return -50.0
    z = (math.log(y) - m) / s
    return -math.log(y) - math.log(s) - LOG_SQRT_2PI - 0.5 * z * z


@jit(nopython=True, fastmath=False)
def lognormal_logsf(y, m, s):
    """ln S(y; m, s) = ln P(Y > y)，對應 Stan lognormal_lccdf(y | m, s)。"""
    if y <= 0.0:
        return 0.0                       # 還沒開始跑，一定還沒到終點：S = 1
    return log_norm_sf((math.log(y) - m) / s)


@jit(nopython=True, fastmath=False)
def lnrm_def_logpdf(t, m_win, m_lose, s):
    """
    賽跑的「defective」對數密度（對應 symmetry 的 lba_def_pdf，但直接在 log 空間算）：
        ln f(t; 贏家) + ln S(t; 輸家)
    就是 lnrm2.stan:38-39 那兩行 target += 的和。
    """
    return lognormal_logpdf(t, m_win, s) + lognormal_logsf(t, m_lose, s)


@jit(nopython=True, fastmath=False)
def lnrm_pointwise_loglik(rt, correct, intensity, mu, alpha, alpha2, varZ, psi):
    """逐題 log-likelihood。整個 for 迴圈在 numba 裡跑。"""
    n = rt.shape[0]
    out = np.empty(n)
    for i in range(n):
        # transformed parameters（lnrm2.stan:23-24）
        d = alpha * intensity[i] + alpha2 * intensity[i] * intensity[i]
        z1 = mu - d                     # 答對累積器
        z2 = mu + d                     # 答錯累積器
        t = rt[i] - psi                 # rt[tr] - psi（lnrm2.stan:38）
        if correct[i] == 1:             # if (correct[tr])（lnrm2.stan:37）
            out[i] = lnrm_def_logpdf(t, z1, z2, varZ)
        else:                           # else（lnrm2.stan:41）
            out[i] = lnrm_def_logpdf(t, z2, z1, varZ)
    return out


# ============================================================================
# 2. 資料生成：直接照 lnrm2.stan 的世界觀抽（同 symmetry 的 lba_2dim_random 角色）
# ============================================================================

def lnrm_random(n_trials, mu, alpha, alpha2, varZ, psi, intensity_range=(0.0, 3.0), rng=None):
    """
    兩個對數常態累積器賽跑，先到者決定 correct 與 rt。
    注意：不像 symmetry 那邊把 rt > 5s 丟掉重抽 —— lnrm2.stan 沒有截尾，
    這裡也不截，否則回收會有偏。
    回傳 (n_trials, 3)：[rt, correct, intensity]
    """
    if rng is None:
        rng = np.random.default_rng()
    intensity = rng.uniform(intensity_range[0], intensity_range[1], n_trials)
    d = alpha * intensity + alpha2 * intensity ** 2
    t1 = psi + np.exp(rng.normal(mu - d, varZ))     # 答對累積器完成時間
    t2 = psi + np.exp(rng.normal(mu + d, varZ))     # 答錯累積器完成時間
    rt = np.minimum(t1, t2)
    correct = (t1 < t2).astype(float)
    return np.column_stack([rt, correct, intensity])


# ============================================================================
# 3. PyMC 模型與逐點 likelihood Op
# ============================================================================

class LNRM2_PointwiseOp(pt.Op):
    """
    pt 積木：把 numba 的逐題 log-likelihood 包成 PyTensor 節點。
    用 make_node 而不是死的 itypes，這樣不管參數是純量、(1,) 向量（pm.CustomDist 的 logp
    會這樣傳）、還是 python float，都收得進來。輸出永遠是 float64 向量。
    """
    __props__ = ()

    def make_node(self, rt, correct, intensity, mu, alpha, alpha2, varZ, psi):
        rt = pt.as_tensor_variable(rt).astype('float64')
        correct = pt.as_tensor_variable(correct).astype('int32')
        intensity = pt.as_tensor_variable(intensity).astype('float64')
        scalars = [pt.as_tensor_variable(v).astype('float64') for v in (mu, alpha, alpha2, varZ, psi)]
        return Apply(self, [rt, correct, intensity, *scalars],
                                    [pt.TensorType('float64', shape=(None,))()])

    def perform(self, node, inputs, outputs):
        rt, correct, intensity, mu, alpha, alpha2, varZ, psi = inputs
        f = lambda v: float(np.asarray(v).reshape(-1)[0])       # 純量或 (1,) 都吃
        outputs[0][0] = lnrm_pointwise_loglik(
            np.ascontiguousarray(rt), np.ascontiguousarray(correct), np.ascontiguousarray(intensity),
            f(mu), f(alpha), f(alpha2), f(varZ), f(psi))


def build_model_lnrm2(observed_data, tune=4000, draws=4000, chains=8, fix_params=None,
                      random_seed=42, as_observed=False):
    """
    lnrm2.stan 的 PyMC 模型。

    fix_params: dict，可固定 'mu' / 'alpha' / 'alpha2' / 'varZ' / 'psi' 任一個為常數
                例：fix_params={'psi': 0.12, 'varZ': 0.6}
    as_observed: 兩種 pt / pm 拼法，likelihood 數值完全相同（都對過 scipy）：
        False → pt.Op ──► pm.Potential("obs")                   ＋ Deterministic("log_likelihood")
        True  → pt.Op ──► pm.CustomDist("rt_obs", logp=Op, observed=rt)
                 rt 是常數、psi 在分布參數位置，所以 gaps.md §2 的 observed 限制碰不到；
                 rt_obs 是正規的觀測變數（model.observed_RVs），之後可接 PPC。
    """
    if fix_params is None:
        fix_params = {}
    rt = observed_data[:, 0].astype('float64')
    correct = observed_data[:, 1].astype('int32')
    intensity = observed_data[:, 2].astype('float64')
    min_rt = float(rt.min())                          # lnrm2.stan:5  minRT

    with pm.Model() as model:
        # --- parameters + priors（lnrm2.stan:13-17, 30-33）--------------------
        # 每一個都可以被 fix_params 固定；沒固定就照 Stan 的先驗估
        if 'mu' in fix_params:
            mu = fix_params['mu'];        print(f"  [Fixed] mu={mu:.3f}")
        else:
            mu = pm.Normal("mu", 0.0, 1.0)
        if 'alpha' in fix_params:
            alpha = fix_params['alpha'];  print(f"  [Fixed] alpha={alpha:.3f}")
        else:
            alpha = pm.Normal("alpha", 0.0, 2.0)
        if 'alpha2' in fix_params:
            alpha2 = fix_params['alpha2']; print(f"  [Fixed] alpha2={alpha2:.3f}")
        else:
            alpha2 = pm.Normal("alpha2", 0.0, 1.0)
        if 'varZ' in fix_params:
            varZ = fix_params['varZ'];    print(f"  [Fixed] varZ={varZ:.3f}")
        else:
            # 名字叫 varZ，但在 Stan 是放在「標準差」位置；這裡照樣當標準差用
            varZ = pm.InverseGamma("varZ", alpha=1.0, beta=0.1)
        if 'psi' in fix_params:
            psi = fix_params['psi'];      print(f"  [Fixed] psi={psi:.3f}")
        else:
            # real<lower=0,upper=minRT> psi;  沒有 ~ 敘述 → 有限區間上的 flat = Uniform
            psi = pm.Uniform("psi", lower=0.0, upper=min_rt)

        # --- Likelihood：兩種積木拼法 ---------------------------------------------
        op = LNRM2_PointwiseOp()
        if as_observed:
            # 拼法 B：Op 當 CustomDist 的 logp。value 就是 rt（常數），psi 是分布參數。
            def race_logp(value, mu_, alpha_, alpha2_, varZ_, psi_):
                return op(value, correct, intensity, mu_, alpha_, alpha2_, varZ_, psi_)
            rt_obs = pm.CustomDist("rt_obs", mu, alpha, alpha2, varZ, psi,
                                   logp=race_logp, observed=rt)
            pm.Deterministic("log_likelihood",
                             op(rt, correct, intensity, mu, alpha, alpha2, varZ, psi))
        else:
            # 拼法 A：Op 的逐點向量直接 Potential 進後驗（對應 Stan 的 target +=）
            log_lik_vec = op(rt, correct, intensity, mu, alpha, alpha2, varZ, psi)
            pm.Deterministic("log_likelihood", log_lik_vec)   # 逐點，給 az.waic / az.loo
            pm.Potential("obs", pt.sum(log_lik_vec))

        # --- initvals：只給有估的參數 ----------------------------------------------
        init_vals = {}
        if 'mu' not in fix_params:     init_vals["mu"] = 1.0
        if 'alpha' not in fix_params:  init_vals["alpha"] = 0.5
        if 'alpha2' not in fix_params: init_vals["alpha2"] = -0.1
        if 'varZ' not in fix_params:   init_vals["varZ"] = 0.5
        if 'psi' not in fix_params:    init_vals["psi"] = 0.3 * min_rt

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          step=pm.DEMetropolisZ(), random_seed=random_seed,
                          initvals=init_vals, progressbar=False)
    return trace


# ============================================================================
# 4. 自我測試：生一批資料，看能不能把真值找回來
# ============================================================================
if __name__ == "__main__":
    true_params = {'mu': 1.5, 'alpha': 0.8, 'alpha2': -0.15, 'varZ': 0.6, 'psi': 0.12}
    data = lnrm_random(1000, **true_params, rng=np.random.default_rng(42))
    print(f"N={len(data)}  minRT={data[:, 0].min():.3f}  accuracy={data[:, 1].mean():.3f}")

    trace = build_model_lnrm2(data, tune=3000, draws=3000, chains=8)

    summary = az.summary(trace, var_names=list(true_params))
    summary.insert(0, "true", [true_params[p] for p in summary.index])
    print(summary[["true", "mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
    print("\n注意：mu 與 psi 在後驗上高度相關，N 不大時會互相補償；這是模型本身的性質。")
