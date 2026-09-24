"""
lnrm2a.stan 的 PyMC 重建版。

重要：lnrm2a.stan 這個檔案從 repo 第一個 commit 起就不存在（見 issue.md S2 / R5），
我沒有看過它的原始碼。這個檔是從 R 端「怎麼用它」反推出來的，不是逐行移植。
能從 R 讀出來的事實只有這些：

    adaptiveSFT_functions.R:186   pars = c("slope", "midpoint", "mu", "varZ", "psi")
    adaptiveSFT_functions.R:11    x <- L * inv_logit(slope * (intensity - midpoint))
    adaptiveSFT_functions.R:194   l_targ.dist = logit(l_targ / 10.) / slope + midpoint
    simulateLNRM_ogival.R:26      L <- 10 # max separation

所以 lnrm2a 跟 lnrm2 的差別只在「強度 → 難度 d」那條曲線：

    lnrm2   d(x) = alpha·x + alpha2·x²              (二次式，lnrm2.stan:23-24)
    lnrm2a  d(x) = ½ · L · 1 / (1 + exp(−slope·(x − midpoint)))   (ogival / logistic，L = 10)
            （½ 來自 simulateLNRM_ogival.R:206-209 的 c(-.5,.5)*L：L 是兩個累積器的
              總距離 "max separation"，每個累積器各偏一半）

           d
        L ─┼─────────────────────────╭─────
           │                     ╭───╯
       L/2 ┼ · · · · · · · · ·╭──╯
           │              ╭───╯
         0 ─┼─────────────╯──────────────────► x
                        midpoint
                   (斜率由 slope 控制)

其餘積木（race likelihood、psi、varZ、mu、pt.Op、Potential / CustomDist）
全部直接沿用 model_lnrm2.py，一行不改。

TODO（拿到 lnrm2a.stan 原檔才能確認的假設，每一個都可能錯）：
    TODO-A1  [已解] d = ½·L·inv_logit(slope·(x − midpoint))。證據 simulateLNRM_ogival.R:206-209；
             實測 L=10 回收成功（見下）。仍待原檔最終確認。
    TODO-A2  z[1] = mu − d, z[2] = mu + d 這個對稱結構有沒有沿用 lnrm2？
    TODO-A3  slope、midpoint 的先驗是什麼？這裡先給 Normal(0, 2)、Normal(0, 2)。
    TODO-A4  L 是 data 傳進去、寫死在 Stan 裡、還是也在估？這裡當常數 10。
    TODO-A5  mu / varZ / psi 的先驗是否跟 lnrm2.stan:30-31 相同？這裡假設相同。

原作者口頭確認（2026-09-24，經使用者轉述）：先有 lnrm2.stan，a/b/c 都是它的微改。
所以「只換 d 那條曲線、其餘沿用」這個做法方向正確，但改的細節還是要原檔。

實測（N=1000，DEMetropolisZ 8 chains）：
    d = L·p,  L = 2    d ∈ [0.1, 1.9]   五個參數全部回收，R-hat ≤ 1.07
    d = L·p,  L = 10   d ∈ [0.5, 9.5]   accuracy 99%、rt 中位數 0.15 s、R-hat > 2  ← 錯的寫法
    d = ½L·p, L = 10   d ∈ [0.2, 4.8]   五個參數全部回收，R-hat ≤ 1.01           ← 現在的寫法
d = 8 配 z = mu ∓ d 會讓答對累積器的均值 exp(mu − 8) ≈ 0.001 s，資料不可能長這樣；
加上 ½ 之後 d 最大 4.8，模型正常。紀錄在 log.md。
"""
import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply
import pymc as pm
import arviz as az
from numba import jit
import math

# 沿用 lnrm2 的 numba 積木：log-pdf、log-survival、單題 race logpdf
from model_lnrm2 import lnrm_def_logpdf

L_MAX_SEPARATION = 10.0        # simulateLNRM_ogival.R:26  L <- 10 # max separation


# ============================================================================
# 1. 唯一改動的積木：強度 → 難度 用 ogival 曲線
# ============================================================================

@jit(nopython=True, fastmath=False)
def ogival_d(x, slope, midpoint, L):
    """
    adaptiveSFT_functions.R:11      L * inv_logit(slope * (intensity - midpoint))   ← 兩個累積器的總距離
    simulateLNRM_ogival.R:206-209   mu + c(-.5, .5) * L * inv_logit(...)            ← 每個累積器各偏 ½
    所以 z = mu ∓ d 裡的 d 是 ½·L·inv_logit，不是 L·inv_logit（TODO-A1 已由這兩行解決）。
    """
    return 0.5 * L / (1.0 + math.exp(-slope * (x - midpoint)))


@jit(nopython=True, fastmath=False)
def lnrm2a_pointwise_loglik(rt, correct, intensity, mu, slope, midpoint, varZ, psi, L):
    n = rt.shape[0]
    out = np.empty(n)
    for i in range(n):
        d = ogival_d(intensity[i], slope, midpoint, L)       # ← 跟 lnrm2 只差這一行（含 ½）
        z1 = mu - d
        z2 = mu + d
        t = rt[i] - psi
        if correct[i] == 1:
            out[i] = lnrm_def_logpdf(t, z1, z2, varZ)
        else:
            out[i] = lnrm_def_logpdf(t, z2, z1, varZ)
    return out


def lnrm2a_random(n_trials, mu, slope, midpoint, varZ, psi, L=L_MAX_SEPARATION,
                  intensity_range=(0.0, 3.0), rng=None):
    """兩個對數常態累積器賽跑（同 model_lnrm2.lnrm_random，只換 d）。回傳 (n, 3)。"""
    if rng is None:
        rng = np.random.default_rng()
    intensity = rng.uniform(intensity_range[0], intensity_range[1], n_trials)
    d = 0.5 * L / (1.0 + np.exp(-slope * (intensity - midpoint)))   # 同 ogival_d
    t1 = psi + np.exp(rng.normal(mu - d, varZ))
    t2 = psi + np.exp(rng.normal(mu + d, varZ))
    rt = np.minimum(t1, t2)
    correct = (t1 < t2).astype(float)
    return np.column_stack([rt, correct, intensity])


# ============================================================================
# 2. pt 積木：Op
# ============================================================================

class LNRM2A_PointwiseOp(pt.Op):
    __props__ = ()

    def make_node(self, rt, correct, intensity, mu, slope, midpoint, varZ, psi, L):
        rt = pt.as_tensor_variable(rt).astype('float64')
        correct = pt.as_tensor_variable(correct).astype('int32')
        intensity = pt.as_tensor_variable(intensity).astype('float64')
        scalars = [pt.as_tensor_variable(v).astype('float64')
                   for v in (mu, slope, midpoint, varZ, psi, L)]
        return Apply(self, [rt, correct, intensity, *scalars],
                     [pt.TensorType('float64', shape=(None,))()])

    def perform(self, node, inputs, outputs):
        rt, correct, intensity, mu, slope, midpoint, varZ, psi, L = inputs
        f = lambda v: float(np.asarray(v).reshape(-1)[0])
        outputs[0][0] = lnrm2a_pointwise_loglik(
            np.ascontiguousarray(rt), np.ascontiguousarray(correct), np.ascontiguousarray(intensity),
            f(mu), f(slope), f(midpoint), f(varZ), f(psi), f(L))


# ============================================================================
# 3. pm 積木：模型
# ============================================================================

def build_model_lnrm2a(observed_data, tune=4000, draws=4000, chains=8, fix_params=None,
                       random_seed=42, as_observed=False, L=L_MAX_SEPARATION):
    """
    參數：mu, slope, midpoint, varZ, psi（adaptiveSFT_functions.R:186 的 pars 順序）。
    fix_params / as_observed 語意同 model_lnrm2.build_model_lnrm2。
    """
    if fix_params is None:
        fix_params = {}
    rt = observed_data[:, 0].astype('float64')
    correct = observed_data[:, 1].astype('int32')
    intensity = observed_data[:, 2].astype('float64')
    min_rt = float(rt.min())

    with pm.Model() as model:
        # TODO-A5：mu / varZ / psi 先驗照抄 lnrm2.stan:30-31,17
        mu   = fix_params['mu']   if 'mu'   in fix_params else pm.Normal("mu", 0.0, 1.0)
        varZ = fix_params['varZ'] if 'varZ' in fix_params else pm.InverseGamma("varZ", alpha=1.0, beta=0.1)
        psi  = fix_params['psi']  if 'psi'  in fix_params else pm.Uniform("psi", lower=0.0, upper=min_rt)
        # TODO-A3：slope / midpoint 先驗是猜的
        slope    = fix_params['slope']    if 'slope'    in fix_params else pm.Normal("slope", 0.0, 2.0)
        midpoint = fix_params['midpoint'] if 'midpoint' in fix_params else pm.Normal("midpoint", 0.0, 2.0)
        for k in fix_params:
            print(f"  [Fixed] {k}={fix_params[k]:.3f}")

        op = LNRM2A_PointwiseOp()
        if as_observed:
            def race_logp(value, mu_, slope_, midpoint_, varZ_, psi_):
                return op(value, correct, intensity, mu_, slope_, midpoint_, varZ_, psi_, L)
            pm.CustomDist("rt_obs", mu, slope, midpoint, varZ, psi,
                          logp=race_logp, observed=rt)
            pm.Deterministic("log_likelihood",
                             op(rt, correct, intensity, mu, slope, midpoint, varZ, psi, L))
        else:
            log_lik_vec = op(rt, correct, intensity, mu, slope, midpoint, varZ, psi, L)
            pm.Deterministic("log_likelihood", log_lik_vec)
            pm.Potential("obs", pt.sum(log_lik_vec))

        init_vals = {}
        if 'mu'       not in fix_params: init_vals["mu"] = 1.0
        if 'slope'    not in fix_params: init_vals["slope"] = 1.0
        if 'midpoint' not in fix_params: init_vals["midpoint"] = 1.0
        if 'varZ'     not in fix_params: init_vals["varZ"] = 0.5
        if 'psi'      not in fix_params: init_vals["psi"] = 0.3 * min_rt

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          step=pm.DEMetropolisZ(), random_seed=random_seed,
                          initvals=init_vals, progressbar=False)
    return trace


# ============================================================================
# 4. 自我測試
# ============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=3000)
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--chains", type=int, default=8)
    a = ap.parse_args()

    # d 落在 0–10 之間，midpoint 放在強度範圍中間，讓 accuracy 從 ~50% 走到 ~100%
    true_params = {'mu': 1.5, 'slope': 2.0, 'midpoint': 1.5, 'varZ': 0.6, 'psi': 0.12}
    data = lnrm2a_random(a.n, **true_params, rng=np.random.default_rng(42))
    print(f"N={len(data)}  minRT={data[:, 0].min():.3f}  accuracy={data[:, 1].mean():.3f}")

    trace = build_model_lnrm2a(data, tune=a.tune, draws=a.draws, chains=a.chains)
    summary = az.summary(trace, var_names=list(true_params))
    summary.insert(0, "true", [true_params[p] for p in summary.index])
    print(summary[["true", "mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
