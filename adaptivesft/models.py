"""
四個 LNRM 模型的 PyMC 版，共用 race.py 的核心，只差「強度 → 難度 d」那條曲線。

    Stan 檔          link         d(x)                                      repo 裡有沒有
    lnrm0.stan       "none"       α                （每層各自擬合，無強度項）   沒有；用法見 simulateLNRM_ogival.R:150-162
    lnrm1.stan       "linear"     α·x                                        沒有；adaptiveSFT_functions.R:210
    lnrm2.stan       "quadratic"  α·x + α₂·x²      （lnrm2.stan:23-24）        有
    lnrm2a.stan      "ogival"     ½·L·inv_logit(slope·(x − midpoint))        沒有；model_lnrm2a.py 的重建

    z1 = mu − d,  z2 = mu + d   →   z2 − z1 = 2d。
    find_salience 的目標值 h_targ / l_targ 指的是 z2 − z1（= 2d），不是 d：
      多項式：R :224 的註解「a2 i² + a1 i − ½ h_targ」；ogival：R :194 的 logit(targ / L)。

先驗照 lnrm2.stan:30-33 與宣告（:13-17）；slope / midpoint 照 model_lnrm2a.py 的猜測 Normal(0, 2)
（原檔遺失，issue.md S2）。varZ 是 SD（race.py 檔頭）。

取樣器：自訂 pt.Op 沒有梯度，用 DEMetropolisZ（同 model_lnrm2.py）。
資料格式：(N, 3) numpy array = [rt, correct, intensity]，同 model_lnrm2.py；`make_data()` 對應 R 的 dataframe2stan。
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph import Apply

from .race import lnrm_pointwise_loglik

__all__ = ["LINKS", "make_data", "d_expr", "LNRM_PointwiseOp", "fit_lnrm", "fit_lnrm0_by_level"]

LINKS = ("none", "linear", "quadratic", "ogival")
L_MAX_SEPARATION = 10.0        # simulateLNRM_ogival.R:26  L <- 10 # max separation


def make_data(intensity, rt, correct):
    """R 的 dataframe2stan（adaptiveSFT_functions.R:168-173）：整理成 (N, 3) [rt, correct, intensity]。"""
    rt = np.asarray(rt, dtype=float)
    correct = np.asarray(correct, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if not (len(rt) == len(correct) == len(intensity)):
        raise ValueError("rt / correct / intensity 長度必須一致")
    if np.any(rt <= 0):
        raise ValueError("rt 必須全為正（lnrm2.stan:6 real<lower=0>）")
    return np.column_stack([rt, correct, intensity])


def d_expr(link, x, p, L=L_MAX_SEPARATION):
    """依 link 組出每題的 d。x 可以是 numpy 或 pytensor；p 是參數 dict。"""
    if link == "none":
        return p["alpha"] * pt.ones_like(pt.as_tensor_variable(x))
    if link == "linear":
        return p["alpha"] * x
    if link == "quadratic":
        return p["alpha"] * x + p["alpha2"] * x ** 2
    if link == "ogival":
        return 0.5 * L * pm.math.invlogit(p["slope"] * (x - p["midpoint"]))
    raise ValueError(f"未知的 link: {link!r}，可用 {LINKS}")


def d_numpy(link, x, p, L=L_MAX_SEPARATION):
    """d_expr 的 numpy 版，給 salience.py 與測試用（p 的值可以是純量或 draw 向量）。"""
    x = np.asarray(x, dtype=float)
    if link == "none":
        return p["alpha"] * np.ones_like(x)
    if link == "linear":
        return p["alpha"] * x
    if link == "quadratic":
        return p["alpha"] * x + p["alpha2"] * x ** 2
    if link == "ogival":
        return 0.5 * L / (1.0 + np.exp(-p["slope"] * (x - p["midpoint"])))
    raise ValueError(f"未知的 link: {link!r}，可用 {LINKS}")


class LNRM_PointwiseOp(pt.Op):
    """把 numba 的逐題 log-likelihood 包成 PyTensor 節點。輸入 d 是向量（每題一個）。"""
    __props__ = ()

    def make_node(self, rt, correct, d, mu, varZ, psi):
        rt = pt.as_tensor_variable(rt).astype("float64")
        correct = pt.as_tensor_variable(correct).astype("int32")
        d = pt.as_tensor_variable(d).astype("float64")
        scalars = [pt.as_tensor_variable(v).astype("float64") for v in (mu, varZ, psi)]
        return Apply(self, [rt, correct, d, *scalars], [pt.TensorType("float64", shape=(None,))()])

    def perform(self, node, inputs, outputs):
        rt, correct, d, mu, varZ, psi = inputs
        f = lambda v: float(np.asarray(v).reshape(-1)[0])
        outputs[0][0] = lnrm_pointwise_loglik(
            np.ascontiguousarray(rt), np.ascontiguousarray(correct),
            np.ascontiguousarray(np.asarray(d, dtype=float).reshape(-1)), f(mu), f(varZ), f(psi))


def fit_lnrm(data, link="quadratic", L=L_MAX_SEPARATION, fix_params=None,
             tune=3000, draws=3000, chains=8, random_seed=42, progressbar=False, **sample_kwargs):
    """
    擬合 LNRM，回傳 arviz.InferenceData；attrs 裡有 link / L / min_rt，salience.py 會用到。

    fix_params：把任一參數固定為常數，例 {'alpha2': 0.0}（= lnrm1）、{'psi': 0.1}。
    """
    if link not in LINKS:
        raise ValueError(f"未知的 link: {link!r}，可用 {LINKS}")
    fix_params = dict(fix_params or {})
    data = np.asarray(data, dtype=float)
    rt = data[:, 0]
    correct = data[:, 1].astype("int32")
    x = data[:, 2]
    min_rt = float(rt.min())

    with pm.Model() as model:
        p = {}

        def param(name, dist, init):
            if name in fix_params:
                p[name] = float(fix_params[name])
            else:
                p[name] = dist()
                init_vals[name] = init

        init_vals = {}
        param("mu", lambda: pm.Normal("mu", 0.0, 1.0), 1.0)                       # lnrm2.stan:31
        param("varZ", lambda: pm.InverseGamma("varZ", alpha=1.0, beta=0.1), 0.5)  # lnrm2.stan:30，SD
        param("psi", lambda: pm.Uniform("psi", lower=0.0, upper=min_rt), 0.3 * min_rt)  # lnrm2.stan:17
        if link in ("none", "linear", "quadratic"):
            param("alpha", lambda: pm.Normal("alpha", 0.0, 2.0), 0.5)             # lnrm2.stan:32
        if link == "quadratic":
            param("alpha2", lambda: pm.Normal("alpha2", 0.0, 1.0), -0.1)          # lnrm2.stan:33
        if link == "ogival":
            param("slope", lambda: pm.Normal("slope", 0.0, 2.0), 1.0)             # model_lnrm2a.py 的猜測
            param("midpoint", lambda: pm.Normal("midpoint", 0.0, 2.0), float(np.mean(x)))

        d = d_expr(link, pt.as_tensor_variable(x), p, L)
        log_lik = LNRM_PointwiseOp()(rt, correct, d, p["mu"], p["varZ"], p["psi"])
        pm.Deterministic("log_likelihood", log_lik)
        pm.Potential("obs", pt.sum(log_lik))                                      # 對應 Stan 的 target +=

        trace = pm.sample(draws=draws, tune=tune, chains=chains, step=pm.DEMetropolisZ(),
                          random_seed=random_seed, initvals=init_vals,
                          progressbar=progressbar, **sample_kwargs)

    trace.attrs["link"] = link
    trace.attrs["L"] = float(L)
    trace.attrs["min_rt"] = min_rt
    trace.attrs["fix_params"] = {k: float(v) for k, v in fix_params.items()}
    trace.attrs["intensity_levels"] = [float(v) for v in np.unique(x)]
    return trace


def fit_lnrm0_by_level(data, **kwargs):
    """
    simulateLNRM_ogival.R:150-162 的做法：每個強度層各自擬合一個無強度項的賽跑（lnrm0.stan），
    回傳每層的 (intensity, 後驗平均 d, trace)。R :160-161 取 mean(mu[,2] − mu[,1]) / 2，
    這裡 z2 − z1 = 2α，所以就是 mean(α)。
    """
    data = np.asarray(data, dtype=float)
    out = []
    for lv in np.unique(data[:, 2]):
        sub = data[data[:, 2] == lv]
        tr = fit_lnrm(sub, link="none", **kwargs)
        out.append((float(lv), float(tr.posterior["alpha"].values.mean()), tr))
    return out
