"""
models_v2 —— lnrm 的第二版：likelihood 全部用 PyMC 內建的東西拼出來，不再手寫公式。

    v1（models.py）                              v2（本檔）
    ───────────────────────────────────────      ────────────────────────────────────────────
    log 密度：自己寫 −ln t − ln σ − ½ln2π − …      pm.logp(pm.LogNormal.dist(z, σ), t)
    log 存活：自己寫 ln(½·erfc(…))                 pm.math.log1mexp(pm.logcdf(pm.LogNormal.dist(z, σ), t))
    接進模型：pm.Potential + pm.Deterministic       pm.CustomDist(logp=…, random=…, observed=rt)
    逐題 log-likelihood：手動存 Deterministic       pm.sample(idata_kwargs={"log_likelihood": True})
    後驗預測：沒有                                 pm.sample_posterior_predictive 直接能用（random= 給了賽跑模擬）
    numba Op + DEMetropolisZ 那條路                 拿掉；只留 NUTS

模型本身（lnrm2.stan）沒變：z₁ = μ − d、z₂ = μ + d，答對 = 累積器 1 先到：
    ln L = ln f(t; z_win, σ) + ln S(t; z_lose, σ)，t = rt − ψ
先驗、link、L、ogival_offset、positive_slope、fix_params、attrs 都跟 v1 一樣，所以 salience.find_salience 直接可用。
tests/test_models_v2.py：逐題 logp 與 v1 差 < 1e−6；對 Stan oracle 的後驗平均差 < 0.35 SD。
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .models import LINKS, L_MAX_SEPARATION, d_expr, make_data

__all__ = ["race_logp", "race_random", "fit_lnrm_v2", "make_data"]


def race_logp(rt, mu, d, sigma, psi, correct):
    """
    每題的 log-likelihood（CustomDist 的 logp）。rt 是觀測值；correct 當參數傳進來（0/1 向量）。
    全部由 PyMC 內建的 LogNormal logp / logcdf 組成。
    """
    c = pt.as_tensor_variable(correct).astype("float64")
    z_win = mu - d * (2.0 * c - 1.0)          # correct=1 → μ − d；correct=0 → μ + d
    z_lose = mu + d * (2.0 * c - 1.0)
    t = rt - psi
    log_pdf = pm.logp(pm.LogNormal.dist(mu=z_win, sigma=sigma), t)
    log_sf = pm.math.log1mexp(pm.logcdf(pm.LogNormal.dist(mu=z_lose, sigma=sigma), t))
    return pt.switch(t > 0, log_pdf + log_sf, -np.inf)


def race_random(mu, d, sigma, psi, correct, rng=None, size=None):
    """
    CustomDist 的 random（後驗預測用）：兩個對數常態累積器賽跑，回傳先到者的時間 rt。
    PyMC 以逐題的 signature 呼叫，所以 mu / d / sigma / psi 都已廣播成每題一個值；correct 只是占位，模擬時不用。
    """
    rng = np.random.default_rng() if rng is None else rng
    mu, d, sigma, psi = (np.asarray(v, dtype=float) for v in (mu, d, sigma, psi))
    shape = np.broadcast(mu, d, sigma, psi, np.asarray(correct)).shape if size is None else size
    t1 = psi + np.exp(rng.normal(mu - d, sigma, size=shape))
    t2 = psi + np.exp(rng.normal(mu + d, sigma, size=shape))
    return np.minimum(t1, t2)


def fit_lnrm_v2(data, link="quadratic", L=L_MAX_SEPARATION, fix_params=None, ogival_offset=0.5, positive_slope=True,
                tune=1000, draws=1000, chains=4, random_seed=42, progressbar=False, target_accept=0.9, **sample_kwargs):
    """
    同 models.fit_lnrm（NUTS 版），回傳 arviz.InferenceData；attrs 與 v1 相同。
    多出來的：idata.log_likelihood["obs"]（PyMC 內建）、可以直接 pm.sample_posterior_predictive(idata, model=idata.attrs 無法存 model，
    所以本函式把 model 放在 idata.attrs["model"] 以外的地方：用 build_model() 重建）。
    """
    if link not in LINKS:
        raise ValueError(f"未知的 link: {link!r}，可用 {LINKS}")
    if ogival_offset not in (0.5, 1.0):
        raise ValueError("ogival_offset 必須是 0.5（½L）或 1.0（L）")
    model, info = build_model(data, link=link, L=L, fix_params=fix_params, ogival_offset=ogival_offset,
                              positive_slope=positive_slope)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=random_seed, initvals=info["initvals"],
                          progressbar=progressbar, target_accept=target_accept,
                          idata_kwargs={"log_likelihood": True}, **sample_kwargs)
    for k, v in info["attrs"].items():
        idata.attrs[k] = v
    return idata


def build_model(data, link="quadratic", L=L_MAX_SEPARATION, fix_params=None, ogival_offset=0.5, positive_slope=True):
    """建 pm.Model（給 fit_lnrm_v2 與後驗預測用）。回傳 (model, info)；info 有 initvals 與要寫進 idata.attrs 的東西。"""
    estimate_L = isinstance(L, str) and L == "estimate"
    fix_params = dict(fix_params or {})
    data = np.asarray(data, dtype=float)
    rt, correct, x = data[:, 0], data[:, 1], data[:, 2]
    min_rt = float(rt.min())
    initvals, p = {}, {}

    with pm.Model() as model:
        def param(name, dist, init):
            if name in fix_params:
                p[name] = float(fix_params[name])
            else:
                p[name] = dist()
                initvals[name] = init

        param("mu", lambda: pm.Normal("mu", 0.0, 1.0), 1.0)                        # lnrm2.stan:31
        param("varZ", lambda: pm.InverseGamma("varZ", alpha=1.0, beta=0.1), 0.5)   # lnrm2.stan:30（SD）
        param("psi", lambda: pm.Uniform("psi", lower=0.0, upper=min_rt), 0.3 * min_rt)   # lnrm2.stan:17
        if link in ("none", "linear", "quadratic"):
            param("alpha", lambda: pm.Normal("alpha", 0.0, 2.0), 0.5)              # lnrm2.stan:32
        if link == "quadratic":
            param("alpha2", lambda: pm.Normal("alpha2", 0.0, 1.0), -0.1)           # lnrm2.stan:33
        if link == "ogival":
            if positive_slope:
                param("slope", lambda: pm.HalfNormal("slope", 2.0), 1.0)
            else:
                param("slope", lambda: pm.Normal("slope", 0.0, 2.0), 1.0)
            param("midpoint", lambda: pm.Normal("midpoint", 0.0, 2.0), float(np.mean(x)))
            if estimate_L:
                param("L", lambda: pm.HalfNormal("L", 10.0), L_MAX_SEPARATION)

        L_used = p["L"] if estimate_L else float(L)
        x_t = pm.Data("intensity", x)
        c_t = pm.Data("correct", correct)
        d = d_expr(link, x_t, p, L_used) * (2.0 * ogival_offset if link == "ogival" else 1.0)
        pm.CustomDist("obs", p["mu"], d, p["varZ"], p["psi"], c_t, logp=race_logp, random=race_random, observed=rt)

    attrs = dict(link=link, L="estimate" if estimate_L else float(L), ogival_offset=float(ogival_offset),
                 positive_slope=bool(positive_slope), sampler="nuts", min_rt=min_rt,
                 fix_params={k: float(v) for k, v in fix_params.items()},
                 intensity_levels=[float(v) for v in np.unique(x)], implementation="v2")
    return model, dict(initvals=initvals, attrs=attrs)
