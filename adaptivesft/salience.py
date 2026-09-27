"""
從 LNRM 後驗反解 high / low salience 的刺激強度。

介面沿用 adaptiveSFT_functions.R：h_targ / l_targ 是**目標的累積器分離 z2 − z1（= 2d）**，
漂移差單位，不是正確率（simulateLNRM_ogival.R:24-25 用 1.3 / 8.0，配 L = 10）。
要用正確率當目標，先用 accuracy_to_targ() 換算：

    P(correct | x) = Φ( (z2 − z1) / (varZ·√2) )        （兩累積器共用 varZ 時的封閉解）
    targ = varZ · √2 · Φ⁻¹(p)

R1（issue.md）在這裡定案 —— `if (post.diff$alpha2 < 0)`（adaptiveSFT_functions.R:228）的語意
用 alpha2_rule 參數明講，預設 "all_draws"：

    all_draws   刪掉 if，全部 draw 代進公式，判別式 < 0 的 draw 變 NaN，取 nanmean。
                舊 R（< 4.2）只要第一筆 draw 的 alpha2 < 0 就是這條路，數值一個 bit 都不差。
    first_draw  舊 R 的字面行為：只看第一筆；第一筆 ≥ 0 時 R 會 object not found，這裡 raise。
    all         全部 alpha2 < 0 才算，否則 raise。
    mean        mean(alpha2) < 0 才算，否則 raise。
    filter      只用 alpha2 < 0 的 draw。

plan_r_modernization.md P4：R :279 的 mean(na.rm=TRUE) 靜默丟掉無解的 draw，這裡回報 dropped 比例，
超過 5% 就放進 warnings。
"""
import numpy as np
from scipy.special import logit
from scipy.stats import norm

from .models import L_MAX_SEPARATION, d_numpy

__all__ = [
    "ALPHA2_RULES", "accuracy_to_targ", "targ_to_accuracy",
    "find_salience_polynomial", "find_salience_ogival", "find_salience", "summarize",
]

ALPHA2_RULES = ("all_draws", "first_draw", "all", "mean", "filter")
_SQRT2 = np.sqrt(2.0)


def _flat(trace, name):
    return np.asarray(trace.posterior[name].values).ravel()


def _param(trace, name):
    """估的參數從後驗拿；fix_params 固定的從 attrs 拿（廣播成純量）。"""
    if name in trace.posterior:
        return _flat(trace, name)
    return float(trace.attrs["fix_params"][name])


def accuracy_to_targ(p, varZ):
    """目標正確率 → 目標分離 z2 − z1。p ∈ (0.5, 1)。"""
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0.5) or np.any(p >= 1.0):
        raise ValueError("目標正確率必須落在 (0.5, 1) 開區間")
    return _SQRT2 * np.asarray(varZ, dtype=float) * norm.ppf(p)


def targ_to_accuracy(targ, varZ):
    return norm.cdf(np.asarray(targ, dtype=float) / (np.asarray(varZ, dtype=float) * _SQRT2))


def _pack(name, targ, x_draws, warnings):
    finite = np.isfinite(x_draws)
    dropped = 1.0 - float(finite.mean()) if x_draws.size else 1.0
    if dropped > 0.05:
        warnings.append(f"{name}: {dropped:.1%} 的 draw 無解被丟掉（R :279 的 na.rm=TRUE 不會告訴你這件事）")
    return {
        "targ": float(targ),
        "intensity": float(np.nanmean(x_draws)) if finite.any() else float("nan"),   # R :279-280 用 mean
        "median": float(np.nanmedian(x_draws)) if finite.any() else float("nan"),
        "ci90": tuple(float(v) for v in np.nanpercentile(x_draws, [5, 95])) if finite.any() else (np.nan, np.nan),
        "dropped": dropped,
        "draws": x_draws,
    }


def _invert_quadratic(alpha, alpha2, targ):
    """adaptiveSFT_functions.R:229-232 逐字：(−α/α₂ − √((α/α₂)² + 2·targ/α₂)) / 2。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = (alpha / alpha2) ** 2 + 2.0 / alpha2 * targ
        return (-alpha / alpha2 - np.sqrt(disc)) / 2.0


def find_salience_polynomial(trace, h_targ, l_targ, alpha2_rule="all_draws"):
    """lnrm2 / lnrm1 的反解（R :203-281）。回傳 dict(high=…, low=…, warnings=[…])。"""
    if alpha2_rule not in ALPHA2_RULES:
        raise ValueError(f"alpha2_rule 必須是 {ALPHA2_RULES}")
    link = trace.attrs["link"]
    warnings = []
    alpha = np.atleast_1d(_param(trace, "alpha")).astype(float)

    fixed_zero_alpha2 = link == "quadratic" and trace.attrs["fix_params"].get("alpha2") == 0.0
    if link == "linear" or fixed_zero_alpha2:
        # lnrm1：α·x = targ/2
        out = {name: _pack(name, t, t / (2.0 * alpha), warnings) for name, t in (("high", h_targ), ("low", l_targ))}
        out["warnings"] = warnings
        out["rule"] = "linear"
        return out
    if link != "quadratic":
        raise ValueError(f"find_salience_polynomial 只接受 link = linear / quadratic，拿到 {link!r}")

    alpha2 = np.atleast_1d(_param(trace, "alpha2")).astype(float)
    alpha, alpha2 = np.broadcast_arrays(alpha, alpha2)
    if alpha2_rule == "first_draw" and not alpha2[0] < 0:
        raise ValueError("alpha2_rule='first_draw'：第一筆 draw 的 alpha2 ≥ 0，舊 R 在這裡會 object not found")
    if alpha2_rule == "all" and not np.all(alpha2 < 0):
        raise ValueError("alpha2_rule='all'：不是所有 draw 的 alpha2 都 < 0")
    if alpha2_rule == "mean" and not np.mean(alpha2) < 0:
        raise ValueError("alpha2_rule='mean'：mean(alpha2) ≥ 0")
    keep = alpha2 < 0 if alpha2_rule == "filter" else np.ones_like(alpha2, dtype=bool)
    if alpha2_rule == "filter":
        warnings.append(f"alpha2_rule='filter'：只用 {keep.mean():.1%} 的 draw（alpha2 < 0）")

    out = {}
    for name, t in (("high", h_targ), ("low", l_targ)):
        x = _invert_quadratic(alpha[keep], alpha2[keep], t)
        out[name] = _pack(name, t, x, warnings)
    out["warnings"] = warnings
    out["rule"] = alpha2_rule
    return out


def find_salience_ogival(trace, h_targ, l_targ, L=None):
    """lnrm2a 的反解（R :180-199）：x = logit(targ / L) / slope + midpoint，逐 draw。targ ≥ L 無解。"""
    if trace.attrs["link"] != "ogival":
        raise ValueError("find_salience_ogival 只接受 link='ogival'")
    L = float(trace.attrs["L"]) if L is None else float(L)
    slope = np.atleast_1d(_param(trace, "slope")).astype(float)
    midpoint = np.atleast_1d(_param(trace, "midpoint")).astype(float)
    slope, midpoint = np.broadcast_arrays(slope, midpoint)
    warnings = []
    out = {}
    for name, t in (("high", h_targ), ("low", l_targ)):
        frac = t / L
        if not 0.0 < frac < 1.0:
            warnings.append(f"{name}: targ={t} 不在 (0, L={L}) 內，ogival 永遠到不了")
            x = np.full(slope.shape, np.nan)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                x = logit(frac) / slope + midpoint
            x = np.where(slope > 0, x, np.nan)   # slope ≤ 0 的 draw：曲線反向，反解沒意義
        out[name] = _pack(name, t, x, warnings)
    out["warnings"] = warnings
    out["rule"] = "ogival"
    return out


def find_salience(trace, h_targ=None, l_targ=None, acc_high=None, acc_low=None, **kw):
    """
    依 trace.attrs['link'] 自動分派。目標可以給分離（h_targ / l_targ）或正確率（acc_high / acc_low）；
    給正確率時逐 draw 用該 draw 的 varZ 換算（所以 draw 之間 targ 不同），並把回代正確率放進結果。
    """
    link = trace.attrs["link"]
    if (h_targ is None) == (acc_high is None):
        raise ValueError("h_targ/l_targ 與 acc_high/acc_low 二選一")
    if acc_high is not None:
        varZ = np.atleast_1d(_param(trace, "varZ")).astype(float)
        h_targ = accuracy_to_targ(acc_high, varZ)
        l_targ = accuracy_to_targ(acc_low, varZ)
    fn = find_salience_ogival if link == "ogival" else find_salience_polynomial
    if acc_high is None:
        res = fn(trace, h_targ, l_targ, **kw)
        varZ = np.atleast_1d(_param(trace, "varZ")).astype(float)
        for name in ("high", "low"):
            res[name]["implied_accuracy"] = float(np.median(targ_to_accuracy(res[name]["targ"], varZ)))
        return res
    # 正確率介面：targ 是 draw 向量；逐 draw 反解後再包
    res = {"warnings": [], "rule": link}
    params = {k: np.atleast_1d(_param(trace, k)).astype(float)
              for k in trace.posterior.data_vars if k != "log_likelihood"}
    for name, targ, p in (("high", h_targ, acc_high), ("low", l_targ, acc_low)):
        if link == "ogival":
            L = float(trace.attrs["L"])
            with np.errstate(divide="ignore", invalid="ignore"):
                x = logit(targ / L) / params["slope"] + params["midpoint"]
            x = np.where((targ > 0) & (targ < L) & (params["slope"] > 0), x, np.nan)
        elif link == "linear":
            x = targ / (2.0 * params["alpha"])
        else:
            x = _invert_quadratic(params["alpha"], params["alpha2"], targ)
        res[name] = _pack(name, float(np.median(targ)), x, res["warnings"])
        res[name]["target_accuracy"] = float(p)
    return res


def summarize(res):
    lines = [f"rule = {res['rule']}"]
    for name in ("high", "low"):
        r = res[name]
        lo, hi = r["ci90"]
        extra = f"  回代acc {r['implied_accuracy']:.3f}" if "implied_accuracy" in r else ""
        lines.append(f"  {name:5} targ={r['targ']:6.3f}  x={r['intensity']:8.4f}  "
                     f"90%[{lo:8.4f}, {hi:8.4f}]  dropped={r['dropped']:.1%}{extra}")
    for w in res["warnings"]:
        lines.append(f"  ⚠ {w}")
    return "\n".join(lines)
