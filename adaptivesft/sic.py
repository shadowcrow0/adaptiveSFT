"""
sft::sic 系列的移植（CRAN sft/R/sic.R，sicGroup :1-130、sic :132-152、sic.test :154-187、
siDominance :189-216、mic.test :219-248）。

輸入是四個 cell 的**答對試次 RT** 陣列（R 端呼叫時就是 HH$rt[HH$x==1]）。
KS 類的 p 值全部手算 exp(−2·n·D²)，不經過 scipy.stats.ks_2samp，
這樣跟 R 的 ks.test(exact=FALSE) 單尾漸近公式逐位元相同（R stats/R/ks.test.R）。
"""
import numpy as np
from scipy.stats import f as f_dist, rankdata

__all__ = ["ecdf", "sic", "sic_test", "si_dominance", "mic_test", "sic_group", "MODEL_NAMES"]

MODEL_NAMES = ("ParallelOR", "ParallelAND", "SerialOR", "SerialAND", "Coactive")


def ecdf(sample):
    s = np.sort(np.asarray(sample, dtype=float))
    return lambda t: np.searchsorted(s, np.asarray(t, dtype=float), side="right") / s.size


def _ks_one_sided(x, y, alternative):
    """R ks.test(x, y, alternative, exact=FALSE) 的單尾統計量與漸近 p 值。
    greater：D⁺ = max(F_x − F_y)；less：D⁻ = max(F_y − F_x)；p = exp(−2·n·D²)，n = nx·ny/(nx+ny)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.sort(np.concatenate([x, y]))
    z = ecdf(x)(w) - ecdf(y)(w)
    D = float(np.max(z)) if alternative == "greater" else float(np.max(-z))
    D = max(D, 0.0)
    n = x.size * y.size / (x.size + y.size)
    return D, float(np.exp(-2.0 * n * D * D))


def _harmonic_n(HH, HL, LH, LL):
    return 1.0 / (1.0 / len(HH) + 1.0 / len(HL) + 1.0 / len(LH) + 1.0 / len(LL))    # sic.R:142-143


def sic_test(HH, HL, LH, LL):
    """Houpt–Townsend KS-SIC 檢定（sic.R:154-187）。回傳 dict(positive=(D⁺, p), negative=(D⁻, p))。"""
    RTall = np.unique(np.concatenate([HH, HL, LH, LL]))
    N = _harmonic_n(HH, HL, LH, LL)
    sicall = ecdf(LH)(RTall) + ecdf(HL)(RTall) - ecdf(HH)(RTall) - ecdf(LL)(RTall)   # sic.R:167
    Dplus = max(0.0, float(sicall.max()))
    Dminus = abs(min(0.0, float(sicall.min())))
    return {"positive": (Dplus, float(np.exp(-2 * N * Dplus ** 2))),
            "negative": (Dminus, float(np.exp(-2 * N * Dminus ** 2)))}


def si_dominance(HH, HL, LH, LL):
    """選擇性影響的八個單尾 KS（sic.R:189-216）。回傳 list of (test, statistic, p)，順序同 R。"""
    pairs = [("hh", "hl", HH, HL), ("hh", "lh", HH, LH), ("hl", "ll", HL, LL), ("lh", "ll", LH, LL)]
    rows = []
    for a, b, A, B in pairs:                      # S.a > S.b  ⇔  ks.test(A, B, "greater")
        D, p = _ks_one_sided(A, B, "greater")
        rows.append((f"S.{a} > S.{b}", D, p))
    for a, b, A, B in pairs:                      # S.a < S.b  ⇔  ks.test(A, B, "less")
        D, p = _ks_one_sided(A, B, "less")
        rows.append((f"S.{a} < S.{b}", D, p))
    return rows


def _seq_anova_interaction_F(y, h1, h2):
    """anova(lm(y ~ h1*h2)) 第三列（序列 SS）的 F 與 df。"""
    Xr = np.column_stack([np.ones_like(h1), h1, h2])
    Xf = np.column_stack([Xr, h1 * h2])

    def rss(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(np.sum((y - X @ beta) ** 2))

    df2 = y.size - Xf.shape[1]
    F = (rss(Xr) - rss(Xf)) / 1.0 / (rss(Xf) / df2)
    return F, 1, df2


def mic_test(HH, HL, LH, LL, method="art"):
    """MIC 檢定（sic.R:219-248）。method = 'art'（Adjusted Rank Transform）或 'anova'。"""
    HH, HL, LH, LL = (np.asarray(v, dtype=float) for v in (HH, HL, LH, LL))
    HH, HL, LH, LL = (v[~np.isnan(v)] for v in (HH, HL, LH, LL))
    MIC = (LL.mean() - LH.mean()) - (HL.mean() - HH.mean())
    allrt = np.concatenate([HH, HL, LH, LL])
    n1, n2, n3, n4 = len(HH), len(HL), len(LH), len(LL)
    h1 = np.concatenate([np.ones(n1 + n2), np.zeros(n3 + n4)])
    h2 = np.concatenate([np.ones(n1), np.zeros(n2), np.ones(n3), np.zeros(n4)])
    if method == "art":
        mA0 = allrt[h1 == 0].mean()
        mA1 = allrt[h1 == 1].mean()
        mB0 = allrt[h2 == 0].mean()
        mB1 = allrt[h2 == 1].mean()
        adj = np.round(allrt - (1 - h1) * mA0 - h1 * mA1 - (1 - h2) * mB0 - h2 * mB1, 15)
        y = rankdata(adj, method="average")
    elif method == "anova":
        y = allrt
    else:
        raise ValueError("method 必須是 art / anova")
    F, df1, df2 = _seq_anova_interaction_F(y, h1, h2)
    return {"statistic": float(MIC), "F": float(F), "df": (df1, df2), "p_value": float(f_dist.sf(F, df1, df2)),
            "method": method}


def sic(HH, HL, LH, LL, mictest="art"):
    """sft::sic（sic.R:132-152）。SIC 回傳 (時間點, SIC 值)，可用 np.interp 當階梯函數用。"""
    HH, HL, LH, LL = (np.asarray(v, dtype=float) for v in (HH, HL, LH, LL))
    RTall = np.unique(np.concatenate([HH, HL, LH, LL]))
    sicall = ecdf(LH)(RTall) + ecdf(HL)(RTall) - ecdf(HH)(RTall) - ecdf(LL)(RTall)
    return {
        "SIC": (RTall, sicall),
        "Dominance": si_dominance(HH, HL, LH, LL),
        "SICtest": sic_test(HH, HL, LH, LL),
        "MICtest": mic_test(HH, HL, LH, LL, method=mictest),
        "N": _harmonic_n(HH, HL, LH, LL),
    }


def classify(sic_result, alpha_sic=0.05):
    """sicGroup 的決策表（sic.R:75-105）：哪些檢定顯著 → 哪些模型被拒 → 預測哪個架構。"""
    rejected = np.zeros(5, dtype=bool)   # ParallelOR, ParallelAND, SerialOR, SerialAND, Coactive
    st = sic_result["SICtest"]
    mic = sic_result["MICtest"]
    positive = st["positive"][1] < alpha_sic
    negative = st["negative"][1] < alpha_sic
    if positive:
        rejected[[1, 2]] = True
    if negative:
        rejected[[0, 2]] = True
    mic_sig = mic["p_value"] < alpha_sic
    if mic_sig:
        rejected[[2, 3]] = True
    r = rejected.astype(int).tolist()
    if sum(r) == 0:
        predicted = "SerialOR"
    elif sum(r) == 4:
        predicted = "Coactive"
    elif r[:3] == [0, 1, 1]:
        predicted = "ParallelOR"
    elif r[:3] == [1, 0, 1]:
        predicted = "ParallelAND"
    elif r[:4] == [1, 1, 1, 0]:
        predicted = "SerialAND"
    else:
        predicted = "NA"
    dom_p = [row[2] for row in sic_result["Dominance"]]
    if all(p < 0.05 for p in dom_p[:4]) and not any(p < 0.05 for p in dom_p[4:]):
        si = "Pass"
    elif any(p < 0.05 for p in dom_p[4:]):
        si = "Fail"
    else:
        si = "Ambiguous"
    return {
        "Selective.Influence": si,
        "Positive.SIC": "Significant" if positive else "Nonsignificant",
        "Negative.SIC": "Significant" if negative else "Nonsignificant",
        "MIC": ("Positive" if mic["statistic"] > 0 else "Negative") if mic_sig else "Nonsignificant",
        "Predicted_by": predicted,
        "Rejected.Models": ",".join(m for m, rj in zip(MODEL_NAMES, rejected) if rj),
    }


def sic_group(subject, condition, channel1, channel2, correct, rt, alpha_sic=0.05, mictest="art"):
    """
    sft::sicGroup（sic.R:1-130）。六個等長陣列，欄位意義同 R：Channel 2 = 高 salience、1 = 低。
    每個 (condition, subject) 四格答對 RT 都 > 10 筆才分析。回傳 list of dict（overview 一列）與 sic 結果。
    """
    subject = np.asarray(subject)
    condition = np.asarray(condition)
    channel1 = np.asarray(channel1)
    channel2 = np.asarray(channel2)
    correct = np.asarray(correct).astype(bool)
    rt = np.asarray(rt, dtype=float)
    overview, sics = [], []
    for cond in sorted(set(condition.tolist())):
        for subj in sorted(set(subject[condition == cond].tolist())):
            base = (subject == subj) & (condition == cond) & correct
            cells = {name: rt[base & (channel1 == c1) & (channel2 == c2)]
                     for name, (c1, c2) in (("HH", (2, 2)), ("HL", (2, 1)), ("LH", (1, 2)), ("LL", (1, 1)))}
            if min(len(v) for v in cells.values()) <= 10:
                continue
            res = sic(**cells, mictest=mictest)
            row = {"Subject": subj, "Condition": cond, **classify(res, alpha_sic)}
            overview.append(row)
            sics.append(res)
    return overview, sics
