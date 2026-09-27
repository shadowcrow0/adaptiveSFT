"""sic.py：KS 單尾與 R 公式一致（對 scipy asymp 交叉）；五種架構的簽名；sicGroup 決策表；若有 R oracle 則逐位元比對。"""
import json
import os

import numpy as np
import pytest
from scipy.stats import ks_2samp

from adaptivesft.ddm import dfp_ddm
from adaptivesft.sic import _ks_one_sided, classify, mic_test, sic, sic_group, sic_test

HERE = os.path.dirname(os.path.abspath(__file__))


def _cells(arch, rule, H=3.0, L=1.0, a=3.0, n=250, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, (c1, c2) in dict(HH=(H, H), HL=(H, L), LH=(L, H), LL=(L, L)).items():
        rt, cr = dfp_ddm(n, c1, c2, a, 0.1, 0.2, arch, rule, rng=rng)
        out[name] = rt[cr == 1]
    return out


def test_one_sided_ks_matches_scipy_asymptotic():
    rng = np.random.default_rng(5)
    x, y = rng.normal(0, 1, 120), rng.normal(0.4, 1, 90)
    n = x.size * y.size / (x.size + y.size)
    for alt in ("greater", "less"):
        D, p = _ks_one_sided(x, y, alt)
        ref = ks_2samp(x, y, alternative=alt, method="asymp")
        assert abs(D - ref.statistic) < 1e-12
        # p 值故意不對 scipy：scipy 的單尾漸近式帶修正項，R ks.test(exact=FALSE) 是裸的 exp(−2·n·D²)，
        # sft::siDominance 用的是後者；逐位元比對在 test_against_r_oracle。
        assert abs(p - np.exp(-2 * n * D * D)) < 1e-15


def test_parallel_or_signature():
    s = sic(**_cells("PAR", "OR"))
    assert s["SICtest"]["positive"][1] < 0.05 and s["SICtest"]["negative"][1] > 0.05
    assert s["MICtest"]["statistic"] > 0 and s["MICtest"]["p_value"] < 0.05
    c = classify(s)
    assert c["Predicted_by"] == "ParallelOR" and c["Selective.Influence"] == "Pass"


def test_parallel_and_signature():
    s = sic(**_cells("PAR", "AND"))
    assert s["SICtest"]["negative"][1] < 0.05 and s["SICtest"]["positive"][1] > 0.05
    assert s["MICtest"]["statistic"] < 0


def test_coactive_signature():
    s = sic(**_cells("COA", None))
    assert s["SICtest"]["positive"][1] < 0.05 and s["MICtest"]["statistic"] > 0


def test_mic_anova_and_art_agree_in_sign():
    c = _cells("PAR", "OR")
    art = mic_test(**c, method="art")
    aov = mic_test(**c, method="anova")
    assert np.sign(art["statistic"]) == np.sign(aov["statistic"]) and art["df"][0] == 1


def test_classify_decision_table():
    def fake(pp, pn, pm, mic=1.0):
        return {"SICtest": {"positive": (0.1, pp), "negative": (0.1, pn)},
                "MICtest": {"statistic": mic, "p_value": pm},
                "Dominance": [("", 0, 0.01)] * 4 + [("", 0, 0.9)] * 4}
    assert classify(fake(0.9, 0.9, 0.9))["Predicted_by"] == "SerialOR"
    assert classify(fake(0.01, 0.9, 0.01))["Predicted_by"] == "ParallelOR"
    assert classify(fake(0.9, 0.01, 0.01, mic=-1))["Predicted_by"] == "ParallelAND"
    assert classify(fake(0.01, 0.01, 0.9))["Predicted_by"] == "SerialAND"
    assert classify(fake(0.01, 0.01, 0.01))["Predicted_by"] == "Coactive"
    # sic.R:75-105 的表：D⁺ 單獨顯著就是 ParallelOR（不管 MIC）；只有 MIC 顯著 → 拒 SerialOR/SerialAND → NA
    assert classify(fake(0.01, 0.9, 0.9))["Predicted_by"] == "ParallelOR"
    assert classify(fake(0.9, 0.9, 0.01))["Predicted_by"] == "NA"
    assert classify(fake(0.9, 0.9, 0.9))["Selective.Influence"] == "Pass"


def test_sic_group_long_format():
    n = 60
    rows = {"subject": [], "condition": [], "channel1": [], "channel2": [], "correct": [], "rt": []}
    rng = np.random.default_rng(6)
    for subj in (1, 2):
        for (c1, c2), (d1, d2) in zip(((2, 2), (2, 1), (1, 2), (1, 1)), ((3, 3), (3, 1), (1, 3), (1, 1))):
            rt, cr = dfp_ddm(n, d1, d2, 3.0, 0.1, 0.2, "PAR", "OR", rng=rng)
            rows["subject"] += [subj] * n
            rows["condition"] += ["PAR.OR"] * n
            rows["channel1"] += [c1] * n
            rows["channel2"] += [c2] * n
            rows["correct"] += cr.tolist()
            rows["rt"] += rt.tolist()
    overview, sics = sic_group(**rows)
    assert [r["Subject"] for r in overview] == [1, 2] and len(sics) == 2
    assert all(r["Predicted_by"] == "ParallelOR" for r in overview)


@pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "data", "sic_r_oracle.json")),
                    reason="沒有 R 的 oracle（tests/data/make_sic_oracle.R 產生）")
def test_against_r_oracle():
    with open(os.path.join(HERE, "data", "sic_r_oracle.json"), encoding="utf-8") as f:
        oracle = json.load(f)
    cells = {k: np.asarray(oracle["cells"][k], dtype=float) for k in ("HH", "HL", "LH", "LL")}
    s = sic(**cells)
    t, v = s["SIC"]
    assert np.allclose(t, oracle["SIC_times"], atol=0, rtol=0)
    assert np.allclose(v, oracle["SIC_values"], atol=1e-12, rtol=0)
    assert abs(s["SICtest"]["positive"][0] - oracle["Dplus"]) < 1e-12
    assert abs(s["SICtest"]["positive"][1] - oracle["p_Dplus"]) < 1e-12
    assert abs(s["SICtest"]["negative"][0] - oracle["Dminus"]) < 1e-12
    assert abs(s["SICtest"]["negative"][1] - oracle["p_Dminus"]) < 1e-12
    for row, (stat, p) in zip(s["Dominance"], zip(oracle["dom_stat"], oracle["dom_p"])):
        assert abs(row[1] - stat) < 1e-12 and abs(row[2] - p) < 1e-10
    assert abs(s["MICtest"]["statistic"] - oracle["MIC"]) < 1e-12
    assert abs(s["MICtest"]["p_value"] - oracle["p_MIC_art"]) < 1e-8
    assert abs(mic_test(**cells, method="anova")["p_value"] - oracle["p_MIC_anova"]) < 1e-8
