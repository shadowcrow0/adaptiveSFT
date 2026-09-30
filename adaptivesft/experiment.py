"""
把校準與 DFP 接到真人實驗要的三個零件，全部不依賴 PsychoPy（PsychoPy 腳本只負責畫面與按鍵）。

    Calibrator（Psi 或 LNRM）      校準區塊：next() 給刺激強度 → 受試者作答 → record() → finish() 給 H/L
    dfp_trial_list()               主實驗的 2×2 試次表（HH/HL/LH/LL 隨機化），欄位對得上 sic_group
    參與者資料 → analyze_participant()   DFP 資料 → SIC / MIC / dominance / 預測架構

流程（每個維度各跑一次校準區塊，兩維各得一組 H/L）：

   校準區塊                                   主實驗
   ┌──────────────────────────────┐          ┌──────────────────────────────┐
   │ x = cal.next()               │          │ for row in dfp_trial_list(): │
   │ 播 x → 受試者答 r（0/1）、rt   │   H/L    │   播 (row.c1_level, row.c2_level)│
   │ cal.record(x, r, rt)         │ ───────► │   記 rt, correct              │
   │ … n 試 … cal.finish()        │          │ analyze_participant(csv)     │
   └──────────────────────────────┘          └──────────────────────────────┘

Psi 校準：每試線上更新（psiSimulation_functions.R 的做法），目標是正確率 p_high / p_low。
LNRM 校準：定值刺激法（moc），區塊結束擬合一次 lnrm2 / lnrm2a（NUTS 約 20 s），
          目標是漂移差 h_targ / l_targ（或正確率 acc_high / acc_low）。

強度單位：校準與主實驗都用「物理單位」（ΔE00、dB、角度…）；LNRM 內部要不要縮放由呼叫端決定
（simulateLNRM_ogival.R 用 (x − thres50)/(x_max − thres50)），這裡不替你縮放。
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from .psi import Psi, make_psi, salience_levels
from .sic import classify, sic, sic_group

__all__ = ["PsiCalibrator", "LNRMCalibrator", "dfp_trial_list", "DFP_COLUMNS", "analyze_participant",
           "write_rows", "read_rows"]

DFP_COLUMNS = ("subject", "block", "trial", "condition", "channel1", "channel2", "c1_level", "c2_level",
               "correct", "rt")


# ============================================================================================
# 校準
# ============================================================================================

@dataclass
class CalibrationResult:
    method: str
    dim: str
    n_trials: int
    high: float
    low: float
    params: dict
    warnings: list = field(default_factory=list)
    in_range: bool = True

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


class PsiCalibrator:
    """
    Psi 線上校準（psiSimulation_functions.R 的 Est.Trial.Psi.* 流程 + inv.pm.function）。

        cal = PsiCalibrator("colour", x=(-24, 24, 0.5), a=(-20, 20, 0.5), b=(1, 30, 0.5), d=0.02,
                            p_high=0.90, p_low=0.75)
        for t in range(n):
            x = cal.next()             # 這一試的刺激強度（物理單位）
            r = ...                    # 受試者的反應位元：1 = 答「高」那一邊（同 R 的 r）
            cal.record(x, r, rt)
        res = cal.finish()             # CalibrationResult：high / low / params(alpha, beta)

    x/a/b 三個網格用 (下限, 上限, 步長)；dim 給 "colour" / "orientation" 時可省略網格（用 R 的常數）。
    r 的定義：P(r = 1 | x) 隨 x 遞增（AGRT 的 [bi]/pink 那種「高端」反應）。
    """

    def __init__(self, dim, x=None, a=None, b=None, d=None, p_high=0.99, p_low=0.90, prior=None):
        self.dim = dim
        self.psi: Psi = make_psi(dim if x is None else None, x=x, a=a, b=b, d=d, prior=prior)
        self.p_high, self.p_low = float(p_high), float(p_low)
        self.log = []

    def next(self):
        return float(self.psi.next_intensity)

    def record(self, x, response, rt=None):
        if abs(x - self.psi.next_intensity) > 1e-9:
            raise ValueError(f"record 的 x={x} 不是 next() 給的 {self.psi.next_intensity}（Psi 不支援換題）")
        self.psi.update(int(response))
        a_hat, b_hat = self.psi.estimate()
        self.log.append(dict(trial=len(self.log) + 1, x=float(x), response=int(response), rt=rt,
                             alpha_hat=a_hat, beta_hat=b_hat))

    def finish(self):
        a_hat, b_hat = self.psi.estimate()
        x_range = (float(self.psi.x.min()), float(self.psi.x.max()))
        (high, low), warns = salience_levels(a_hat, b_hat, self.psi.d, [self.p_high, self.p_low], x_range=x_range)
        return CalibrationResult("psi", self.dim, len(self.log), high, low,
                                 dict(alpha=a_hat, beta=b_hat, d=self.psi.d, p_high=self.p_high, p_low=self.p_low),
                                 warns, in_range=not warns)


class LNRMCalibrator:
    """
    定值刺激法 + LNRM（simulateLNRM_ogival.R / find_salience_* 的流程）。

        cal = LNRMCalibrator("colour", levels=[0, 2, 4, 6, 8, 10, 12], n_per_level=15,
                             link="ogival", h_targ=4.0, l_targ=1.0)        # 或 acc_high=.9, acc_low=.75
        for t in range(cal.n_trials):
            x = cal.next()                       # 已隨機化的層級
            correct, rt = ...                    # 1/0，秒
            cal.record(x, correct, rt)
        res = cal.finish()                       # 這一步擬合 LNRM，NUTS 約 20 s

    levels 是物理單位；要縮放（R 的 (x − thres50)/(x_max − thres50)）就在呼叫端做。
    """

    def __init__(self, dim, levels, n_per_level, link="ogival", h_targ=None, l_targ=None,
                 acc_high=None, acc_low=None, L=10.0, seed=None, **fit_kwargs):
        self.dim = dim
        self.levels = [float(v) for v in levels]
        self.link, self.L, self.fit_kwargs = link, L, fit_kwargs
        if (h_targ is None) == (acc_high is None):
            raise ValueError("h_targ/l_targ 與 acc_high/acc_low 二選一")
        self.targets = (dict(h_targ=h_targ, l_targ=l_targ) if h_targ is not None
                        else dict(acc_high=acc_high, acc_low=acc_low))
        rng = np.random.default_rng(seed)
        plan = np.repeat(self.levels, n_per_level)
        rng.shuffle(plan)
        self.plan = [float(v) for v in plan]
        self.n_trials = len(self.plan)
        self.log = []

    def next(self):
        if len(self.log) >= self.n_trials:
            raise StopIteration
        return self.plan[len(self.log)]

    def record(self, x, correct, rt):
        if rt is None or rt <= 0:
            raise ValueError("LNRM 需要正的 rt")
        self.log.append(dict(trial=len(self.log) + 1, x=float(x), correct=int(correct), rt=float(rt)))

    def finish(self, rt_min=0.15, rt_max=5.0):
        from .models import fit_lnrm, make_data
        from .salience import find_salience
        rows = [r for r in self.log if rt_min <= r["rt"] <= rt_max]
        data = make_data([r["x"] for r in rows], [r["rt"] for r in rows], [r["correct"] for r in rows])
        tr = fit_lnrm(data, link=self.link, L=self.L, **self.fit_kwargs)
        res = find_salience(tr, **self.targets)
        high, low = res["high"]["intensity"], res["low"]["intensity"]
        warns = list(res["warnings"])
        lo_lv, hi_lv = min(self.levels), max(self.levels)
        in_range = bool(np.isfinite(high) and np.isfinite(low) and lo_lv <= high <= hi_lv and lo_lv <= low <= hi_lv)
        if not in_range:
            warns.append(f"H={high:.3f} / L={low:.3f} 不在校準層級 [{lo_lv}, {hi_lv}] 內")
        params = {k: float(tr.posterior[k].values.mean()) for k in tr.posterior.data_vars if k != "log_likelihood"}
        params.update(n_used=len(rows), n_trimmed=len(self.log) - len(rows), divergences=int(tr.sample_stats["diverging"].sum()),
                      dropped_high=res["high"]["dropped"], dropped_low=res["low"]["dropped"], **self.targets)
        self.trace = tr
        return CalibrationResult("lnrm_" + self.link, self.dim, len(self.log), high, low, params, warns, in_range)


# ============================================================================================
# DFP 主實驗
# ============================================================================================

def dfp_trial_list(subject, high1, low1, high2, low2, n_per_cell, blocks=1, condition="DFP", seed=None):
    """
    2×2 DFP 試次表：每個 block 內四格各 n_per_cell 試、隨機排列。
    channel1/2 = 2 表示高 salience、1 表示低（同 sft::sicGroup 的編碼）；c1_level / c2_level 是要播的物理強度。
    correct / rt 留空，實驗程式填。
    """
    rng = np.random.default_rng(seed)
    rows = []
    t = 0
    for b in range(1, blocks + 1):
        cells = [(2, 2), (2, 1), (1, 2), (1, 1)] * n_per_cell
        rng.shuffle(cells)
        for ch1, ch2 in cells:
            t += 1
            rows.append(dict(subject=subject, block=b, trial=t, condition=condition, channel1=ch1, channel2=ch2,
                             c1_level=high1 if ch1 == 2 else low1, c2_level=high2 if ch2 == 2 else low2,
                             correct="", rt=""))
    return rows


def write_rows(rows, path, columns=DFP_COLUMNS):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ============================================================================================
# 分析
# ============================================================================================

def analyze_participant(rows, rt_min=0.15, rt_max=5.0, alpha_sic=0.05):
    """
    一位受試者的 DFP 資料（dfp_trial_list 的欄位，correct / rt 已填）→ dict：
    cells（四格答對 RT 數）、accuracy、sic 結果、classify 結果。多個 condition 各算一次。
    """
    keep = []
    for r in rows:
        try:
            rt = float(r["rt"])
            c = int(float(r["correct"]))
        except (TypeError, ValueError):
            continue
        if rt_min <= rt <= rt_max:
            keep.append((r, rt, c))
    out = {}
    for cond in sorted({r["condition"] for r, _, _ in keep}):
        cells, acc = {}, {}
        for name, (c1, c2) in dict(HH=(2, 2), HL=(2, 1), LH=(1, 2), LL=(1, 1)).items():
            sel = [(rt, c) for r, rt, c in keep if r["condition"] == cond
                   and int(r["channel1"]) == c1 and int(r["channel2"]) == c2]
            cells[name] = np.array([rt for rt, c in sel if c == 1])
            acc[name] = float(np.mean([c for _, c in sel])) if sel else float("nan")
        if min(len(v) for v in cells.values()) <= 10:
            out[cond] = dict(error="每格答對試次要 > 10（sft::sicGroup 的門檻）", n=[len(v) for v in cells.values()])
            continue
        s = sic(**cells)
        out[cond] = dict(n_correct={k: int(len(v)) for k, v in cells.items()}, accuracy=acc,
                         mean_rt={k: float(v.mean()) for k, v in cells.items()},
                         Dplus=s["SICtest"]["positive"], Dminus=s["SICtest"]["negative"],
                         MIC=(s["MICtest"]["statistic"], s["MICtest"]["p_value"]),
                         dominance=s["Dominance"], classification=classify(s, alpha_sic), sic=s)
    return out


def report(result):
    lines = []
    for cond, r in result.items():
        lines.append(f"condition {cond}")
        if "error" in r:
            lines.append(f"  {r['error']}: {r['n']}")
            continue
        lines.append("  " + "  ".join(f"{k}: n={r['n_correct'][k]} acc={r['accuracy'][k]:.3f} RT={r['mean_rt'][k]:.3f}"
                                      for k in ("HH", "HL", "LH", "LL")))
        lines.append(f"  D+ = {r['Dplus'][0]:.3f} (p={r['Dplus'][1]:.3f})   D- = {r['Dminus'][0]:.3f} (p={r['Dminus'][1]:.3f})"
                     f"   MIC = {r['MIC'][0]:+.3f} (p={r['MIC'][1]:.3f})")
        c = r["classification"]
        lines.append(f"  selective influence: {c['Selective.Influence']}   predicted architecture: {c['Predicted_by']}"
                     f"   rejected: {c['Rejected.Models'] or '-'}")
    return "\n".join(lines)
