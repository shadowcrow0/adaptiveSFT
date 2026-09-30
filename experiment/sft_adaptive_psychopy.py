#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自適應 SFT 實驗骨架（PsychoPy）：兩個校準區塊 → DFP 主實驗，邏輯全部來自 adaptivesft.experiment，
這個檔只負責畫面、聲音、按鍵、存檔。

    區塊 1  校維度 1（例：顏色）   Psi 線上 或 LNRM 定值刺激法        → H1 / L1
    區塊 2  校維度 2（例：聲音）   同上                              → H2 / L2
    區塊 3  DFP 主實驗            HH / HL / LH / LL × n_per_cell     → csv → scripts/analyze_participant.py

要改的只有兩個函式（標 TODO）：
    draw_stimulus(level1, level2)     把兩個維度的物理強度變成畫面 / 聲音
    calib_response_bit(key, level)    校準試次的按鍵 → Psi 要的 r（1 = 「高」那一邊）或 LNRM 要的 correct
以及最上面的參數區。這個容器沒有 PsychoPy，本檔只做過 py_compile；請在實驗機器上試跑。
"""
import os
import sys

from psychopy import core, data, event, gui, visual

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptivesft.experiment import LNRMCalibrator, PsiCalibrator, dfp_trial_list, write_rows  # noqa: E402

# ─────────────────────────── 參數（TODO：照你的作業改） ───────────────────────────
CALIB_METHOD = "psi"                 # "psi"（線上）或 "lnrm"（定值刺激法，區塊結束擬合 ~20 s）
DIMS = {
    # name:  psi 用的網格 (下限, 上限, 步長) 與 lapse；lnrm 用的層級
    "dim1": dict(psi=dict(x=(-24.0, 24.0, 0.5), a=(-20.0, 20.0, 0.5), b=(1.0, 30.0, 0.5), d=0.02),
                 lnrm=dict(levels=[-24, -16, -8, -4, 0, 4, 8, 16, 24], n_per_level=12),
                 fixed_other=0.0),   # 校這一維時另一維固定的值
    "dim2": dict(psi=dict(x=(-18.0, 18.0, 0.5), a=(-15.0, 15.0, 0.5), b=(1.0, 20.0, 0.5), d=0.02),
                 lnrm=dict(levels=[-18, -12, -6, -3, 0, 3, 6, 12, 18], n_per_level=12),
                 fixed_other=0.0),
}
N_CALIB_PSI = 144                    # Psi 每維試次（Glavan 用 144；GRTv3_ada 用 72）
PSI_TARGETS = dict(p_high=0.90, p_low=0.75)             # 正確率目標（Psi 路）
LNRM_TARGETS = dict(h_targ=4.0, l_targ=1.0)             # 漂移差目標（LNRM 路）；或 dict(acc_high=.9, acc_low=.75)
N_PER_CELL = 100                     # DFP 每格試次
DFP_BLOCKS = 2
KEYS = {"low": "f", "high": "j"}     # 校準：f = 低端、j = 高端；DFP：照你的作業定義
RT_MAX = 3.0
# ─────────────────────────────────────────────────────────────────────────────────

info = {"participant": "", "session": "001"}
if not gui.DlgFromDict(info, title="adaptive SFT").OK:
    core.quit()
subj = info["participant"]
out_dir = os.path.join("data", subj)
os.makedirs(out_dir, exist_ok=True)

win = visual.Window(fullscr=True, color="grey", units="deg")
msg = visual.TextStim(win, text="", height=0.8, wrapWidth=25)
clock = core.Clock()


def draw_stimulus(level1, level2):
    """TODO：把 (dim1 強度, dim2 強度) 畫出來 / 播出來。回傳後才開始計時。"""
    msg.text = f"dim1 = {level1:.2f}   dim2 = {level2:.2f}"     # 佔位
    msg.draw()
    win.flip()


def calib_response_bit(key, level):
    """TODO：校準試次。Psi 要「答高端 = 1」；LNRM 要「答對 = 1」（答對 = 按的方向與 level 的符號一致）。"""
    high = key == KEYS["high"]
    if CALIB_METHOD == "psi":
        return int(high)
    return int(high == (level > 0))


def show(text, wait_key="space"):
    msg.text = text
    msg.draw()
    win.flip()
    event.waitKeys(keyList=[wait_key, "escape"]) or core.quit()


def get_key():
    clock.reset()
    keys = event.waitKeys(maxWait=RT_MAX, keyList=list(KEYS.values()) + ["escape"], timeStamped=clock)
    if not keys:
        return None, None
    k, t = keys[0]
    if k == "escape":
        core.quit()
    return k, t


def calibrate(dim):
    cfg = DIMS[dim]
    other = [d for d in DIMS if d != dim][0]
    if CALIB_METHOD == "psi":
        cal = PsiCalibrator(dim, **cfg["psi"], **PSI_TARGETS)
        n = N_CALIB_PSI
    else:
        cal = LNRMCalibrator(dim, cfg["lnrm"]["levels"], cfg["lnrm"]["n_per_level"], link="ogival",
                             **LNRM_TARGETS, seed=hash(subj + dim) % 2**32)
        n = cal.n_trials
    show(f"校準區塊：{dim}\n\n按 {KEYS['low']} = 低、{KEYS['high']} = 高\n\n按空白鍵開始")
    log = []
    for t in range(n):
        x = cal.next()
        levels = {dim: x, other: DIMS[other]["fixed_other"]}
        draw_stimulus(levels["dim1"], levels["dim2"])
        key, rt = get_key()
        win.flip()
        if key is None:                       # 逾時：Psi 不更新（下一試同一題）；LNRM 記 NaN 跳過
            log.append(dict(trial=t + 1, x=x, key="", rt="", bit=""))
            if CALIB_METHOD == "lnrm":
                cal.log.append(dict(trial=len(cal.log) + 1, x=float(x), correct=0, rt=RT_MAX + 1))  # finish() 會修剪掉
            continue
        bit = calib_response_bit(key, x)
        cal.record(x, bit, rt)
        log.append(dict(trial=t + 1, x=x, key=key, rt=rt, bit=bit))
        core.wait(0.3)
    write_rows(log, os.path.join(out_dir, f"{subj}_calib_{dim}.csv"), columns=("trial", "x", "key", "rt", "bit"))
    show("校準完成，計算中…", wait_key="space") if CALIB_METHOD == "lnrm" else None
    res = cal.finish()
    res.to_json(os.path.join(out_dir, f"{subj}_calib_{dim}.json"))
    print(f"[calib {dim}] H={res.high:.3f} L={res.low:.3f} in_range={res.in_range} {res.warnings}")
    if not res.in_range:
        show(f"警告：{dim} 的 H/L 落在範圍外\n{res.warnings}\n\n按空白鍵繼續（或 Esc 結束）")
    return res


# ── 區塊 1、2 ──
r1 = calibrate("dim1")
r2 = calibrate("dim2")

# ── 區塊 3：DFP ──
rows = dfp_trial_list(subj, r1.high, r1.low, r2.high, r2.low, N_PER_CELL, blocks=DFP_BLOCKS, seed=hash(subj) % 2**32)
show(f"主實驗：{len(rows)} 試\n\n按空白鍵開始")
for i, row in enumerate(rows):
    draw_stimulus(row["c1_level"], row["c2_level"])
    key, rt = get_key()
    win.flip()
    # TODO：DFP 的「答對」由你的作業定義（例如 target present = j）；這裡佔位：按 high 算對
    row["correct"] = int(key == KEYS["high"]) if key else 0
    row["rt"] = rt if rt is not None else ""
    core.wait(0.3)
    if (i + 1) % (len(rows) // DFP_BLOCKS) == 0 and i + 1 < len(rows):
        show("休息一下，按空白鍵繼續")
path = os.path.join(out_dir, f"{subj}_dfp.csv")
write_rows(rows, path)
show(f"結束，謝謝。\n\n資料：{path}\n分析：python scripts/analyze_participant.py {path}")
win.close()
core.quit()
