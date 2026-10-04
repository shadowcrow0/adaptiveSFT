"""
ex5：從後驗反解 H / L（純 numpy，不用 PyMC）。

目標 h_targ / l_targ 是兩個累積器的「漂移差」z₂ − z₁ = 2·d。要找 x 使

    2·(α·x + α₂·x²) = targ

R 的寫法（adaptiveSFT_functions.R:229-232，取較小的根）：

    x = ( −α/α₂ − √( (α/α₂)² + 2·targ/α₂ ) ) / 2

逐個後驗 draw 算一次，再取平均（α₂ ≥ 0 的 draw 會給負根或 NaN；這題照 R 全部平均，NaN 用 nanmean 跳過）。

要寫的：invert(alpha, alpha2, targ) —— alpha、alpha2 是等長的 numpy 向量（後驗 draw），回傳每個 draw 的 x。
check.py 用 tests/data/lnrm_stan_oracle.json 裡 Stan 的後驗平均（當成一個 draw）與 json 裡 R 算好的
salience.high / low 比；再用 ex4 的 trace（若已跑過）算 nanmean。
"""
import numpy as np


def invert(alpha, alpha2, targ):
    raise NotImplementedError("在這裡寫反解")
