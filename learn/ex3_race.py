"""
ex3：兩個累積器的賽跑 —— lnrm2.stan 的 model 區塊（:35-46）。

每一題有兩個累積器，各自的完成時間是位移對數常態：
    「答對」累積器   ln T₁ ~ Normal(z₁, σ)
    「答錯」累積器   ln T₂ ~ Normal(z₂, σ)
    rt = ψ + min(T₁, T₂)，correct = 1 表示 T₁ 先到

一題的 log likelihood = 贏的那個的 log 密度 + 輸的那個「還沒到」的 log 機率（log survival）：

    correct = 1：  ln f(t; z₁, σ) + ln S(t; z₂, σ)
    correct = 0：  ln f(t; z₂, σ) + ln S(t; z₁, σ)
    其中 t = rt − ψ，ln S(t; z, σ) = ln(1 − F(t; z, σ))，F 是對數常態的累積分布

兩塊都用 PyMC 內建的 LogNormal 拼，不手寫公式（和 adaptivesft/models_v2.py 一樣）：

    ln f(t; z, σ)  =  pm.logp(  pm.LogNormal.dist(mu=z, sigma=σ), t)
    ln F(t; z, σ)  =  pm.logcdf(pm.LogNormal.dist(mu=z, sigma=σ), t)
    ln S           =  pm.math.log1mexp(ln F)          ← log1mexp(a) = ln(1 − eᵃ)，在 a ≈ 0 時仍穩定

Stan 用 if 分兩種；pytensor 不要寫 if，用 correct 當 0/1 權重：
    z_win  = correct·z₁ + (1 − correct)·z₂
    z_lose = correct·z₂ + (1 − correct)·z₁

要寫的：race_logp(rt, correct, z1, z2, sigma, psi) 回傳長度 N 的 pytensor 向量（每題一個 logp）。
輸入可以是 numpy 陣列或 pytensor 變數；用 pt.as_tensor_variable 包一下就都能算。
check.py 會在幾組固定的參數上，把你的向量 .eval() 出來，對 adaptivesft.race.lnrm_pointwise_loglik 的值
（numba、手寫公式的獨立實作；兩條浮點路徑在 logp ≈ −50 時差約 1e−6，檢查用的是相對容忍度）。

提示：不要寫 pt.log(1 − pt.exp(logcdf))，很慢的 RT 會變成 ln(0)；log1mexp 就是為了這個存在的。
      t ≤ 0（ψ 比 rt 大）時密度是 0，用 pt.switch(t > 0, ..., -np.inf) 擋掉。
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def race_logp(rt, correct, z1, z2, sigma, psi):
    raise NotImplementedError("在這裡寫 likelihood")
