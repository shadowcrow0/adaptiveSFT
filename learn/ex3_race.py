"""
ex3：兩個累積器的賽跑 —— lnrm2.stan 的 model 區塊（:35-46）。

每一題有兩個累積器，各自的完成時間是位移對數常態：
    「答對」累積器   ln T₁ ~ Normal(z₁, σ)
    「答錯」累積器   ln T₂ ~ Normal(z₂, σ)
    rt = ψ + min(T₁, T₂)，correct = 1 表示 T₁ 先到

一題的 log likelihood = 贏的那個的 log 密度 + 輸的那個「還沒到」的 log 機率（log survival）：

    correct = 1：  ln f(t; z₁, σ) + ln S(t; z₂, σ)
    correct = 0：  ln f(t; z₂, σ) + ln S(t; z₁, σ)
    其中 t = rt − ψ，ln S(t; z, σ) = ln(1 − Φ((ln t − z)/σ)) = ln(½·erfc((ln t − z)/(σ·√2)))

Stan 用 if 分兩種；pytensor 不要寫 if，用 correct 當 0/1 權重：
    z_win  = correct·z₁ + (1 − correct)·z₂
    z_lose = correct·z₂ + (1 − correct)·z₁

要寫的：race_logp(rt, correct, z1, z2, sigma, psi) 回傳長度 N 的 pytensor 向量（每題一個 logp）。
輸入可以是 numpy 陣列或 pytensor 變數；用 pt.as_tensor_variable 包一下就都能算。
check.py 會在幾組固定的參數上，把你的向量 .eval() 出來，對 adaptivesft.race.lnrm_pointwise_loglik 的值（numba，獨立實作）。

提示：pt.erfc 存在；不要用 pt.log(1 − Φ) 的寫法，尾端會變成 ln(0)。
"""
import pytensor.tensor as pt


def race_logp(rt, correct, z1, z2, sigma, psi):
    raise NotImplementedError("在這裡寫 likelihood")
