"""
ex2：位移對數常態的 RT，自己寫 likelihood。

    rt_i = ψ + T_i,   ln T_i ~ Normal(μ, σ)
    ψ ~ Uniform(0, min(rt))        ← 和 lnrm2.stan:17 一樣：psi 不能超過最小 RT，不然 ln(rt − ψ) 會 NaN
    μ ~ Normal(0, 1)
    σ ~ HalfNormal(1)

PyMC 有 LogNormal，但沒有「位移」版，所以 log 密度要自己組 —— 不過不用手寫公式，
直接跟 PyMC 要內建分布的 log 密度：

    ln f(t; μ, σ)  =  pm.logp(pm.LogNormal.dist(mu=μ, sigma=σ), t)        ← t = rt − ψ
                   （等於手寫的 −ln t − ln σ − ½·ln(2π) − (ln t − μ)² / (2σ²)）

pm.LogNormal.dist(...) 是「沒掛進模型的分布物件」，pm.logp(分布, 值) 回傳它在那個值的 log 密度，
是 pytensor 向量，可以直接參與運算。adaptivesft/models_v2.py 的正式版就是這樣寫的。

要寫的：build_model(rt) 回傳 pm.Model，變數名 "psi"、"mu"、"sigma"。
likelihood 用 pm.Potential("obs", pt.sum(logp)) 加進去，logp 是長度 N 的 pytensor 向量。
check.py 用 ψ = 0.2、μ = −0.5、σ = 0.4 模擬 400 筆，檢查後驗平均。

提示：rt 是 numpy 陣列，rt − psi 會自動變成 pytensor 向量；pt.sum 把 N 個 logp 加起來。
      想對照手寫版：½·ln(2π) = 0.9189385332046727，兩種寫法 .eval() 出來應該一樣到 1e−12。
"""
import pymc as pm
import pytensor.tensor as pt


def build_model(rt):
    raise NotImplementedError("在這裡寫模型")
