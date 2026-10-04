"""
ex2：位移對數常態的 RT，自己寫 likelihood。

    rt_i = ψ + T_i,   ln T_i ~ Normal(μ, σ)
    ψ ~ Uniform(0, min(rt))        ← 和 lnrm2.stan:17 一樣：psi 不能超過最小 RT，不然 ln(rt − ψ) 會 NaN
    μ ~ Normal(0, 1)
    σ ~ HalfNormal(1)

對數常態的 log 密度（自己寫，不用 pm.LogNormal）：

    ln f(t; μ, σ) = −ln t − ln σ − ½·ln(2π) − (ln t − μ)² / (2σ²)

要寫的：build_model(rt) 回傳 pm.Model，變數名 "psi"、"mu"、"sigma"。
likelihood 用 pm.Potential("obs", pt.sum(logp)) 加進去，logp 是長度 N 的 pytensor 向量。
check.py 用 ψ = 0.2、μ = −0.5、σ = 0.4 模擬 400 筆，檢查後驗平均。

提示：import pytensor.tensor as pt；pt.log、pt.sum；½·ln(2π) = 0.9189385332046727。
"""
import pymc as pm
import pytensor.tensor as pt


def build_model(rt):
    raise NotImplementedError("在這裡寫模型")
