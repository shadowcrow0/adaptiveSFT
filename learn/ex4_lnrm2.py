"""
ex4：完整的 lnrm2 —— 把 ex3 接上強度。

    d_i  = α·x_i + α₂·x_i²                 （lnrm2.stan:23-24）
    z₁,i = μ − d_i      「答對」累積器
    z₂,i = μ + d_i      「答錯」累積器        → x 越大，答對的越快、答錯的越慢
    先驗（lnrm2.stan:30-33、:17）：
        varZ ~ InverseGamma(1, 0.1)          ← 這是 σ（SD），名字沿用 Stan 的 varZ
        μ    ~ Normal(0, 1)
        α    ~ Normal(0, 2)
        α₂   ~ Normal(0, 1)
        ψ    ~ Uniform(0, min(rt))

要寫的：build_model(rt, correct, x) 回傳 pm.Model，變數名 "mu" "alpha" "alpha2" "varZ" "psi"。
likelihood 用你 ex3 的 race_logp（from learn.ex3_race import race_logp）。

check.py 會：
    1. 讀 tests/data/lnrm_oracle_input.csv（真值 μ 1.5、α 0.8、α₂ −0.15、varZ 0.6、ψ 0.12）
    2. pm.sample(draws=1000, tune=1000, chains=4)，約 1–2 分鐘
    3. 對 tests/data/lnrm_stan_oracle.json（Arc 上 cmdstanr 跑的 Stan 後驗）：
       五個參數 |你的平均 − Stan 的平均| / Stan 的 SD < 0.35，Rhat < 1.05
這和 tests/test_lnrm_vs_stan.py 對 adaptivesft 的要求一模一樣。

提示：psi 的上界要用資料算（float(rt.min())）；NUTS 起點不好會發散，可以給 pm.sample(initvals={...})。
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from learn.ex3_race import race_logp


def build_model(rt, correct, x):
    raise NotImplementedError("在這裡寫模型")
