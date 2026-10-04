"""
ex1：最小的 PyMC 模型。

    y_i ~ Normal(μ, σ)        i = 1 … N
    μ ~ Normal(0, 10)
    σ ~ HalfNormal(5)

要寫的：build_model(y) 回傳一個 pm.Model，裡面有兩個隨機變數，名字必須叫 "mu" 和 "sigma"。
check.py 會用 μ = 2、σ = 1.5 模擬 200 筆資料，抽樣，檢查後驗平均在真值附近。

提示：
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 10)
        ...
        pm.Normal("y", mu, sigma, observed=y)
    return model
"""
import pymc as pm


def build_model(y):
    raise NotImplementedError("在這裡寫模型")
