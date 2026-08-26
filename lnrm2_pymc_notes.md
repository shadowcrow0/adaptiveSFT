# 把 `lnrm2.stan` 移植到 PyMC 的調查筆記

這份筆記記錄的是「如果要用 Python + PyMC 重做 `lnrm2.stan`，會遇到什麼」。
所有結論都在本機環境實測過：**PyMC 5.28.5 / PyTensor 2.38.3 / Python 3.11**。

`lnrm2.stan` 本身寫了什麼，見 `lnrm2_stan_explained.md`。

---

## 1. 結論

Python 生態圈**目前沒有現成的、解析形式的 Log-Normal Race Model 套件**。
要移植就得自己把 likelihood 寫出來。

---

## 2. Stan 與 PyMC 的對應

| Stan | PyMC v5 |
|---|---|
| `target += ...` | `pm.Potential("name", expr)` |
| `lognormal_lpdf(y \| m, s)` | `pm.logp(pm.LogNormal.dist(mu=m, sigma=s), y)` |
| `lognormal_lccdf(y \| m, s)` | 沒有安全的直接對應，見第 3 節 |
| `inv_gamma(1, .1)` | `pm.InverseGamma("varZ", alpha=1, beta=0.1)` |
| `real<lower=0,upper=minRT> psi` | `pm.Uniform("psi", lower=0, upper=minRT)` |
| `for` 迴圈加 `if/else` | `pt.switch(cond, a, b)` |

兩個版本差異：`pm.DensityDist` 是 PyMC v3 的舊 API，v5 改用 `pm.CustomDist`。
`rv.logp(x)` 這種呼叫方式在 v5 已移除，要用 `pm.logp(rv, x)`。

---

## 3. 最大的坑：`lccdf` 沒有安全的直接對應

PyMC 沒有 `logsf` 或 `logccdf` 這種 API。多數移植指南給的答案是

```python
lccdf = pm.math.log1mexp(pm.logcdf(rv, value))
```

**這個寫法會壞掉。**

先講符號慣例，因為 PyTensor 與 PyMC 兩邊的 docstring 互相矛盾。
實測結果：`pt.log1mexp(x)` 與 `pm.math.log1mexp(x)` 算的都是 `log(1 - exp(x))`，
要求 `x <= 0`，正數輸入回傳 `NaN`。因為 `logcdf` 本來就 `<= 0`，
所以**不需要手動加負號**。

符號對了，問題還在。實測拿 `sigma` 由大到小去試：

```
sigma      真值 lccdf          log1mexp 路線
0.6        -11.046460         -11.046459887      對
0.2        -81.307775         -81.307778005      對
0.05     -1250.565570               -inf         錯
0.033    -2865.061096               -inf         錯
0.01    -31149.836613               -inf         錯
```

原因：`sigma` 偏小時，輸家的 CDF 在 float64 下**飽和到恰好 1.0**，
`logcdf` 回傳 `-0.0`，於是 `log1mexp(-0.0) = log(0) = -inf`。
但真值是 -1250、-2865 這種**有限**的大負數。
Stan 的 `lognormal_lccdf` 是直接算尾端機率，沒有這個問題。

後果是致命的。NUTS 的 `jitter+adapt_diag` 初始化會隨機探到小 sigma 的區域，
模型會在還沒開始採樣前就拋出：

```
SamplingError: Initial evaluation of model at starting point failed!
{'varZ': -1.92, 'psi': -1.56, 'race': -inf}
```

**修法**：繞過 `logcdf`，改用常態存活函數。因為

```
S(x) = P(X > x) = P(Z > (log x - mu) / sigma) = Phi( -(log x - mu) / sigma )
```

寫成：

```python
def lognormal_lccdf(x, mu, sigma):
    return pm.logcdf(pm.Normal.dist(0.0, 1.0), -(pt.log(x) - mu) / sigma)
```

PyTensor 的常態 `logcdf` 內部用 `erfcx` 處理尾端，數值穩定。
實測：上表所有 sigma（含 sigma=0.01，lccdf = -31149）都與
`scipy.stats.lognorm.logsf` 完全吻合。

---

## 4. 次要的坑：`CustomDist` 的 `observed` 不能依賴參數

如果用 `pm.CustomDist` 而不是 `pm.Potential`，這樣寫會直接報錯（實測）：

```python
pm.CustomDist("rt_obs", ..., observed=rt - psi)
# TypeError: Variables that depend on other nodes cannot be used for
#            observed data. The data variable was: Sub.0
```

`observed=` 只能是常數資料。`rt - psi` 的位移必須搬進 `logp` 函式內部，
把 `psi` 當成 `logp` 的一個參數。`pm.Potential` 沒有這個限制。

（代價是 `Potential` 不是隨機變數，做不了 posterior predictive check。）

---

## 5. 可跑的移植碼

```python
import numpy as np, pytensor
pytensor.config.floatX = "float64"      # 尾端計算需要，預設 float32 精度不足
import pytensor.tensor as pt, pymc as pm

def lognormal_lccdf(x, mu, sigma):
    return pm.logcdf(pm.Normal.dist(0.0, 1.0), -(pt.log(x) - mu) / sigma)

with pm.Model() as model:
    mu     = pm.Normal("mu", 0, 1)
    alpha  = pm.Normal("alpha", 0, 2)
    alpha2 = pm.Normal("alpha2", 0, 1)
    varZ   = pm.InverseGamma("varZ", alpha=1, beta=0.1)
    psi    = pm.Uniform("psi", lower=0, upper=minRT)

    d      = alpha * intensity + alpha2 * intensity**2
    z1, z2 = mu - d, mu + d

    ok     = pt.constant(correct == 1)
    zw, zl = pt.switch(ok, z1, z2), pt.switch(ok, z2, z1)

    u = pt.clip(pt.constant(rt) - psi, 1e-12, np.inf)

    lp    = pm.logp(pm.LogNormal.dist(mu=zw, sigma=varZ), u)
    lccdf = lognormal_lccdf(u, zl, varZ)
    pm.Potential("race", pt.sum(lp + lccdf))

    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                      initvals={"psi": 0.3 * minRT})
```

注意 `varZ` 要用 `sigma=` 傳入，不要開根號。理由見
`lnrm2_stan_explained.md` 第 8 節：Stan 那邊它就是放在標準差的位置。

**驗證結果。** 逐試次比對：三組不同參數下，這個 PyMC 圖算出的每筆試次
log-likelihood 與 scipy 參考實作的最大絕對誤差為 4e-8 到 2e-6，
相對誤差穩定在 3e-8 左右，純粹是浮點捨入。

用 N=500 的模擬資料（真值 `mu=1.5, alpha=0.8, alpha2=-0.15, sigma=0.6, psi=0.12`）
跑 4 chains x 1000 draws，15 秒完成：

```
         mean     sd  hdi_3%  hdi_97%  ess_bulk  r_hat
mu      1.387  0.039   1.314    1.462    2312.0   1.00
alpha   0.683  0.068   0.560    0.812    1599.0   1.00
alpha2 -0.115  0.025  -0.163   -0.068    1727.0   1.01
varZ    0.591  0.033   0.533    0.650    1455.0   1.00
psi     0.156  0.073   0.003    0.266    1354.0   1.00
divergences: 0
```

`varZ` 與 `alpha2` 回收得很好。`psi` 偏高、`mu` 偏低，兩者互相補償——
這是 `mu` 與 `psi` 在後驗上高度相關造成的，原始 Stan 模型會有相同行為，
不是移植錯誤。

---

## 6. 現有套件（2026-08 查證）

| 套件 | 狀態 | 能不能直接用 |
|---|---|---|
| **HSSM** | 活躍，v0.4.0（2026-07-15），需 Python 3.12 以上。建立在 PyMC 5 上 | 不行。有 LBA、racing-diffusion、Poisson race，但**沒有 lognormal race**。而且那些模型用的是 LAN（神經網路近似 likelihood），不是解析形式 |
| **HDDM** | 停擺。最後版本 1.0.1（2023-12-03），依賴已淘汰的 PyMC 2 | 不建議新專案使用 |
| **ssm-simulators** | 活躍，v0.13.2（2026-07-16） | 是 HSSM 的模擬器後端，用來訓練 LAN，不提供解析 likelihood |
| `pm.LogNormal` / `pm.Wald` | PyMC 內建 | 只是單一分布，race 的邏輯仍要自己組 |

版本與日期是查 PyPI 確認的。HSSM 內部沒有 LNR 設定檔、LBA 走 LAN 近似這部分，
來自檢視其原始碼樹，未在本機安裝驗證（HSSM 需要 Python 3.12 以上，
與本次驗證環境不符）。

---

## 7. 版本與效能

**PyMC 6 需要 Python 3.12 以上。** 這點容易誤判：在 Python 3.11 下
`pip install pymc` 只會裝到 5.28.5，看起來像最新版，
但 PyPI 上已有 6.3.1（2026-08-16）。`pip index versions pymc` 也只列出 5.x。

PyMC 6 把 nutpie（Rust 實作的 NUTS）設為預設 sampler。
在 PyMC 5 下可以自己指定：

```python
pm.sample(nuts_sampler="nutpie")     # 或 "numpyro"，需額外安裝
```

其他：

- **全程向量化**。用 `pt.switch` 取代 Python 迴圈，逐筆建圖會產生巨大的計算圖。
- **明確給 `initvals`**。`psi` 的邊界問題在 PyMC 一樣存在，
  `initvals={"psi": 0.3*minRT}` 比依賴預設 jitter 可靠。
- **設 `floatX="float64"`**。預設 float32 在尾端計算精度不夠。
