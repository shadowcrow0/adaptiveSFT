# learn/ —— 自己寫出 lnrm2 的五個練習

目標：不靠 AI，把 `lnrm2.stan` 用 PyMC 寫出來，擬合同一份資料，後驗對上 Stan。
規則：**每個練習先自己寫，跑 `python learn/check.py exN` 對答案；沒過之前不要開 `adaptivesft/models.py`。**
答案在 `adaptivesft/` 裡，但它們是「對過的參考」，不是教材；你的版本可以長得不一樣，數字對就好。

```
   ex1  y ~ Normal(μ, σ)                 PyMC 的骨架：Model / 先驗 / observed / sample / summary
    │
   ex2  rt ~ ψ + LogNormal(μ, σ)         自己寫 likelihood（pm.Potential）、有上界的位移參數 ψ
    │
   ex3  兩個累積器的賽跑                   lnrm2.stan 的 model 區塊：lpdf(贏的) + lccdf(輸的)，用 pytensor 寫
    │
   ex4  d = α·x + α₂·x²，接上 ex3          完整的 lnrm2；擬合 tests/data/lnrm_oracle_input.csv，對 Stan 的後驗
    │
   ex5  反解 H / L（純 numpy）             adaptiveSFT_functions.R:229-232；對 Stan oracle 裡的數字
```

每個練習一個檔，裡面有：數學（code block）、要你填的函式（`raise NotImplementedError`）、提示。
`check.py` 只呼叫你的函式、比數字，不看你怎麼寫。

## 進度建議

| 週 | 做什麼 | 過關的樣子 |
|---|---|---|
| 1 | ex1、ex2。讀 PyMC 官方 "Getting started" 與 "Using a custom likelihood"（pm.Potential） | `check.py ex1 ex2` 綠 |
| 2 | ex3。先在紙上把 lnrm2.stan 的 for 迴圈翻成「贏的累積器 / 輸的累積器」兩個向量 | `check.py ex3` 綠（logp 對到 1e−6） |
| 3 | ex4。NUTS 跑 2000 draws 要 1–2 分鐘；對 Stan 後驗，五個參數的平均差 < 0.35 個 SD | `check.py ex4` 綠 |
| 4 | ex5，然後把 ex4 的模型改成 ogival（d = ½·L·inv_logit(slope·(x − mid))），不給檢查器，自己想怎麼驗 | 能跟老闆解釋為什麼 L 固定 10 會讓 2.0/0.5 的目標掉到範圍外 |

## Stan ↔ PyMC 對照（ex3、ex4 會一直用到）

```
   lnrm2.stan                                   PyMC
   ─────────────────────────────────────────    ──────────────────────────────────────────────────
   data { real rt[N]; ... }                     numpy array，直接當常數用
   parameters { real alpha; }                   alpha = pm.Normal("alpha", 0, 2)     ← 先驗寫在這裡
   real<lower=0> varZ;  varZ ~ inv_gamma(1,.1)  varZ = pm.InverseGamma("varZ", alpha=1, beta=0.1)
   real<lower=0,upper=minRT> psi;（無先驗）      psi = pm.Uniform("psi", 0, min_rt)
   transformed parameters { z[1,tr] = mu - d }  z1 = mu - d          ← pytensor 向量運算，不用 for
   target += lognormal_lpdf(t | z, s)           手寫：−ln t − ln s − ½ln(2π) − (ln t − z)²/(2s²)
   target += lognormal_lccdf(t | z, s)          手寫：ln(½·erfc((ln t − z)/(s·√2)))   ← ln(1 − Φ)
   target += ...（整個 model 區塊）              pm.Potential("obs", pt.sum(logp))
   if (correct[tr]) {...} else {...}            用 correct 當 0/1 權重選 z_win / z_lose，不用 if
```

## 環境

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
python learn/check.py ex1          # 一個
python learn/check.py all          # 全部（ex4 要幾分鐘）
```
