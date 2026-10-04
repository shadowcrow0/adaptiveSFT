# learn/ —— 自己寫出 lnrm2 的五個練習

**這是什麼。** 五個由淺到深的練習，讓你不靠 AI，把原作者的 Stan 模型 `lnrm2.stan` 用 PyMC
自己寫出來，擬合同一份資料，後驗對上 Stan。

**一個比喻：學做菜。** 食譜（`adaptivesft/` 的正式程式）已經有了，但照抄不會學會。
每個練習只給你食材和目標味道，你自己下廚。`check.py` 是試吃員：只嘗味道（比數字），
不看刀工（不管你怎麼寫）。

幾個名詞：

- **Stan**：寫貝氏模型的語言，原作者用的。**PyMC**：Python 裡做同一件事的工具。
- **後驗（posterior）**：模型看完資料後對參數的看法，一整組可能值。
- **先驗（prior）**：看資料之前的預設看法。
- **likelihood**：在某組參數下，看到這筆資料的機率。模型的核心就是把它寫對。
- **logp**：likelihood 取對數。PyMC 內部都用這個。
- **NUTS**：PyMC 的取樣器，自動走遍可能的參數值，產生後驗。
- **draw**：後驗裡的一筆樣本。「2000 draws」＝抽了 2000 組參數。
- **累積器（accumulator）**：賽跑模型裡的計時器，一條答對、一條答錯，先到的贏。
- **位移對數常態**：反應時間 = 固定延遲 ψ + 一個對數常態分布的隨機量。
- **反解**：已知參數，倒過來算出哪個刺激強度會給目標難度；H 是高顯著度、L 是低顯著度。

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

每一階在練什麼：

- **ex1**：最簡單的模型，學 PyMC 的五個固定動作。
- **ex2**：PyMC 沒有現成的「位移對數常態」，所以學自己寫 likelihood 塞進去（`pm.Potential`）。
  ψ 不能超過最小反應時間，學怎麼給參數上界。
- **ex3**：賽跑的核心。贏的那條用密度（lpdf），輸的那條用「還沒到」的機率（lccdf，
  log 補累積分布）。`pytensor` 是 PyMC 的計算引擎，要用它的運算而不是普通 Python。
- **ex4**：把刺激強度 x 接進來。難度 d 是 x 的二次式。擬合同一份資料，和 Stan 的後驗比。
- **ex5**：不用 PyMC，純算術。已知 α、α₂，解一個二次方程式得到強度。

每個練習一個檔，裡面有：數學（code block）、要你填的函式（`raise NotImplementedError`）、提示。
`check.py` 只呼叫你的函式、比數字，不看你怎麼寫。

「`raise NotImplementedError`」＝函式裡故意留空，跑到就報錯，提醒你這裡要自己寫。
「oracle」＝用 Stan 原版跑出來存檔的標準答案（`tests/data/lnrm_stan_oracle.json`）。

## 進度建議

| 週 | 做什麼 | 過關的樣子 |
|---|---|---|
| 1 | ex1、ex2。讀 PyMC 官方 "Getting started" 與 "Using a custom likelihood"（pm.Potential） | `check.py ex1 ex2` 綠 |
| 2 | ex3。先在紙上把 lnrm2.stan 的 for 迴圈翻成「贏的累積器 / 輸的累積器」兩個向量 | `check.py ex3` 綠（logp 對到 1e−6） |
| 3 | ex4。NUTS 跑 2000 draws 要 1–2 分鐘；對 Stan 後驗，五個參數的平均差 < 0.35 個 SD | `check.py ex4` 綠 |
| 4 | ex5，然後把 ex4 的模型改成 ogival（d = ½·L·inv_logit(slope·(x − mid))），不給檢查器，自己想怎麼驗 | 能跟老闆解釋為什麼 L 固定 10 會讓 2.0/0.5 的目標掉到範圍外 |

「綠」＝ `check.py` 印出 PASS。「logp 對到 1e−6」＝你算的和參考答案差不到百萬分之一。
「平均差 < 0.35 個 SD」＝五個參數的後驗平均和 Stan 的差距，不到後驗標準差的三分之一。
第 4 週的 ogival 是 S 形曲線版的難度函數，L 是曲線最高能給的分離；因為 L 固定 10，
目標分離設 2.0 / 0.5 這麼小時，反解出的強度會跑到實驗能呈現的範圍外。

## Stan ↔ PyMC 對照（ex3、ex4 會一直用到）

左邊是 Stan 檔裡的寫法，右邊是 PyMC 裡做同一件事的寫法。

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

兩個容易卡住的地方：Stan 的 `parameters` 區塊只宣告參數，先驗寫在 `model` 區塊；PyMC 把
兩件事合成一行。Stan 用 `if` 逐題分「答對 / 答錯」；PyMC 要整個向量一起算，所以用 `correct`
（0 或 1）當權重去挑哪條累積器是贏家。`Φ` 是標準常態的累積分布函數；`erfc` 是算它尾端
機率的穩定寫法，直接寫 `ln(1 − Φ)` 在尾端會變成 ln(0)。

## 環境

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
python learn/check.py ex1          # 一個
python learn/check.py all          # 全部（ex4 要幾分鐘）
```

第一行裝釘住版本的工具（`requirements-lock.txt` 把每個套件的版本號寫死，三年後重裝也一樣），
再把本 repo 裝進去但不另外抓依賴（`--no-deps`）。

## 所以你要做什麼

1. 跑上面「環境」的第一行。
2. 打開 `learn/ex1_normal.py`，照檔頭的數學和提示寫 `build_model`，跑 `python learn/check.py ex1`。
3. 綠了才開下一題。卡住先讀該檔的提示和上面的對照表，不要開 `adaptivesft/models.py`。
4. 四週走完，你應該能自己解釋 lnrm2 的每一行，以及為什麼 L 固定 10 會讓小目標掉到範圍外。
