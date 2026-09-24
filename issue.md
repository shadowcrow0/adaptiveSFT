# issue.md — adaptiveSFT 目前遇到的所有問題、能不能解、怎麼解

> 直接回答：**跟 `lnrm2.stan` 有關的問題，PyMC + PyTensor 全部能解，而且已經解完**
> （`model_lnrm2.py`）。**跟 R 有關的問題，PyMC 幫不上，但都是一行到五行的修法。**
> **真正無解的只有一種：檔案不在 repo 裡**（`lnrm2a.stan`、`lnrm1.stan`、`lnrm0.stan`、
> `post95.Rdata`、輸入 csv）。這不是版本問題，是東西沒了。
>
> R 端逐項證據（每條都在 R 4.3.3 / stanc 2.32.2 + 2.39.0 上重現過）見
> `r_version_inventory.md`（subagent 寫，英文，行號我抽驗過）。

```
   問題來源                 PyMC/PyTensor 能解？    解法
   ─────────────────────────────────────────────────────────────────────────
   R 4.2 改了 if() 規則      ✘ 跟 PyMC 無關          R 改一行 (R1)
   Stan 2.33 改了陣列語法    ✔ 走 PyMC 就繞過        或 Stan 改五行 (S1)
   PyMC 內建 RV 拼不出 race  ✔ 已解                  pt.Op + numba (M1–M7)
   pystan 2 API 死了         ✔ 換 PyMC               adaptive_sft2.py 重寫 (P3)
   .stan / .Rdata / csv 遺失 ✘ 誰都不能              問原作者 (S2, R5)
   2018 程式本身的 bug       ✘ 跟版本無關            逐個修 (R6, P1, P2)
```

---

## 1. R 端（版本更改造成的）

### R1 `if()` 條件長度 > 1 — R 4.2.0 起是錯誤 — **可解，一行**

**程式碼字面** `adaptiveSFT_functions.R:228`：

```r
  if (post.diff$alpha2 <0) {
```

`post.diff` 是 `extract(fitModel, ...)`（`:219`）的結果，`$alpha2` 是 4000 筆後驗抽樣，
不是一個數。

**誰改了什麼**：R < 4.2 只警告 `only the first element will be used`，拿第一筆決定；
R ≥ 4.2.0 直接報錯。R 4.3.3 重現：

```
Error in if (post.diff$alpha2 < 0) 1 : the condition has length > 1
```

**舊 R 實際行為**（這是推導，不是字面）：

```
   alpha2[1] < 0   ──►  全部 4000 筆都代進公式，mean(…, na.rm=TRUE)   (:279)
   alpha2[1] ≥ 0   ──►  h_targ.dist 沒被賦值 ──► :279 object not found
```

**解法**（不改 repo，由 owner 決定後再套）：

```r
  l_targ.dist <- with(post.diff, ifelse(alpha2 < 0,
        (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * l_targ)) / 2, NaN))
  h_targ.dist <- with(post.diff, ifelse(alpha2 < 0,
        (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * h_targ)) / 2, NaN))
```

**數值會不會變**：`alpha2 < 0` 的 draw 結果完全相同；`alpha2 ≥ 0` 的 draw 舊碼算出
`sqrt(負數)` = NaN 被 `na.rm` 丟掉，新碼直接給 NaN，也一樣。唯一差別：`alpha2 > 0`
但判別式恰好 ≥ 0 的稀有 draw，舊碼保留、新碼丟掉。

另外兩種寫法 `all(alpha2 < 0)`、`mean(alpha2) < 0` 數值都不同——**owner 要選一個**。

同類潛伏案例：`psiSimulation_functions.R:45` `if (is.na(prior)) {`。現在 `prior <- NA`
是純量所以沒事，一旦傳真正的 prior 陣列進去就炸。改 `if (all(is.na(prior)))`。

### R2 rstan 的 Stan 語言版本由 StanHeaders 決定 — **推翻之前的說法**

之前的文件（`plan_r_modernization.md`、`lnrm2_pymc_gaps.md`）說「CRAN rstan 仍綁
Stan 2.32，`real x[N]` 還能編」。**錯。** rstan 2.32.7 只要求 `StanHeaders >= 2.32.0`，
parser 是從 `StanHeaders/inst/stanc.js` 載入的（`rstan/R/zzz.R:29-30`）。CRAN 現在的
StanHeaders 是 2.39.1，帶 stanc 2.39.0，**會拒絕 `lnrm2.stan`**：

```
Syntax error in 'string', line 3, column 18 to column 19, parsing error:
     3:     real intensity[N];
                           ^
Ill-formed declaration. ";" expected after variable declaration.
  It looks like you are trying to use the old array syntax.
```

所以「裝新 R + 新 rstan 就能跑」是假的，S1 的五行一定要改。

### R3 rstan 呼叫參數 — **沒壞**

`stan(file=, data=, pars=, open_progress=)`、`extract(fit, pars)` 都還是合法簽名。
唯一小毛病：`simulateLNRM_ogival.R:158` 寫 `permute=TRUE`，正式參數名是 `permuted`，
靠 S4 partial match 撐著。改成 `permuted=TRUE`。

### R4 `stringsAsFactors`、`class(x)=="matrix"`、`sample()` — **沒中**

三個常見的 R 4.0 / 3.6 陷阱，五個 R 檔都沒踩到。不用改。

### R5 檔案不在 repo — **不可解**

| 誰引用 | 檔名 | 狀態 |
|---|---|---|
| `adaptiveSFT_functions.R:185`、`simulateLNRM_ogival.R:62` | `lnrm2a.stan`（ogival 模型，**每個腳本實際用的都是這個**） | 沒有 |
| `adaptiveSFT_functions.R:210` | `lnrm1.stan` | 沒有 |
| `simulateLNRM_ogival.R:157` | `lnrm0.stan` | 沒有 |
| `simulateLNRM_ogival.R:111` | `post95.Rdata` | 沒有 |
| `psi Simulation_26MAR2019.R:507,510` | `Psi_Simulation_SFTresults.csv`、`PsiDDM_Simulation_Pars.csv` | 沒有，且前者的欄位跟 2018 腳本寫出來的對不上 |

`git log --all` 顯示 repo 從第一個 commit 起就只有 `lnrm2.stan` 一個 Stan 檔。
**只有原作者能補。**

### R6 2018 程式本身的 bug（跟版本無關，但一樣擋路）

| 位置 | 字面 | 問題 | 修法 |
|---|---|---|---|
| `adaptiveSFT_functions.R:11` | `with(postSamps, L * inv_logit(...))` | `L` 不在 pars 裡，只在 `simulateLNRM_ogival.R:26` 定義成全域 | 加參數 `L = 10` |
| `adaptiveSFT_functions.R:98` | `sigmasq=sigmasqx` | 打錯字 | 改 `sigmasq` |
| `adaptiveSFT_functions.R:184` | `if (anyNA(fitModel))` 對 S4 stanfit | 會警告但結果正確 | 改 `if (!is(fitModel, "stanfit"))` |
| `psi Simulation_26MAR2019.R:535` 等 8 處 | `psi_color_ddm(result.color, nDFP, allpars[sn,])` 三個引數 | 定義 `psiSimulation_functions.R:173` 只收兩個 | 對齊簽名 |
| `psi Simulation_25JUNE2018.R:62` | `x.axis` | 定義的是 `axis.x` | 改名 |
| `psi Simulation_25JUNE2018.R:2`、`26MAR2019.R:2` | `setwd("C:/Users/w018elf/...")` | Windows 硬路徑，Linux/mac 第一行就死 | 刪掉 |
| `psi Simulation_25JUNE2018.R:71` | `windows()` | Windows 專用繪圖裝置 | 改 `dev.new()` |
| `simulateLNRM_ogival.R:571` | `dp[sn] <- ...` | `dp` 沒初始化 | 前面加 `dp <- c()` |

完整清單（含 `rsamp`、`postOpt.diff`、`polynomial_order`、`N` 未定義等）見
`r_version_inventory.md` F8–F9。

**現況**：五個 R 檔在 R 4.3.3 都 `parse()` 得過，但沒有一個能跑到底。
`find_salience_polynomial`（唯一用 `lnrm2.stan` 的函式）**沒有任何腳本呼叫它**——
所有腳本走的都是 `find_salience_ogival` → 遺失的 `lnrm2a.stan`。

---
## 2. Stan 端 (`lnrm2.stan`)

### S1 `real x[N]` 舊陣列語法 — **可解**

**程式碼字面** `lnrm2.stan:3,4,6,9,20`：

```stan
   real intensity[N];                 // :3
   int<lower=0,upper=1> correct[N];   // :4
   real<lower=0> rt[N];               // :6
   real square_intensity[N];          // :9
   real z[2,N];                       // :20
```

**誰改了什麼**：Stan 2.33 (2023-09) 移除 `type name[N]` 語法，只接受
`array[N] type name`。stanc 2.32.2 編得過（5 個 deprecation 警告）；
stanc 2.39.0 直接 parse error。CRAN 新裝 rstan 會拿到 2.39.0（見 R2），
cmdstanr / cmdstan ≥ 2.33 同樣拒絕。

**解法**（5 行，語意完全不變；subagent 用 stanc 2.32.2 和 2.39.0 都驗過編譯通過）：

```stan
   array[N] real intensity;
   array[N] int<lower=0,upper=1> correct;
   array[N] real<lower=0> rt;
   array[N] real square_intensity;
   array[2,N] real z;
```

或者不改 Stan：走 PyMC（見 §4），`lnrm2.stan` 只留作對照文件。

### S2 `lnrm0.stan` / `lnrm1.stan` / `lnrm2a.stan` 不在 repo — **不可解（檔案遺失）**

`adaptiveSFT_functions.R:185` 呼叫 `lnrm2a.stan`，`:210` 呼叫 `lnrm1.stan`，
`adaptive_sft2.py:167` 呼叫 `lnrm2a.stan`。三個檔案都不存在。
`find_salience_ogival` 和 `polynomial_order=1` 這兩條路完全跑不了。
只有 `polynomial_order=2`（用 `lnrm2.stan`）能跑。

PyMC / PyTensor 幫不上：檔案內容不知道就不能重寫。
只能問原作者，或用 `getPr_ogival`（`adaptiveSFT_functions.R:9-18`）反推
lnrm2a 的 likelihood 自己重建——那是新模型不是移植。

### S3 `varZ` 命名陷阱 — **不是 bug，但要記得**

`lnrm2.stan:38` `lognormal_lpdf(rt[tr] - psi | z[1,tr], varZ)`：
`varZ` 在 **標準差** 的位置。R 端 `dlognormalrace`
（`adaptiveSFT_functions.R:61-62`）卻做 `sqrt(sigmasq)`。
移植時 PyMC 要傳 `sigma=varZ`，不能開根號。已在 `model_lnrm2.py` 照 Stan 做。

---

## 3. Python 端 (`adaptive_sft2.py`)

這個檔案是 2018 年的半成品，三個硬錯誤：

| # | 行 | 字面 | 問題 | 可解？ |
|---|---|---|---|---|
| P1 | `:12` | `posterior_samples['intensity']^2` | Python `^` 是 XOR 不是次方 | 可，改 `**2` |
| P2 | `:27` | `for i in range(allchannels)` | `allchannels` 沒定義 | 可，改成 `len(mu)` |
| P3 | `:2,:167,:172,:176` | `import pystan` / `pystan.StanModel` / `.sampling` / `.extract` | pystan 2 API；pystan 3 完全不同且已停止維護 | 可，換 cmdstanpy 或 PyMC |
| P4 | `:167` | `file="lnrm2a.stan"` | 同 S2，檔案不存在 | 否 |
| P5 | `:161,:176,:181-186` | `slope`, `midpoint` | 這是 lnrm2a (ogival) 的參數，不是 lnrm2 的 | 要換 lnrm2 就得改回 `alpha/alpha2` + 二次式反解（被註解在 `:158-160`） |

**結論**：`adaptive_sft2.py` 沒有一條能跑的路。PyMC 可以取代它的 Stan 部分
（`model_lnrm2.py`），但 P5 的反解和 `find_salience` 的流程要重寫。

---

## 4. PyMC / PyTensor 端 — 哪些解了、哪些沒解

詳細在 `lnrm2_pymc_gaps.md`，這裡只列結論。

```
                     純 PyMC 內建 RV 路           pt.Op (numba) 路
                     (lnrm2_pymc.py)              (model_lnrm2.py)
   ──────────────────────────────────────────────────────────────────
   lccdf 尾端 −inf      要繞 (Normal CDF)           解了 (erfc + 漸近級數)
   observed = rt−psi    不行 → Potential            解了 (減法在 Op 裡)
   逐題 if/else         pt.switch 向量化            解了 (numba 迴圈)
   float32 尾端         要設 floatX                 解了 (固定 float64)
   內建 Censored/Wald   錯或 −inf                   不需要
   梯度 / NUTS          有                          沒有 → DEMetropolisZ
   逐題 log_lik 精度    1e−6                        1e−14
```

| # | 問題 | 狀態 | 怎麼解 |
|---|---|---|---|
| M1 | `lognormal_lccdf` 在 PyMC 用 `log1mexp(logcdf)` → −inf | **已解** | `model_lnrm2.py` `log_norm_sf`：u ≤ 30 用 `erfc`，u > 30 用漸近級數；對 scipy 到 u=250 誤差 1e−16 |
| M2 | `pm.CustomDist(observed=rt - psi)` 被拒（observed 必須是常數） | **已解** | `rt` 當 data 進 Op，`psi` 當參數，減法在 `perform` 裡；`pm.CustomDist("rt_obs", ..., logp=Op, observed=rt)` 驗過 = scipy |
| M3 | `real<lower=0,upper=minRT> psi` 宣告即 prior | 不是問題 | 寫成 `pm.Uniform("psi", 0, min_rt)`，同一件事 |
| M4 | 逐題 `if (correct[tr])` | **已解** | numba 迴圈 |
| M5 | `transformed parameters` 沒有對應 block | 不是問題 | `pm.Deterministic`，要存才寫 |
| M6 | float32 尾端下溢 | **已解** | numba float64 |
| M7 | 用 `pm.LogNormal` + `pm.Censored` 拼 race | **放棄這條** | `Censored(LogNormal)` −inf、`Censored(Wald)` 尾端錯 11 個數量級；改走 Op |
| M8 | Op 沒梯度 → 不能 NUTS | **未解** | 給 Op 寫 `grad`（`∂(ln f + ln S)/∂θ` 解析式）。DEMetropolisZ 目前夠用：1000 題 8 chains × 3000，R-hat 1.01，ESS ≥ 858 |
| M9 | `mu` / `psi` 有 ridge，`mu` 回收略偏低 | 模型本身性質 | Stan 版也有，不是移植問題；加資料或給 `psi` 更緊的 prior |

**所以「Stan 那部分」PyMC + PyTensor 全部能解，而且已經做完了。**

---

## 5. 誰能被 PyMC / PyTensor 解、誰不能

```
   adaptiveSFT 流程                                誰來解
   ─────────────────────────────────────────────────────────────────────
   模擬資料 (moc_ddm / dfp_ddm / simdiffT)    R ── 跟 PyMC 無關，要嘛修 R 要嘛用 numpy 重寫
         |
         v
   dataframe2stan                             R ── 3 行，任何語言都行
         |
         v
   lnrm2.stan 擬合                            ✔ PyMC：model_lnrm2.py 已完成
         |
         v
   extract 後驗                               ✔ PyMC：az.extract(idata) 已完成
         |
         v
   if (alpha2 < 0) 二次式反解 → high/low       R ── R 4.2 壞掉 (R1)；PyMC 無關；
                                                     要嘛修 R 一行，要嘛 numpy 寫 4 行
         |
         v
   psi Simulation_*.R 用 high/low 跑 DFP      R ── 跟 PyMC 無關
```

**直接回答「能不能用 pymc / pytensor 解決」：**

- **能**：所有跟 `lnrm2.stan` 有關的（S1, S3, M1–M7）。已解，在 `model_lnrm2.py`。
- **不能，但跟 PyMC 無關、各自另有解**：R 4.2 的 `if` 長度問題（R1，一行）、
  pystan 2 API（P3，換工具）、`^` XOR（P1，一個字元）。
- **真的不能**：`lnrm1.stan` / `lnrm2a.stan` 遺失（S2, P4）。這不是版本問題，
  是檔案沒了，只有原作者能補。

---

## 6. 建議的最短路徑

| 路徑 | 要動什麼 | 工作量 | 結果 |
|---|---|---|---|
| **A 修 R**（`plan_A_implementation.md`） | R 一行 (`:228`) + Stan 5 行 (S1) + 決定 `if` 語意 | 半天–2 天 | 原流程整條在 R 跑，只有 order=2 |
| **B 全 Python** | `model_lnrm2.py` 已有；補 `find_salience`（反解 4 行）+ 模擬（numpy 重寫 `moc_ddm`/`dfp_ddm`） | 1–3 天 | 不用 R、不用 Stan；只有 order=2 |
| **C 混合** | Stan 換 PyMC，反解和模擬留 R；用 `reticulate` 或存 csv 交換 | 1 天 | 兩套環境都要裝 |

無論哪條，`lnrm2a` / `lnrm1` 路線都回不來，除非拿到檔案。

**owner 要決定的事**（其他人替你決定不了）：

1. `adaptiveSFT_functions.R:228` 舊 R 的 `if` 是「只看第一筆 draw」——修的時候要用
   `mean(alpha2) < 0`、`all(alpha2 < 0)`，還是逐 draw 篩？三種數值結果不同。
2. `lnrm2a.stan` / `lnrm1.stan` 去哪裡要。
3. `varZ` 到底是 SD 還是 variance（Stan 當 SD，R 的 `dlognormalrace` 當 variance）。
