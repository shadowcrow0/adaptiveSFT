# adaptiveSFT 整體移植到 Python：可行性評估與規劃

日期：2026-09-27
狀態：**評估 + 提案**。本文件不修改任何既有程式碼。引用的執行時間與數字都是本 session 實跑的
（Python 3.11、`requirements.txt` 釘住的 PyMC 5.28.5 / numpy 2.4.6 / scipy 1.17.1，4 核）。
與 `plan_grtv3ada_psi_python.md`（只講 Psi 路）的關係：本文件涵蓋**整個 repo**，Psi 路是其中一塊。

---

## 0. 一句話結論

**可行。3,792 行 R / Stan / 舊 Python 裡，六成已經有能跑的 Python 對應物，兩成有本 session 驗過的
概念驗證，剩下兩成是模擬迴圈與畫圖。估 9–10 個工作天。**
不能「逐行」移植的只有一類：**檔案不在 repo 裡的東西**（`lnrm0/lnrm1/lnrm2a.stan`、`post95.Rdata`、
兩個輸入 csv）——只能重建，重建版已存在但無法證明與原檔相同。

```
   adaptiveSFT (R / Stan)                            Python 對應                  狀態
   ──────────────────────────────────────────────────────────────────────────────────────
   lnrm2.stan + rstan                        →  PyMC：model_lnrm2.py / lnrm2_pymc.py /
                                                Visual_AudioWM/adaptivesft.models        ✔ 三個都能跑
   lnrm2a.stan（遺失）                       →  model_lnrm2a.py / adaptivesft(link=ogival) ✔ 重建版能跑
   lnrm0 / lnrm1.stan（遺失）                →  fix_params / link=linear                  ◐ 積木有，沒組
   find_salience_*（R）                      →  adaptivesft.salience                      ✔ 兩種介面
   dlognormalrace / plognormalrace（R）       →  poc/lnrm_race_poc.py                     ✔ PoC，對 MC
   diffIRT::simdiffT、dfp_ddm、moc_ddm       →  poc/psi_sft_poc.py                        ✔ PoC
   sft::sic / sicGroup                       →  poc/psi_sft_poc.py                        ✔ PoC，未對 R
   psiSimulation_functions.R（Psi）          →  AGRT.py agrtPsiObject（去 PsychoPy）       ✔ PoC
   simulateLNRM_ogival.R（682 行腳本）        →  要寫                                      ✘
   psi Simulation_26MAR2019.R（1003 行腳本）  →  要寫                                      ✘
   繪圖（eps、loess）                        →  matplotlib、statsmodels.lowess            ✘
   adaptive_sft2.py、Lab2RGB.py              →  作廢 / 換 colour-science                  —
```

---

## 1. 逐檔盤點

### 1.1 `adaptiveSFT_functions.R`（282 行）—— 核心函式庫

| 行 | 函式 | 做什麼 | Python 對應 | 狀態 |
|---|---|---|---|---|
| 1-5 | `require(rstan/diffIRT/sft)` | 載入 | 不需要 | 消失 |
| 9-18 | `getPr_ogival` | 後驗中 d(x) 落在目標 ± range 的比例 | 3 行 numpy | 未寫，沒人呼叫 |
| 21-57 | `getPr / get_log_pr / get_inv_log_pr` | 同上（多項式版），給註解掉的 `optim` 路（`:240-276`）用 | 不做 | 死碼 |
| 61-77 | `dlognormalrace` | 賽跑密度：勝者 pdf × 敗者 sf | `poc/lnrm_race_poc.py`；numba 版在 `model_lnrm2.py:67-73` | ✔ PoC，見 §2.3 |
| 81-111 | `plognormalrace / lnrm_adjusted_integral` | 賽跑 CDF（`integrate` + 失敗補救） | `scipy.integrate.quad`，不需補救 | ✔ PoC |
| 115-150 | `dfp_ddm` | 五種架構的 DFP 資料 | `poc/psi_sft_poc.py` | ✔ PoC |
| 153-165 | `moc_ddm` | 定值刺激法資料 | 3 行 | 未寫 |
| 168-173 | `dataframe2stan` | 整理成 Stan data | 不需要 | 消失 |
| 177-178 | `logit / inv_logit` | — | `scipy.special` | — |
| 180-199 | `find_salience_ogival` | 擬合 `lnrm2a.stan`，反解 `logit(targ/10)/slope + midpoint` | `adaptivesft.salience.find_salience_delta`（Houpt 介面）與 `find_salience`（正確率介面） | ✔ |
| 203-281 | `find_salience_polynomial` | 擬合 `lnrm2/lnrm1.stan`，二次式反解；`:228` 的 `if` 在 R ≥ 4.2 報錯 | `adaptivesft.salience._invert_delta_std`（quadratic / linear） | ✔，但 `if` 語意要決定（§4 D2） |

### 1.2 `lnrm2.stan`（48 行）—— 唯一存在的 Stan 檔

三個 Python 實作都能跑，本 session 實測：

| 實作 | 取樣器 | 參數化 | N | 時間 | 結果 |
|---|---|---|---|---|---|
| `model_lnrm2.py` | DEMetropolisZ（numba `pt.Op`，無梯度） | Stan 字面：`z = μ ∓ (αx + α₂x²)` | 1000 | 31 s（8 鏈 × 3000+3000） | 五參數回收，R-hat 1.01 |
| `lnrm2_pymc.py` | NUTS（純 PyMC 運算式） | 同上 | 500 | 44 s | 0 divergence |
| `Visual_AudioWM/adaptivesft.models`（`link="quadratic"`） | NUTS | `δ = 2(αx + α₂x²)`，`z = μ ∓ δ/2`（代數相同），**x 先標準化** | — | — | 未在本 session 跑 quadratic |

三者的 likelihood 數值已互相對過（`issue.md` M1–M7、`lnrm2_pymc_gaps.md`）。**要選一個當正式 API**（§4 D1）。

### 1.3 遺失的 Stan 檔與重建版

| 檔 | 誰用 | 重建版 | 差異 | 本 session 實測 |
|---|---|---|---|---|
| `lnrm2a.stan`（ogival） | `find_salience_ogival`、`simulateLNRM_ogival.R:62` | `model_lnrm2a.py`：`d = ½·L·inv_logit(slope·(x−midpoint))`，L=10 固定 | 照 R 的用法反推（`model_lnrm2a.py:1-45` 列 TODO-A1–A5） | 19 s（8 鏈 × 2000+2000）回收但 ESS 46–255；`log.md` 用 3000 才到 R-hat 1.01 |
| 同上 | 同上 | `adaptivesft.models(link="ogival")`：`D` 用 HalfNormal(2) 估、x 標準化 | 改了介面與先驗，不是重現 | 74 s，`sim_recovery` PASS（需 `colour-science`） |
| `lnrm1.stan`（線性） | `find_salience_polynomial(order=1)` | `adaptivesft(link="linear")`；或 `model_lnrm2.py` `fix_params={'alpha2': 0}` | 差先驗 | 未跑 |
| `lnrm0.stan`（每個強度層各自一個 d，無強度項） | `simulateLNRM_ogival.R:157`，只在 `fit.separate=TRUE`（`:14,148`）時 | 沒有；用 `fix_params` 把 `alpha2=0`、每層單獨擬合 `alpha` 即可 | — | 未寫 |

**兩個 ogival 重建版彼此也不同**（L 固定 vs 估、x 是否標準化、先驗、目標值介面）。移植時必須二選一或明確保留兩者並命名清楚。

### 1.4 `simulateLNRM_ogival.R`（682 行）—— LNRM 路的整場模擬

| 行 | 區段 | 依賴 | 移植難度 | 備註 |
|---|---|---|---|---|
| 1-81 | 產生 MOC 資料、擬合、反解 | `moc_ddm`、`lnrm2a.stan` | 低 | 每步都有對應物 |
| 83-140 | 參數收斂測試：N = 1…300，每個 N 擬合一次 | 300 次擬合；`post95.Rdata` 遺失（`:109-111`） | **計算量**：300 × ~20 s ≈ 100 分鐘單核；4 核 joblib ≈ 25 分鐘 | 畫圖用 `loess`（`:128-135`）→ `statsmodels.nonparametric.lowess` |
| 142-300 | 後驗預測檢查：每層 ecdf vs 模型 CDF | `plognormalrace`、`lnrm0.stan`（`:157`，只在 `fit.separate`）、`rsamp`、`postOpt.diff`（**未定義**，`:258, 281`） | 中 | `:204/:213` 傳 `varZ` 給 `sigmasq`（變異數位）→ 多開一次根號，原碼自身不一致（§3.4） |
| 308-468 | 單一乾淨 DFP：五種架構各畫 survivor + SIC | `dfp_ddm`、`sft::sic` | 低 | PoC 已做同一件事 |
| 470-585 | 整場實驗：10 位受試者參數抽樣（`:496-513`）→ 兩維各擬合 → DFP → `sicGroup` | 20 次擬合 + `sicGroup` | 中 | `:571-573` 的 `dp/dn/micp` 未初始化（issue R6） |
| 588-682 | `printsft` | 只印表 | 低 | — |

### 1.5 `psiSimulation_functions.R`（425 行）—— Psi 路的函式庫

| 行 | 函式 | Python 對應 | 狀態 |
|---|---|---|---|
| 3 | `pm.function` | `adaptivesft/psi.py::pm_function` | ✔ |
| 5-168 | `Est.Trial.Psi.Color` | `adaptivesft/psi.py` 的 `Psi`（逐行移植，網格照 R） | ✔ |
| 171 | `inv.pm.function` | `inv_pm_function` / `salience_levels()` | ✔ |
| 173-208 | `psi_color_ddm` | 反解 .99/.90 → DDM 驗證 | ✔ PoC |
| 214-373 | `Est.Trial.Psi.Orientation` | 與 Color 版**逐行重複**，只差範圍常數 | 合併成一個函式 |
| 378-413 | `psi_orientation_ddm` | 同上 | 合併 |
| 417-419 | `rmse_fun` | `scipy.optimize.minimize` | ✔ PoC |

### 1.6 `psi Simulation_25JUNE2018.R`（415 行）與 `psi Simulation_26MAR2019.R`（1003 行）

2019 版是 2018 版的修訂（`26MAR2019.R:195`「PER REQUEST FOR REVISITION (21FEB2019)」），結構相同、多了
PAR-AND / SER-OR / SER-AND 三段整場模擬（`:609-1003`）。**只移植 2019 版。**

| 行（2019） | 區段 | 依賴 | 難度 |
|---|---|---|---|
| 9-148 | 629 次 × 300 試 Psi 收斂模擬（方位） | Psi、DDM | 計算量：629 × 300 × 30 ms ≈ 95 分鐘；可平行 |
| 150-171 | 同上（顏色，論文未收） | 同上 | 可選 |
| 177-244 | 從硬寫的 H/L 產生 DDM 資料（`:208-237`） | `simdiffT` | 低 |
| 245-459 | 五種架構各畫 survivor + SIC | `dfp_ddm`、`sic` | 低 |
| 460-1003 | 四種架構各一場 10 人實驗 | `Psi_Simulation_SFTresults.csv`、`PsiDDM_Simulation_Pars.csv`（**遺失**，`:507,510`）；`psi_color_ddm` 三引數呼叫（issue R6） | 中：csv 只能重新產生 |

兩版腳本都有 Windows 硬路徑（`:2`）與 `windows()`；不移植。

### 1.7 其他

| 檔 | 處置 |
|---|---|
| `adaptive_sft2.py`（202 行，pystan 2、`^` 當次方、`allchannels` 未定義） | 作廢；功能已由 `adaptivesft.salience` 取代 |
| `Lab2RGB.py`（79 行，CIELAB→sRGB，MATLAB 翻譯） | 保留可用（純 numpy）；`Visual_AudioWM/adaptivesft/color.py` 用 `colour-science` 做 ΔE00，功能更全 |
| `model_lnrm2.py`、`model_lnrm2a.py`、`lnrm2_pymc.py` | 已是 Python；§4 D1 決定誰留誰併 |

### 1.8 外部 R 套件 → Python

| R | 用在哪 | Python | 狀態 |
|---|---|---|---|
| `rstan` | 擬合 | PyMC（已釘版本） | ✔ |
| `diffIRT::simdiffT` | 所有模擬 | 逐行移植（拒絕抽樣，`simdiffT.r:1-33`） | ✔ PoC |
| `sft::sic / sic.test / siDominance / mic.test` | SIC | 移植（`sft/R/sic.R:132-248`） | ✔ PoC（未對 R） |
| `sft::sicGroup` | 整場實驗決策表（`sic.R:75-105`） | 40 行 | 未寫 |
| `stats::ks.test`（單尾、漸近） | dominance | `scipy.stats.ks_2samp(alternative=…, method='asymp')` | ✔ PoC |
| `stats::anova(lm(…))`（序列 SS） | MIC ART | 手寫 F（兩個 OLS 的 RSS 差） | ✔ PoC |
| `stats::loess` | 收斂圖 | `statsmodels.nonparametric.lowess`（演算法不同，曲線會略異） | 未寫 |
| `stats::integrate` | `plognormalrace` | `scipy.integrate.quad` | ✔ PoC |
| `postscript / png` | 圖 | matplotlib | 未寫 |

---

## 2. 已經在 Python 裡、本 session 實跑確認的

| 指令 | 時間 | 結果 |
|---|---|---|
| `python model_lnrm2.py` | 31 s | 五參數回收，R-hat 1.01，ESS ≥ 858 |
| `python model_lnrm2a.py --n 1000 --tune 2000 --draws 2000 --chains 8` | 19 s | 回收但 ESS 46–255、R-hat ≤ 1.11（試次不夠；`log.md` 用 3000） |
| `python lnrm2_pymc.py` | 44 s | NUTS，0 divergence |
| `python -m adaptivesft.sim_recovery`（在 `Visual_AudioWM`） | 74 s | PASS：六參數落在 94% HDI、0 divergence；反解 0.65–0.95 回代誤差 < .001 |
| `poc/psi_sft_poc.py` 等三支 | 3 分鐘 | 見 `plan_grtv3ada_psi_python.md` §5 |
| `poc/lnrm_race_poc.py` | 1 s | 見 §2.3 |

### 2.3 `dlognormalrace / plognormalrace` 的 scipy 版對得上

```
   兩個勝者的密度總積分            = 1.000000
   P(累積器 0 先到) 積分            = 0.9703     封閉解 Φ(2d/(σ√2)) = 0.9703
   CDF  x=0.5 / 1.0 / 2.0 / 4.0    = .0027 / .0838 / .4541 / .8555
   蒙地卡羅 200k                    = .0028 / .0835 / .4545 / .8564
```

R 的 `lnrm_adjusted_integral`（`:91-111`）三層 `tryCatch` 是在補 `integrate` 的失敗，
scipy `quad` 從 `psi` 起積就不需要；而且那段補救碼本身有 `sigmasqx` 打錯字（`:98`，issue R6）。

---

## 3. 不能「逐行移植」、只能「重建或決定」的地方

### 3.1 遺失檔案（§1.3、§1.6）

重建版的參數化與先驗是猜的（`model_lnrm2a.py` TODO-A1–A5）。原作者口頭確認「a/b/c 都是 lnrm2 的微改」，
方向對，細節不能證明。**移植文件必須標明哪些數字來自重建版。**

### 3.2 `if (post.diff$alpha2 < 0)`（`adaptiveSFT_functions.R:228`）

舊 R 只看第一筆 draw；新 R 報錯。三種修法數值不同（`issue.md` R1 的表）。Python 版要選一種並寫成測試。

### 3.3 `h_targ / l_targ` 的介面

Houpt 的目標值是**漂移差**（`simulateLNRM_ogival.R:24-25`：1.3 / 8.0，配 L=10），`adaptivesft` 改成**正確率**。
兩者可以並存（`find_salience_delta` / `find_salience`），但腳本移植時要決定預設用哪個，
因為 §1.4 的整場模擬用的是漂移差。

### 3.4 `varZ` 的尺度在 R 內部就不一致

`lnrm2.stan:38` 把 `varZ` 放標準差位；`dlognormalrace`（`:61-62`）收 `sigmasq` 再開根號；
`simulateLNRM_ogival.R:204, 213, 258, 281` 卻直接把 `varZ` 傳進 `sigmasq`。所以 R 的後驗預測圖
用的是 `√varZ` 而不是 `varZ`——**原碼就是錯的**。Python 版要用 `varZ`（= σ）直接傳，並在文件註明
這與 R 原圖不同。

### 3.5 `DDM.pCorrect` 差一個 2 倍

`psi Simulation_26MAR2019.R:117` 用 `1/(1+exp(−2·a·s·v))` 當「真」曲線，`simdiffT.r:6` 實際是
`1/(1+exp(−a·drift))`。重現論文的「真 α/β」（`:120-121` 的 `optim`）時會差在這裡。

### 3.6 原作者自己的 `.99` 也落在刺激範圍外

`psi Simulation_25JUNE2018.R:174, 177`：`highSalience.color = 101.64`、`lowSalience.color = 53.65`，
刺激範圍是 [−55, 50]（`:171`）。2019 版改成硬寫 `50` / `16.45`（`26MAR2019.R:208, 211`）——
即直接用範圍上限當 H。這是 `plan_grtv3ada_psi_python.md` §6 R3 的獨立證據。

### 3.7 未定義變數（issue R6）

`postOpt.diff`（`simulateLNRM_ogival.R:258`）、`rsamp`、`dp/dn/micp`、`N`——R 腳本從來沒有完整跑過一遍。
Python 版不是「翻譯」而是「照意圖重寫」，這點要在 README 講明。

---

## 4. 要先決定的事（Phase 0）

| # | 決定 | 選項 | 建議 |
|---|---|---|---|
| D1 | LNRM 正式實作用哪個 | (a) `adaptivesft.models`（NUTS、三種 link、標準化、正確率介面） (b) `model_lnrm2.py`（Stan 字面、numba、DEMetropolisZ） | **(a) 當 API，(b) 留作 oracle 測試**：兩者逐點 logp 差 ≤ 1e−12 進 CI |
| D2 | `alpha2 < 0` 的語意 | 刪 `if` / `all()` / `mean()` / 逐 draw 篩 | 逐 draw 篩並回報丟棄比例（`adaptivesft.salience` 的 `reachable` 已是這個做法） |
| D3 | 目標值介面 | 漂移差 / 正確率 / 兩者 | 兩者並存，腳本預設正確率，重現論文時用漂移差 |
| D4 | ogival 的 `L` | 固定 10 / 估 | 估（`adaptivesft`），但提供 `fix_params={'D': 10}` 重現原碼 |
| D5 | `varZ` 命名 | 保留 `varZ` / 改 `sigma` | 改 `sigma`，讀 R 資料時提供別名 |
| D6 | 程式碼放哪 | 本 repo / `Visual_AudioWM/adaptivesft` | **本 repo**，把 `Visual_AudioWM/adaptivesft` 搬進來成子套件，那邊改成依賴 |
| D7 | R 對照要做到什麼程度 | 不做 / 只對 SIC / 全對 | 只對 `sft::sic`（純數值，可逐位元）；LNRM 對 R 沒意義（取樣器不同） |
| D8 | 收斂測試的 N 範圍 | 1…300（原碼） / 稀疏 | 稀疏（20 個 N，log 間距），25 分鐘 → 3 分鐘 |

---

## 5. 目標結構

```
   adaptiveSFT/
   ├── adaptivesft/                      ← 從 Visual_AudioWM 搬來，擴充
   │   ├── lnrm/
   │   │   ├── models.py                 fit_lnrm(link=ogival|quadratic|linear, fix_params)
   │   │   ├── race.py                   numba 逐題 logp（自 model_lnrm2.py）、d/p lognormalrace
   │   │   └── salience.py               find_salience（正確率）、find_salience_delta（漂移差）
   │   ├── psi/
   │   │   ├── core.py                   agrtPsiObject 去 PsychoPy 版
   │   │   └── salience.py               salience_levels（同側多目標）
   │   ├── sim/
   │   │   ├── ddm.py                    simdiffT、moc_ddm、dfp_ddm
   │   │   └── participants.py           simulateLNRM_ogival.R:496-513 的參數抽樣
   │   ├── sft/
   │   │   └── sic.py                    sic、sic_test、si_dominance、mic_test、sic_group
   │   └── color.py                      現有
   ├── scripts/
   │   ├── simulate_lnrm_ogival.py       simulateLNRM_ogival.R 四段，各一個 --section
   │   ├── simulate_psi.py               psi Simulation_26MAR2019.R
   │   ├── convergence.py                LNRM 與 Psi 的收斂圖
   │   └── plots.py                      survivor / SIC / 心理計量圖
   ├── tests/
   │   ├── test_race_logp.py             adaptivesft.lnrm vs model_lnrm2 numba oracle（1e−12）
   │   ├── test_race_cdf.py              plognormalrace vs 蒙地卡羅
   │   ├── test_ddm.py                   反應機率 vs 1/(1+exp(−a·drift))；sdv=0 時平均 RT vs 解析式
   │   ├── test_sic_vs_r.py              固定資料 vs R sft::sic 輸出（存成 json，R 只跑一次）
   │   └── test_psi_recovery.py          累積常態受試者回復
   ├── R/                                ← 原 R/Stan 檔搬進來，只讀，附「已被哪個 Python 取代」
   ├── model_lnrm2.py …                  ← 搬進 adaptivesft/lnrm/ 或 legacy/
   └── poc/                              ← 移植完成後刪除或搬進 tests
```

### 5.1 依賴

全部 `pip` 可裝；`requirements.txt` 已釘住前六個（本 session 實裝、實跑）。

| 套件 | 版本 | 用在哪 | 必要 |
|---|---|---|---|
| `numpy` | 2.4.6 | 全部 | ✔ |
| `scipy` | 1.17.1 | `norm/lognorm`、`quad`、`ks_2samp`、`erfinv`、`optimize` | ✔ |
| `pymc` | 5.28.5 | `fit_lnrm`、`pm.sample`、`CustomDist / Potential` | ✔ |
| `pytensor` | 2.38.3 | **PyMC 的計算圖後端**；`model_lnrm2.py:21-22` 的自訂 `pt.Op` 與 `pytensor.graph.Apply`、`adaptivesft/models.py:43` 的 `pt.where / pt.erfc` 都直接用它。PyMC 會把它當依賴帶進來，但 `pt.Op` 介面隨版本變，**必須釘版本** | ✔ |
| `numba` | 0.65.1 | `model_lnrm2.py` 的逐題 logp（`race.py` oracle） | ✔（oracle 測試） |
| `arviz` | 0.23.4 | `az.summary`、`InferenceData` | ✔ |
| `statsmodels` | — | `lowess`（收斂圖，取代 R `loess`） | 只有 P6 |
| `matplotlib` | — | 全部圖 | 只有 P6 |
| `joblib` | — | 收斂測試平行 | 只有 P6 |
| `colour-science` | — | `adaptivesft.color` 的 ΔE00 | 可選（風險 8） |
| `pytest` | — | `tests/` | 開發 |
| `psychopy` | — | 只有真人實驗腳本 | 不進套件依賴 |

不需要：R、rstan、cmdstan、pystan、diffIRT、sft。

---

## 6. 分階段規劃

```
   P0 決策 ─┐
            ▼
   P1 骨架 + LNRM 合併 ──► P2 race 密度/CDF + 反解 ──► P3 DDM ──► P4 SIC ──► P5 Psi
                                                                    │ 閘門 R 比對
                                                                    ▼
                                            P6 腳本重寫（ogival 四段 + psi 2019）──► P7 對照、文件、CI
   ──────────────────────────────────────────────────────────────────────────────────────
   估時  0.5   1.5   1   0.5   1.5   1   2   1   ＝ 9–10 工作天（不含等決策、不含 R 環境安裝 5 分鐘）
```

### P0 決策（0.5 天，研究者）
§4 的 D1–D8。

### P1 套件骨架與 LNRM 合併（1.5 天）
- 建 `adaptivesft/` 目錄結構；搬 `Visual_AudioWM/adaptivesft`；把 `model_lnrm2.py` 的 numba 積木搬進 `lnrm/race.py`。
- `fit_lnrm` 加 `fix_params`（重現 `lnrm0/lnrm1` 與固定 `D=10`）。
- **閘門**：對同一份資料，`adaptivesft.lnrm` 的模型 logp 與 numba oracle 差 ≤ 1e−12（`issue.md` M 系列已有方法）。

### P2 race 密度 / CDF 與反解（1 天）
- `dlognormalrace / plognormalrace`（PoC 已對 MC）；`getPr_ogival`。
- `find_salience_polynomial` 的二次式反解與 D2 語意；`find_salience_ogival` 的 `logit(targ/L)/slope + midpoint`。
- **閘門**：`test_race_cdf`（MC 差 ≤ .002）；D2 的三種語意各一個測試，鎖定選定的那一種。

### P3 DDM 模擬（0.5 天）
- 從 PoC 搬 `simdiffT / dfp_ddm`，補 `moc_ddm` 與參與者參數抽樣。
- **閘門**：`test_ddm`。

### P4 SIC（1.5 天）
- 從 PoC 搬 `sic` 系列，補 `sic_group` 決策表（`sic.R:75-105`）。
- **閘門（必過）**：跑 `setup_r.sh` 裝 R + sft（apt + GitHub 鏡像，5 分鐘），固定種子資料同時餵 R 與 Python，
  SIC 值與 D± 差 ≤ 1e−12、KS p 值 ≤ 1e−10、ART 的 F ≤ 1e−8；R 輸出存成 `tests/data/sic_r_oracle.json`，之後 CI 不需要 R。

### P5 Psi（1 天）
- 從 PoC 搬 `agrtPsiObject` 去 PsychoPy 版與 `salience_levels`；β 上限可覆寫。
- **閘門**：`test_psi_recovery`（144 試 α 誤差 ≤ 2、β ≤ 2.5）。

### P6 腳本重寫（2 天）
- `simulate_lnrm_ogival.py`：四段各一個 `--section`；收斂測試依 D8 稀疏化並用 `joblib` 平行；後驗預測用 `varZ`（§3.4）。
- `simulate_psi.py`：2019 版；csv 自己產生；`Est.Trial.Psi.*` 合併成一個。
- `plots.py`：survivor / SIC / 心理計量 / 收斂圖，matplotlib。
- **閘門**：每個 section 在預設參數下跑完不報錯；輸出欄位與原 R 的 csv 一致（`ParallelOR_Psi_Simulation_*.csv` 欄名）。

### P7 對照、文件、CI（1 天）
- README：每個 R 檔對應到哪個 Python 模組、哪些數字來自重建版、與原 R 已知的差異（§3）。
- `pytest` 進 CI（不需 R、不需 PsychoPy；PyMC 測試用小 N）。
- `issue.md` 每項標最終狀態。

---

## 7. 風險

| # | 風險 | 影響 | 對策 |
|---|---|---|---|
| 1 | 兩個 ogival 重建版不同（§1.3） | 數字對不上彼此、對不上論文 | D1/D4 定一個；另一個用 `fix_params` 重現 |
| 2 | 重建 ≠ 原檔（§3.1） | 無法宣稱「重現」 | 文件標明；向原作者要檔 |
| 3 | R 腳本從未完整跑過（§3.7） | 沒有 R 端的黃金輸出可對 | 只對純數值元件（SIC、DDM、race CDF）；LNRM 對自回復 |
| 4 | 收斂測試計算量（§1.4） | 100 分鐘 | D8 稀疏化 + 平行 |
| 5 | R 環境：CRAN 被擋（`bug.md` §1.3） | P4 閘門 | `setup_r.sh` 走 apt + GitHub；或環境設定開 CRAN |
| 6 | PyMC 版本綁定 | 升版後 `pt.Op` 或 `CustomDist` 行為改 | `requirements.txt` 已釘；CI 固定版本 |
| 7 | `loess` → `lowess` 曲線略異 | 只影響圖 | 註明 |
| 8 | `colour-science` 是 `adaptivesft.color` 的硬依賴 | PsychoPy 內建 Python 沒有 | 維持「離線產 LUT」原則（`Visual_AudioWM/CLAUDE.md`），`color.py` 改成可選 import |

---

## 8. 不做

- 不修 R / Stan 檔（S1 五行、R1 一行、R6 八處）；R 檔搬進 `R/` 只讀。
- 不移植 `psi Simulation_25JUNE2018.R`（被 2019 版取代）、`adaptive_sft2.py`、註解掉的 `optim` 路（`adaptiveSFT_functions.R:240-276`）。
- 不做 `sft::capacity` 系列。
- 不在此容器跑 PsychoPy。

---

## 9. 本文件引用的行號（寫入前已比對）

| 檔案 | 行 | 內容 |
|---|---|---|
| `adaptiveSFT_functions.R` | 9-18, 21-57, 61-77, 81-111, 98, 115-150, 153-165, 168-173, 180-199, 203-281, 228, 240-276 | §1.1 各函式 |
| `lnrm2.stan` | 38 | `varZ` 在標準差位 |
| `simulateLNRM_ogival.R` | 14, 24-26, 62, 86, 109-111, 128-135, 148, 157, 204, 213, 258, 281, 496-513, 571-573 | §1.4、§3.4 |
| `psiSimulation_functions.R` | 3, 5-168, 171, 173-208, 214-373, 378-413, 417-419 | §1.5 |
| `psi Simulation_25JUNE2018.R` | 2, 171, 174, 177 | §3.6 |
| `psi Simulation_26MAR2019.R` | 117, 120-121, 195, 208-237, 507, 510, 609-1003 | §1.6、§3.5 |
| `model_lnrm2.py` | 67-73 | numba race logpdf |
| `model_lnrm2a.py` | 1-45 | TODO-A1–A5 |
| `Visual_AudioWM/adaptivesft/color.py` | 19 | `import colour` |
| `Visual_AudioWM/AGRT.py` | 133 | 反應機率模型 |
| `sft/R/sic.R`（CRAN 鏡像） | 75-105, 132-248 | 決策表、`sic` 系列 |
| `diffIRT/R/simdiffT.r`（CRAN 鏡像） | 1-33, 6 | 拒絕抽樣、反應機率 |
| `poc/psi_sft_poc.py` | 14-51 | Psi 去 PsychoPy 版 |

---

## 10. 進度（2026-09-27）

P0 的決定依「以本 repo 設定為準」定案，P1–P5 已做完並推上分支，程式在 `adaptivesft/`，測試在 `tests/`：

| 決定 | 定案 |
|---|---|
| D1 LNRM 實作 | `model_lnrm2.py` 的 numba 積木 + DEMetropolisZ，四個模型共用一個 Op（`adaptivesft/models.py`） |
| D2 `alpha2 < 0` 語意 | `alpha2_rule` 參數，預設 `all_draws`（舊 R 成功路徑），五種可選 |
| D3 目標值介面 | 漂移差（`h_targ / l_targ`）為主，正確率（`acc_high / acc_low`）為輔 |
| D4 ogival 的 L | 固定 10（`simulateLNRM_ogival.R:26`） |
| D5 `varZ` 命名 | 保留 `varZ`，明定為 SD |
| D6 程式碼位置 | 本 repo `adaptivesft/` |
| D7 R 對照 | 只對 `sft::sic`（`tests/data/make_sic_oracle.R`） |
| D8 收斂測試 N | 8 個 log 間距的點（3–300），`--full` 才 1…300 |

| Phase | 狀態 | 證據 |
|---|---|---|
| P1 骨架 + LNRM | ✔ | `tests/test_models.py`：Op = numba（1e−12）、四種 link 小樣本回收 |
| P2 race 密度 / CDF + 反解 | ✔ | `tests/test_race.py`、`tests/test_salience.py` |
| P3 DDM | ✔ | `tests/test_ddm.py` |
| P4 SIC | ✔（R 逐位元比對見 `tests/test_sic.py::test_against_r_oracle`） | `tests/test_sic.py` |
| P5b Psi 對 R | ✔ 逐試 1e−13，顏色與方位兩組網格 | `tests/test_parity_demo.py` |
| P5 Psi | ✔（R 逐行移植，不用 AGRT.py） | `tests/test_psi.py` |
| P6 腳本重寫 | ✔（兩種 a 慣例並排） | `scripts/`、`p6_results.md`、`results/p6/` |
| P7 文件、CI | ✔ `pyproject.toml`、`.github/workflows/tests.yml`、頂層 README、`issue.md` §8；LNRM 對 Stan 已比對（cmdstanr 2.40.0，後驗平均差 ≤ 0.01 SD，`tests/test_lnrm_vs_stan.py`） | — |

