# 用 GRTv3_ada 的方式改造 adaptiveSFT：可行性評估與執行規劃

日期：2026-09-25
狀態：**評估 + 提案**。附 `poc/` 三支概念驗證腳本，本文件引用的數字都是它們實跑出來的。
不修改任何既有程式碼。

---

## 0. 一句話結論

**可行，而且比「修 R + Stan」便宜**。理由只有一個：adaptiveSFT 裡本來就有第二條路——
2018–2019 的 `psiSimulation_functions.R` 是 Psi 適應法的 R 版；`Visual_AudioWM/AGRT.py`
是同一個演算法的 Python 版，公式逐字相同。所以「GRTv3_ada 方式」不是把 SFT 換成別的
理論，是把原作者自己已經做過、但同樣跑不起來的那條路，用已經能跑的 Python 接回去。

```
   adaptiveSFT 的兩條路                      現況              換成 Python 之後
   ──────────────────────────────────────────────────────────────────────────────
   A  LNRM 路   moc → lnrm2.stan → 反解      壞 (issue.md)     選配：adaptivesft 套件已重建
   B  Psi  路   Psi → inv.pm → DFP → sic      壞 (R6, 缺 csv)   主線：AGRT.py 的 Psi + 三段移植
```

要新寫的只有三塊，PoC 已經寫出來並驗過：

| 要寫的 | 行數 | 來源 | PoC 狀態 |
|---|---|---|---|
| DDM 受試者模擬 | 30 | `diffIRT::simdiffT` 逐行移植 | ✔ |
| DFP 資料產生 | 25 | `adaptiveSFT_functions.R:115-150` `dfp_ddm` | ✔ |
| SIC 檢定 | 60（＋決策表 40） | `sft/R/sic.R` `sic / sic.test / siDominance / mic.test` | ✔ 簽名對；未與 R 逐位元比對 |
| 把 AGRT 的「一個正確率、α 兩側」改成 SFT 的「兩個正確率、同側」 | 5 | `psiSimulation_functions.R:171` | ✔ |

**但有一個取捨必須先講**：Psi 校準的是**正確率**，SFT 需要的是 **RT 分離**。Houpt 2018 用
LNRM 的原因就在這——他的目標值 `h_targ / l_targ` 是漂移差（drift 單位），直接綁在 RT 分布上。
PoC 用原作者 `psiSimulation` 的 DDM 參數跑：`.99` 在刺激範圍內**不可達**，`.95/.80` 給出的
RT 分離**不足以讓 SIC 顯著**（§5）。這不是程式問題，是設計參數問題，規劃裡放了一個前測閘門（Phase 4）。

---

## 1. 兩邊各有什麼

### 1.1 模型是同一條式子

**程式碼字面**，`psiSimulation_functions.R:3`：

```r
pm.function <- function (x,a,b,d) .5 * d + (1-d) * pnorm(x,a,b)
```

**程式碼字面**，`Visual_AudioWM/AGRT.py:133`：

```python
self._probResponseGivenLambdaX = np.array([0,1]).reshape(2,1,1,1) + np.array([1,-1]).reshape(2,1,1,1) * ((self.delta/2) + (1 - self.delta) * stats.norm.cdf(self._alpha, loc=self._x, scale=self._beta))
```

兩者都是：

```
   P(r = 1 | x; α, β, δ)  =  δ/2  +  (1 − δ) · Φ( (x − α) / β )

   α  主觀決策界線      β  知覺標準差      δ  lapse
```

（R 寫 `pnorm(x, a, b)`，Python 寫 `norm.cdf(α, loc=x, scale=β)` 再用 `[1,-1]` 翻轉，代數上相同。）
更新（`AGRT.py:140-158`）、估計（`:160-161`）、選下一題（`:157`）跟 R 的
`psiSimulation_functions.R:111-150` 是同一套 4 維陣列 `[r, α, β, x]` 的熵最小化。

### 1.2 誰有什麼、誰能跑

```
   adaptiveSFT (R / Stan / 半成品 Python)          Visual_AudioWM (Python)
   ────────────────────────────────────────        ──────────────────────────────────────
   psiSimulation_functions.R                        AGRT.py  agrtPsiObject      ✔ 純 numpy
     Est.Trial.Psi.Color  ← R 版 Psi   ✘ R6        AGRT.py  AGRTHandler        需 PsychoPy
     inv.pm.function      ← 反解        ✔ 1 行      GRTv3_ada.py  三區塊校準流程  需 PsychoPy
     psi_color_ddm        ← DDM 驗證    ✘ diffIRT   adaptivesft/  PyMC 版 LNRM   ✔ 已驗證
   adaptiveSFT_functions.R                            models.fit_lnrm(link=ogival|quadratic|linear)
     dfp_ddm / moc_ddm    ← DFP 產生    ✘ diffIRT     salience.find_salience(acc_high, acc_low)
     find_salience_*      ← LNRM 反解   ✘ rstan
   lnrm2.stan             ← 只有這一支  ✘ 語法
   psi Simulation_*.R     ← 整場模擬    ✘ 缺 csv
   model_lnrm2.py / model_lnrm2a.py     ✔ 已驗證（本 repo 的 PyMC 版）
```

`Visual_AudioWM/adaptivesft/`（commit `9b73c21`，2026-08-13）是 LNRM 路的完整 Python
重建，`README.md:81-86` 自述回復通過、0 divergence。在本提案裡它從主線退成**選配的交叉檢核**。

---

## 2. 「GRTv3_ada 方式」套到 SFT 上長什麼樣

### 2.1 GRTv3_ada 現在的流程（`GRTv3_ada.py:494-515, 1173-1199, 2272-2296`）

```
   區塊 1  只校顏色  N_ADAPT_COL=72     Psi₁ 提 x → 兩色 ±|x| → 受試者答 → 位元餵 Psi₁
   區塊 2  只校聲音  N_ADAPT_SND=72     Psi₂ 提 s → 兩級 ±|s| → 受試者答 → 位元餵 Psi₂
        │
        ▼  estimateGRTintensities(OVERALL_ACC)  →  (c_lo, c_hi), (s_lo, s_hi)
   區塊 3  主實驗：練習 15 → 576 試 GRT identification（4 刺激）
```

### 2.2 SFT 版

```
   區塊 1  只校維度 1（例：顏色）  Psi₁ 提 x → 單維 2AFC 速度作業 → 反應餵 Psi₁
   區塊 2  只校維度 2（例：方位）  Psi₂ 提 y → 同上                → 反應餵 Psi₂
        │
        ▼  salience_levels(α, β, δ, [p_H, p_L])   ← 換掉 estimateGRTintensities
        │      H₁, L₁  /  H₂, L₂
   區塊 3  主實驗：DFP 2×2（HH, HL, LH, LL）× n/cell，記 RT + 正確
        │
        ▼  sic(HH, HL, LH, LL)  →  D⁺, D⁻, MIC, dominance → 判架構
```

### 2.3 要改的只有「反解」那一步

AGRT 的 `estimateThreshold`（`AGRT.py:163-171`）回傳**一個**正確率在 α **兩側**的一對值：

```
   x_lo = α − β·√2·erfinv( (2·√p − δ)/(1 − δ) − 1 )
   x_hi = α − β·√2·erfinv( (2·(1 − √p) − δ)/(1 − δ) − 1 )        ← GRT 的 2 維聯合正確率要開根號
```

SFT 要的是**兩個**正確率（H、L）在**同一側**（`psiSimulation_functions.R:171, 183-184`）：

```
   x_p  =  α  +  β · Φ⁻¹( (p − δ/2) / (1 − δ) )        p ∈ { p_H, p_L }

   原碼：inv.pm.function <- function (y,a,b,d) qnorm((y-.5*d)/(1-d), a, b)
         H = inv.pm.function(.99, …)   L = inv.pm.function(.90, …)
```

另外 AGRT 的 lapse 是**邊際** lapse `1 − √(1 − λ)`（`AGRT.py:288`），因為 GRT 的正確率是兩維聯合。
SFT 每維單獨校準，δ 應直接用 λ。這兩處合計 5 行，其餘 Psi 原封不動。

---

## 3. issue.md 裡的每個問題在新架構下怎麼了

| 編號 | 問題 | 新架構 | 為什麼 |
|---|---|---|---|
| R1 | `if (post.diff$alpha2 < 0)` 長度 > 1 | **消失** | 不再有 LNRM 反解；Psi 反解是純量 α、β |
| R2 | StanHeaders / stanc 版本 | **消失** | 不呼叫 rstan，也不 `source()` 任何 R |
| R3 | `permute=` 拼字 | **消失** | 同上 |
| R4 | stringsAsFactors 等 | 本來就沒中 | — |
| R5 | `lnrm2a/lnrm1/lnrm0.stan`、`post95.Rdata` 遺失 | **不再擋路** | 主線不用 LNRM；交叉檢核用 `adaptivesft` 重建的 ogival |
| R5 | `Psi_Simulation_SFTresults.csv`、`PsiDDM_Simulation_Pars.csv` 遺失（`26MAR2019.R:507,510`） | **不再擋路** | 那兩個 csv 是舊模擬的中間產物；Python 重跑模擬自己產生 |
| R6 | 2018 R 腳本的 8 個 bug | **消失** | 整支不再執行；邏輯用 Python 重寫（Phase 4） |
| S1 | `real x[N]` 舊語法 | **消失** | 不編 Stan |
| S2 | 三個 Stan 檔遺失 | 同 R5 | — |
| S3 | `varZ` 是 SD 還是 variance | **只剩選配** | 只有 LNRM 交叉檢核會碰到；`adaptivesft/models.py:33-36` 已統一叫 `sigma` |
| P1–P5 | `adaptive_sft2.py` 五個 bug | **消失** | 整檔作廢 |
| M1–M9 | PyMC 移植的 9 個問題 | **降為選配** | 主線不跑 PyMC；`model_lnrm2.py` 留作 RT 交叉檢核 |

**新增的問題**（原架構沒有、換過來才有）：

| 編號 | 問題 | 嚴重度 | 見 |
|---|---|---|---|
| N1 | Psi 校準正確率，SFT 需要 RT 分離 | **高**（設計層） | §5.3、§6 R1 |
| N2 | AGRT 的 β 網格上限由刺激範圍推出，太窄就把 β 釘死 | 高（參數層） | §5.2、§6 R2 |
| N3 | `.99` 落在心理計量函數尾端，估計誤差被 Φ⁻¹(.99)=2.33 放大 | 中 | §6 R3 |
| N4 | `AGRT.py:64-67` 模組層 `import psychopy`，離線模擬時要拆 | 低（工程） | Phase 1 |
| N5 | SIC 移植未與 R `sft::sic` 逐位元比對 | 中 | Phase 3 閘門 |

---

## 4. 用 Python 取代 .stan 與 R：逐元件對照

| R / Stan 元件 | 位置 | Python 對應 | 狀態 | 驗證方式 |
|---|---|---|---|---|
| Psi 內核（4 維陣列、熵最小化） | `psiSimulation_functions.R:22-150` | `AGRT.agrtPsiObject`（`AGRT.py:75-184`） | **已有** | PoC A：累積常態受試者回復 |
| β 網格範圍 | （R 寫死 `b.range <- c(1,50)`，`:17`） | `AGRT.py:295-303` 由刺激範圍推出 | 已有，**要注意 N2** | PoC B |
| 反解 H / L | `psiSimulation_functions.R:171, 183-184` | `salience_levels()`（PoC，5 行） | **已寫** | 回代 DDM 看實際正確率 |
| DDM 受試者 | `diffIRT::simdiffT`（`simdiffT.r:1-33`） | `simdiffT()`（PoC，30 行，逐行移植含拒絕抽樣） | **已寫** | 對照 `p = 1/(1+exp(−a·drift))`（`simdiffT.r:6`） |
| DFP 資料產生 | `adaptiveSFT_functions.R:115-150` | `dfp_ddm()`（PoC，25 行） | **已寫** | 五架構 SIC 簽名（PoC C） |
| MOC 資料產生 | `adaptiveSFT_functions.R:153-165` | 3 行 | 未寫（選配，LNRM 路才用） | — |
| SIC 函數與 KS 檢定 | `sft/R/sic.R:132-187` | `sic()`（PoC） | **已寫** | 簽名對；**待與 R 逐位元比對** |
| 選擇性影響（dominance） | `sft/R/sic.R:189-216` | `scipy.stats.ks_2samp(alternative=…, method='asymp')` | **已寫** | 同上 |
| MIC ART 檢定 | `sft/R/sic.R:219-248` | 對齊 → 排名 → 兩因子 OLS 序列 SS 的 F | **已寫** | 同上 |
| `sicGroup` 決策表（rejected models → 架構） | `sft/R/sic.R:66-98` | 40 行 | 未寫 | 逐條對照 |
| 整場模擬迴圈 | `psi Simulation_26MAR2019.R:473-590` | Phase 4 | 未寫 | 重現論文表格的形狀 |
| 繪圖（survivor、SIC） | 同上 | matplotlib | 未寫 | — |
| LNRM 擬合（`lnrm2.stan`） | `adaptiveSFT_functions.R:203-281` | `adaptivesft.fit_lnrm` / `model_lnrm2.py` | **已有** | 已驗證（各自 README / log.md） |
| `sft::capacity`、`estimate.bounds` | — | 不做 | — | adaptiveSFT 流程沒用到 |

沒有任何一個元件需要 R、Stan、pystan、cmdstan。PsychoPy 只在真人實驗那一層需要。

---

## 5. PoC 實測結果

環境：本容器，Python 3.11 venv，numpy 2.4.6、scipy 1.17.1。無 R、無 PsychoPy。
腳本與跑法見 `poc/README.md`。

### 5.1 Psi 移植本身沒問題（`psi_sft_poc_recovery_power.py` A 段）

受試者就是模型假設的累積常態，α=6、β=15、λ=.02；刺激範圍 [−55, 50]、100 格；20 次重複：

| 校準試次 | α̂（真值 6） | β̂（真值 15） |
|---|---|---|
| 72 | 6.74 ± 2.35 | 15.00 ± 2.78 |
| 144 | 6.16 ± 1.92 | 14.75 ± 2.19 |
| 300 | 6.70 ± 1.28 | 15.32 ± 1.72 |

每試更新約 30 ms（144 試合計 4 s 純計算），線上出題沒有延遲問題。

### 5.2 DDM 受試者：β 被網格上限釘死（B 段）

用 `psiSimulation_functions.R:96-99, 104` 的參數（a=1.45、v=1.6、sdv=.25、thres50=6）：

```
   把 DDM 的 P(correct | x) 用累積常態擬合  →  α ≈ 5.96,  β ≈ 32.0
   AGRT 依刺激範圍推出的 β 上限（AGRT.py:295）  →  22.6
   Psi 144 試 × 10 次                          →  β̂ = 20.9 ± 0.33   ← SD 小得不自然：撞到上限
   x = 50（範圍最大值）時 P(correct)            →  0.906              ← .99 在範圍內不可達
```

這正是 `GRTv3_ada.py:907` 註解記的現象（「範圍外、β 撐爆」）。原因是 `AGRT.py:295` 假設
「刺激範圍端點 = 99% 正確」來推 β 上限；受試者比這個假設鈍，β 就沒地方放。

補充：原作者的 R 腳本 `psi Simulation_26MAR2019.R:117` 算「真」曲線用的是
`exp(a·s·v)/(exp(−a·s·v)+exp(a·s·v))` = `1/(1+exp(−2·a·s·v))`，而 `simdiffT.r:6` 的實際
反應機率是 `1/(1+exp(−a·drift))`，差一個 2 倍。這是**推導**出的不一致，不影響本提案，但若要重現
論文數字要先釐清。

### 5.3 SFT 檢定力：正確率目標 ≠ RT 分離（B 段）

同一組 DDM，用 Psi 估出的 α̂、β̂ 反解，回代 DDM 3000 試看實際正確率，再跑 DFP 五種架構，
每種 5 次，記 p < .05 的比例：

| 目標 (H/L) | H 實際 acc | L 實際 acc | n/cell | PAR-OR D⁺ | PAR-AND D⁻ | COA D⁻ | MIC 任一 |
|---|---|---|---|---|---|---|---|
| .95 / .80 | .853 | .722 | 250 | 0.2 | 0.0 | 0.4 | ≤ 0.2 |
| .95 / .80 | .853 | .722 | 1000 | 0.0 | 0.0 | 0.2 | ≤ 0.2 |
| .90 / .70 | .788 | .627 | 250 | 0.0 | 0.0 | 0.0 | ≤ 0.2 |
| .90 / .70 | .788 | .627 | 1000 | 0.0 | 0.0 | 0.0 | ≤ 0.4 |

**用這組 DDM 參數，任何可達的正確率目標都給不出 SIC 能看見的 RT 分離。**
對照 Houpt 的 `simulateLNRM_ogival.R:24-26`：`l_targ = 1.3`、`h_targ = 8.0`（drift 單位，a=3、v=2），
是刻意拉開的巨大分離。

### 5.4 SIC 移植對得上理論簽名（`psi_sft_poc_sic_signature.py`）

drift H=3.0 / L=1.0，a=3，250/cell，10 次：

| 架構 | D⁺ 顯著 | D⁻ 顯著 | MIC 顯著（符號） | dominance 通過 |
|---|---|---|---|---|
| PAR-OR | 1.0 | 0.0 | 1.0 (+) | 1.0 |
| PAR-AND | 0.0 | 1.0 | 0.1 (−) | 1.0 |
| SER-OR | 0.0 | 0.0 | 0.8 (混合) | 1.0 |
| SER-AND | 0.1 | 1.0 | 0.3 | 1.0 |
| COA | 1.0 | 1.0 | 1.0 (+) | 1.0 |

PAR-OR、PAR-AND、COA 完全符合 Townsend & Nozawa 的簽名；SER-AND 的負葉抓到、正葉在 250 試
下抓不到（預期內）。SER-OR 的 MIC 用 ART 檢定有 80% 假陽性——這是 R `sft::mic.test` 同一個
方法在混合分布下的性質，不是移植錯誤，但 Phase 3 要用 R 跑同一份資料確認。

---

## 6. 風險與限制（會改變決策的）

**R1（設計層，最重要）Psi 校準正確率，SFT 需要 RT 分離。**
§5.3 是硬證據。三個處理方式，由研究者選：
(a) 把正確率目標當成 DFP 的**設計輸入**，先用 Phase 4 的模擬器對自己的作業參數做檢定力前測，
再定 p_H / p_L 與 n/cell；(b) 混合：Psi 負責出題與初估，校準資料同時記 RT，事後用
`adaptivesft.fit_lnrm` 擬合 LNRM，用 drift 單位交叉檢核（兩套都已存在）；(c) 放棄 Psi，
回到 LNRM 路（`adaptivesft` 套件已能跑，但沒有線上適應出題，仍是 MOC 設計）。
本提案假設 (a)+(b)。

**R2（參數層）刺激範圍要寬到端點真的接近 99%。** 否則 β 上限太小、估計釘死（§5.2）。
GRTv3_ada 的顏色軸用 ±24 ΔE00、聲音軸 ±18 dB 就是為了這個。SFT 的每一維都要先用 10 分鐘
的粗略 MOC 或既有文獻確認端點正確率，再設範圍。或改寫 `AGRT.py:295` 讓 β 上限可手動指定
（1 行，但偏離 Glavan 的原碼）。

**R3 `.99` 在尾端。** `x_.99 = α + 2.33·β`，α、β 的誤差被 2.33 倍放大，且常落在範圍外（§5.2）。
建議 H 用 `.95`，或直接用範圍端點當 H（GRTv3_ada 的 clip 做法，`GRTv3_ada.py:2294-2295`）。

**R4 SIC 移植只驗了簽名，沒有跟 R 逐位元比對。** R 不在此容器；`setup_r.sh` 5 分鐘可裝。
Phase 3 閘門就是這件事。

**R5 校準作業與主作業要同一種作業。** GRTv3_ada 的教訓（`CLAUDE.md`「全部走主實驗的完整作業」）
同樣適用：SFT 校準要用單維 2AFC 速度作業，跟 DFP 主實驗的反應方式相同，估出來的才是
DFP 情境下的 α、β。

**R6 DDM 模擬是刻意的模型不匹配。** Houpt 也是用 DDM 產生資料、用 LNRM/Psi 校準，本提案沿用。
`adaptivesft/simulate.py:3-6` 的自回復是另一層檢查，兩者互補。

**R7 本容器沒有 PsychoPy。** Phase 5 的實驗腳本只能 `py_compile`，不能跑；線上校準要在
實驗機器上試。

---

## 7. 執行規劃

```
   Phase 0  決策 ─┐
                  ▼
   Phase 1  psi_core ──► Phase 2  ddm_sim ──► Phase 3  sft_sic ──► Phase 4  模擬 + 檢定力前測
                                                  │ 閘門：R 逐位元比對        │ 閘門：檢定力 ≥ .8
                                                  ▼                          ▼ 不過 → 回 Phase 0
   Phase 5  實驗腳本（PsychoPy）◄─────────────────────────────────────────────┘
   Phase 6  分析 CLI            Phase 7  文件與 repo 收尾
   ─────────────────────────────────────────────────────────────────────────────
   估時   0.5   1   1   1.5   1   1–2   0.5   0.5   ＝ 7–8 工作天（不含 Phase 0 等決策）
```

### Phase 0 — 決策（0.5 天，研究者）

| 要決定 | 選項 | 影響 |
|---|---|---|
| 兩個維度是什麼、物理單位、刺激範圍 | — | R2：範圍端點要 ≈ 99% |
| p_H / p_L | 建議 .95 / .75 起跑，Phase 4 前測後定案 | R1、R3 |
| H/L 單側還是雙側 | 單側（同 `psi_color_ddm`）或 α 兩側（同 AGRT） | 反解公式；DFP 是否要 4 個刺激值 |
| 每維校準試次 | 144（Glavan）或 72（GRTv3_ada） | §5.1：144 試 β 的 SD 約 2.2 |
| DFP 每格試次 | Phase 4 決定 | 檢定力 |
| 程式碼放哪個 repo | 建議 adaptiveSFT/`python/`；`Visual_AudioWM/adaptivesft` 之後合併 | 維護 |
| 是否做 LNRM 交叉檢核 | 建議做（現成） | Phase 6 多 0.5 天 |

### Phase 1 — `psi_core.py`（1 天）

- 從 `AGRT.py:75-184` 抄出 `agrtPsiObject`，改名、去掉 `AGRT.py:64-67` 的 PsychoPy import；
  `AGRTHandler` 留在原檔給真人實驗用（它繼承 `StairHandler`）。
- 加 `make_psi(range, steps, lapse, beta_max=None)`：預設照 `AGRT.py:295-298` 推 β 上限，
  可覆寫（R2）。
- 加 `salience_levels(alpha, beta, delta, p_list)`（§2.3）。
- lapse：單維用 λ 本身，不做 `1 − √(1 − λ)`。
- **驗收**：累積常態受試者 144 試 × 20 次，α 誤差 ≤ 2.0、β 誤差 ≤ 2.5（§5.1 的數字）。
  PoC 的 `psi_sft_poc.py:14-51` 可直接搬。

### Phase 2 — `ddm_sim.py`（1 天）

- `simdiffT`（拒絕抽樣，含 `max.iter`、`eps`）、`dfp_ddm`（COA / PAR-OR / PAR-AND / SER-OR /
  SER-AND，`pmix`）、`moc_ddm`。PoC 已有前兩個。
- **驗收**：(1) 反應機率對 `1/(1+exp(−a·drift))`（`simdiffT.r:6`）在 5000 試內 ±.02；
  (2) `sdv=0` 時平均 RT 對 DDM 解析式 `ter + (a/2v)·tanh(a·v/2)`（單界線、對稱起點）±2%。
  用 numba 加速可選，目前純 Python 250 試 < 0.1 s，不必。

### Phase 3 — `sft_sic.py`（1.5 天）

- `sic`、`sic_test`（D⁺、D⁻，`p = exp(−2·N·D²)`，`N` 為四格樣本數的調和平均，`sic.R:142-143`）、
  `si_dominance`（8 個單尾 KS）、`mic_test`（ART 與 ANOVA 兩種）、`sic_group` 的決策表
  （`sic.R:75-105`：哪些 p 值拒絕哪些模型 → 預測架構）。
- **閘門（必過）**：跑 `setup_r.sh` 裝 R + sft，用一份固定種子的 DFP 資料同時餵 R 與 Python：
  SIC 函數值與 D± 差 ≤ 1e−12，KS p 值差 ≤ 1e−10，ART 的 F 值差 ≤ 1e−8。
  差異來源預期只有 `scipy.stats.ks_2samp` 與 R `ks.test` 的漸近 p 值公式，需逐條確認。

### Phase 4 — `simulate_psi_sft.py` 與檢定力前測（1 天）

- 重寫 `psi Simulation_26MAR2019.R:473-590` 的整場迴圈：n 個 DDM 受試者 → 每維 Psi 校準 →
  反解 H/L → DFP 五架構 → `sic_group` → 表格。輸出等同 `ParallelOR_Psi_Simulation_*.csv` 的欄位。
- 用研究者 Phase 0 給的 DDM 參數（或從前測資料粗估的參數）跑檢定力表：
  `p_H/p_L × n/cell × 架構 → P(判對架構)`。
- **閘門**：目標架構在選定的 n/cell 下判對率 ≥ .8。不過 → 回 Phase 0 調 p_H/p_L、範圍或 n/cell。
  §5.3 已示範這個表長什麼樣、以及原作者參數下它會全部不過。

### Phase 5 — 實驗腳本（1–2 天，需實驗機器）

- 以 `GRTv3_ada.py` 為骨架：`phase` 分派（`:1173-1199`）、End Routine 餵 Psi（`:2272-2277`）、
  校準結束反解並覆寫刺激值（`:2291-2296`）、存後驗（`agrt.savePosterior`）。
- 換掉的：試次結構（單維 2AFC 速度作業，不是四角落 WM）、反解（`salience_levels`）、
  主實驗（DFP 2×2 隨機化，記 RT、正確、`Channel1/Channel2` 水準，欄位對齊 `sft::sicGroup`
  要的 `Subject / Condition / Correct / Channel1 / Channel2 / RT`）。
- 本容器只能 `py_compile`；線上跑要在有 PsychoPy 的機器。

### Phase 6 — 分析 CLI（0.5 天）

- `python -m sft_analysis data/S01.csv` → 每位受試者的 SIC 圖、D±、MIC、dominance、預測架構。
- 選配：對校準區塊的 (intensity, rt, correct) 跑 `adaptivesft.fit_lnrm(link="ogival")`，
  印出 Psi 反解值在 LNRM 下的隱含正確率與 drift 差（R1 的 (b)）。

### Phase 7 — 文件與收尾（0.5 天）

- README：兩條路的關係、哪些 R/Stan 檔留作對照、`issue.md` 各項的最終狀態（§3 的表）。
- 決定 `Visual_AudioWM/adaptivesft` 與本 repo 的關係（合併或互相引用）。

---

## 8. 不做的事

- 不裝 rstan / cmdstan / pystan，不改 `lnrm2.stan` 的五行語法（S1），不修 R6 的八個 bug。
- 不去找 `lnrm2a.stan`：主線用不到；交叉檢核用 `adaptivesft` 重建的 ogival 已夠。
- 不做 `sft::capacity` 系列：adaptiveSFT 原流程沒用到。

---

## 9. 附：本文件引用的行號（寫入前已比對）

| 檔案 | 行 | 內容 |
|---|---|---|
| `psiSimulation_functions.R` | 3 | `pm.function` |
| 同上 | 15-17 | x / a / b 範圍 |
| 同上 | 96-99, 104 | DDM 參數、強度縮放 |
| 同上 | 171 | `inv.pm.function` |
| 同上 | 183-184 | `.99` / `.90` 反解 |
| `adaptiveSFT_functions.R` | 115-150 | `dfp_ddm` |
| 同上 | 153-165 | `moc_ddm` |
| 同上 | 203-281 | `find_salience_polynomial` |
| `psi Simulation_26MAR2019.R` | 117 | `DDM.pCorrect` 公式 |
| 同上 | 473-476 | `nParticipants`、`sTrials`、`nDFP` |
| 同上 | 507, 510 | 兩個遺失的 csv |
| `simulateLNRM_ogival.R` | 24-26 | `l_targ`、`h_targ`、`L` |
| `Visual_AudioWM/AGRT.py` | 64-67 | PsychoPy import |
| 同上 | 75-184 | `agrtPsiObject` |
| 同上 | 133 | 反應機率模型 |
| 同上 | 163-171 | `estimateThreshold` |
| 同上 | 288, 295-298 | 邊際 lapse、β 範圍 |
| `Visual_AudioWM/GRTv3_ada.py` | 494-496 | `N_ADAPT_COL/SND` |
| 同上 | 907 | β 撐爆的註解 |
| 同上 | 1173-1199 | phase 分派 |
| 同上 | 2272-2296 | 餵 Psi、反解、覆寫 |
| `Visual_AudioWM/adaptivesft/models.py` | 33-36, 83 | `sigma` 命名、`fit_lnrm` |
| `Visual_AudioWM/adaptivesft/salience.py` | 158 | `find_salience` |
| `sft/R/sic.R`（CRAN 鏡像） | 75-105, 132-248 | 決策表、`sic` 系列 |
| `diffIRT/R/simdiffT.r`（CRAN 鏡像） | 1-33, 6 | 拒絕抽樣、反應機率 |
