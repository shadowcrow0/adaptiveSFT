# 企劃書：讓 adaptiveSFT 的 R / Stan 程式碼在現行環境跑起來

日期：2026-09-24
狀態：提案，尚未動工。**本文件不修改任何程式碼。**

---

## 0. 一句話結論

現有 R 程式碼在 R ≥ 4.2 上**一呼叫就報錯**，Stan 檔在 Stan ≥ 2.33 上**編譯不過**，
而且有三個被引用的 `.stan` 檔**不在版本庫裡**。
建議走「最小修補」方案：**9 個步驟，估 2–4 個工作天**，
其中最大的不確定性是那三個遺失檔案要去哪裡找。

---

## 1. 現況盤點（每一條都有證據）

盤點方式：在乾淨容器裝 R 4.3.3，`parse()` 每個 `.R` 檔，再 `source()` 執行；
Stan 部分對照語言版本紀錄。

| # | 問題 | 位置 | 嚴重度 | 怎麼確認的 |
|---|---|---|---|---|
| P1 | `if()` 作用在整個向量上，R ≥ 4.2 直接 error | `adaptiveSFT_functions.R:228` | **高**：`find_salience_polynomial()` 完全不能用 | 實際呼叫，得到 `the condition has length > 1` |
| P2 | Stan 舊陣列語法 `real x[N]`，Stan 2.33 起移除 | `lnrm2.stan:3,4,6,9,20` | **高**：新版編不過 | 語言版本紀錄；未在本機編譯（沒裝 rstan） |
| P3 | 三個 `.stan` 檔被引用但不存在 | `lnrm0.stan` ← `simulateLNRM_ogival.R:157`；`lnrm1.stan` ← `adaptiveSFT_functions.R:210`；`lnrm2a.stan` ← `adaptiveSFT_functions.R:185`、`simulateLNRM_ogival.R:62`、`adaptive_sft2.py:167` | **高**：ogival 主路徑整條斷掉 | `ls` |
| P4 | `mean(..., na.rm=TRUE)` 靜默丟掉無解的抽樣，無任何提示 | `adaptiveSFT_functions.R:279` | 中：結果可能有偏但看不出來 | 讀碼 |
| P5 | `varZ` 在 Stan 是標準差、R 端 `dlognormalrace()` 收 `sigmasq` 再開根號，兩邊尺度不一致 | `lnrm2.stan:38` vs `adaptiveSFT_functions.R:61-62` | 中：跨語言傳後驗會差一個平方 | 讀碼 |
| P6 | Python 驅動檔用 `^` 當平方（其實是 XOR）、`allchannels` 未定義、`pystan` 2 舊 API | `adaptive_sft2.py:12, 27, 2` | 低：這條路目前沒人用 | 讀碼 |

**沒有問題的部分**：五個 `.R` 檔在 R 4.3.3 下語法解析全部通過；
`&&` / `||` 沒有用在向量上；`simulateLNRM_ogival.R:269, 292` 的 `if(sum(...)>0)` 是純量，安全。

**無法在此確認的部分**：`rstan`、`sft`、`diffIRT` 目前在 CRAN 的狀態
（本機連不到 CRAN）。印象中 `diffIRT` 曾被 CRAN 下架，**需在目標機器確認**。

---

## 2. 三個方案

```
   方案 A  最小修補                    方案 B  A + 換 cmdstanr           方案 C  全面改 Python
   ─────────────────────────         ─────────────────────────       ─────────────────────────
   改 1 行 R + 5 行 Stan               同 A，再把 rstan 換成            用 lnrm2_pymc.py 當引擎，
   找回 3 個 .stan                     cmdstanr（Stan 版本可控）        R 只剩畫圖
   ~2-4 天                            ~3-5 天                          ~1-2 週
   風險：rstan 編譯環境                 風險：多一個工具鏈要裝            風險：lnrm2a 等模型要重寫，
                                                                       且反解邏輯要搬家
```

| | A | B | C |
|---|---|---|---|
| 動到的檔案 | 2 個 | 2 個 + 呼叫方式 | 幾乎全部 |
| 保留原作者的 R 結構 | 是 | 是 | 否 |
| Stan 版本鎖得住 | 靠 rstan 版本 | 是（cmdstanr 可指定） | 不適用 |
| 需要重寫模型 | 否 | 否 | 是（ogival 版） |
| 之前你的立場 | 符合「R 不要改太多」 | 同左 | 你已說不要把反解搬到 Python |

**建議：先做 A。** B 是 A 完成後的可選升級，C 目前不符合你的方向。

---

## 3. 方案 A 的步驟

```
   步驟 1   鎖定環境      →  2   找回三個 .stan   →  3   改 Stan 語法
                                      |
                                      v
   步驟 4   修 :228 的 if  →  5   處理 na.rm      →  6   釐清 varZ 尺度
                                      |
                                      v
   步驟 7   加回歸測試     →  8   全流程跑一次     →  9   記錄版本
```

| 步驟 | 做什麼 | 驗收 | 估時 | 需要誰決定 |
|---|---|---|---|---|
| 1 | 在目標機器記錄 `R.version`、`rstan::stan_version()`、三個套件版本；確認 `diffIRT` 裝得起來 | 一份版本清單 | 0.5 天 | — |
| 2 | 找 `lnrm0.stan`、`lnrm1.stan`、`lnrm2a.stan`：問原作者 / 翻舊硬碟 / 上游 `jhoupt/adaptiveSFT` 的歷史。找不到就得從 `getPr_ogival()` 反推重寫 `lnrm2a` | 三個檔案就位，或決定放棄 ogival 路徑 | **0.5 天 ～ 2 天**（最大變數） | **老闆** |
| 3 | `lnrm2.stan:3,4,6,9,20` 改成 `array[N] real intensity;` 等新語法；找回的三個檔比照辦理 | `rstan::stan_model()` 編譯通過 | 0.5 天 | — |
| 4 | `adaptiveSFT_functions.R:228` 的 `if (post.diff$alpha2 < 0)` 改成明確的向量語意 | 呼叫不再報錯 | 0.5 天 | **老闆**：要哪一種語意（見 §6） |
| 5 | `:279` 的 `na.rm=TRUE` 改成至少回報被丟掉的比例 | 輸出多一個「NaN 比例」欄位 | 0.5 天 | 老闆：要不要改 |
| 6 | 決定 `varZ` 到底是標準差還是變異數，統一 Stan 與 R | 兩邊用同一尺度，有註解說明 | 0.5 天 | **老闆** |
| 7 | 用模擬資料寫一個回歸測試：已知真值 → 擬合 → 反解 → 比對 | 一個可重跑的 `test_*.R` | 1 天 | — |
| 8 | 從 `dataframe2stan()` 到 `find_salience_polynomial()` 跑通一次 | 拿到 high / low 兩個數 | 0.5 天 | — |
| 9 | 把步驟 1 的版本清單和步驟 7 的測試結果寫進 README | README 更新 | 0.5 天 | — |

**合計：約 5 天（含最壞情況的步驟 2）；若三個 .stan 檔很快找到，約 2–3 天。**
估時假設執行者熟 R、能自己裝 rstan。

---

## 4. 風險

```
   風險                           機率    影響    對策
   ──────────────────────────────────────────────────────────────────
   三個 .stan 檔找不到              中      高     先問上游 jhoupt；找不到就從 getPr_ogival() 重寫
   rstan 在目標機器編譯不過         中      中     這是走方案 B 的時機
   diffIRT 已不在 CRAN              低-中   中     用 remotes 從 GitHub / CRAN archive 裝
   :228 改法改變了原本的估計結果    中      高     步驟 7 的回歸測試要在改之前先跑一次留底
   varZ 尺度改錯方向                低      高     先用模擬資料驗證哪個尺度能回收真值
```

---

## 5. 驗收標準

1. R 4.3 下 `source("adaptiveSFT_functions.R")` 無 error。
2. `rstan::stan_model("lnrm2.stan")` 在現行 rstan 編譯通過。
3. 模擬資料（已知 `mu, alpha, alpha2, varZ, psi`）擬合後，`find_salience_polynomial()` 回傳的
   high / low 在 N = 2000 時與真值誤差 < 5%。
   （參考：用 PyMC 引擎 + 同一條反解公式，N=2000 時誤差約 3%，N=8000 時約 3%。）
4. 每一步的版本號有記錄。

---

## 6. 需要老闆決定的事

| 決定 | 選項 | 影響哪一步 |
|---|---|---|
| `:228` 的 `if` 要哪種語意 | (a) `all(alpha2 < 0)` 全部才算；(b) 逐抽樣判斷，不符的給 NA；(c) 用 `mean(alpha2) < 0`；(d) 其他 | 步驟 4 |
| 三個 `.stan` 檔的下落 | 找得到 / 找不到就重寫 / 放棄 ogival 路徑 | 步驟 2 |
| `varZ` 的尺度 | 標準差（照 Stan 現況） / 變異數（照 R 命名） | 步驟 6 |
| `na.rm` 丟抽樣要不要改 | 維持 / 加警告 / 改成報錯 | 步驟 5 |
| 要不要順便換 cmdstanr | 是（方案 B）/ 否 | 步驟 1、3 |

---

## 附錄：本次盤點的原始輸出摘要

```
   parse()  R 4.3.3          5/5 檔案 OK
   呼叫 find_salience_polynomial()   → Error: the condition has length > 1
   Stan 舊語法                lnrm2.stan 第 3, 4, 6, 9, 20 行
   遺失檔案                   lnrm0.stan, lnrm1.stan, lnrm2a.stan
   CRAN 連線                  本機不通，套件狀態未確認
```
