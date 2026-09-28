# P6 結果：兩支模擬腳本在兩種 `a` 慣例下的並排輸出

日期：2026-09-28。指令、輸出檔、完整 log 都在 `results/p6/`（`psi_full.log`、`lnrm_full.log`）。
設定照 R 原腳本：Psi 50 次 × 300 試（R 用 629 次）、DFP 100 試/格、10 位受試者；
LNRM 8 鏈 × 3000 + 3000（DEMetropolisZ）、10 層 × 100 試、`h_targ = 8.0`、`l_targ = 1.3`、`L = 10`。
`a` 慣例見 `decisions_for_author.md` Decision A：

```
   separation   simdiffT 收到 a        P(correct) = 1/(1+exp(−a·drift))       ← diffIRT 的定義（repo 字面）
   threshold    simdiffT 收到 2a       P(correct) = 1/(1+exp(−2·a·drift))     ← psi Simulation_26MAR2019.R:117
```

執行時間（4 核）：`simulate_psi.py all` 29 分、`simulate_lnrm_ogival.py all` 15 分。

---

## 1. Psi 路（`simulate_psi.py`）

### 1.1 收斂（方位，50 × 300 試）

| 慣例 | 「真」α | 「真」β | 範圍上限 P(correct) | 300 試後 α̂ | 300 試後 β̂ | β 網格上限 |
|---|---|---|---|---|---|---|
| separation | 63.03 | **18.93** | 0.911 | 63.13 [61.5, 65.5] | **9.28** [9.21, 9.33] | ≈ 9.3 |
| threshold | 63.01 | **9.65** | 0.990 | 62.99 [61.8, 64.7] | **9.08** [8.80, 9.26] | ≈ 9.3 |

```
   separation：β̂ 走到 9.3 就停，5–95% 區間只剩 0.1 寬 ── 撞到 AGRT.py:295 由範圍推出的網格上限，
               真值 18.9 根本不在網格上（results/p6/psi_convergence/Psi_convergence_separation.png 右圖：
               紅虛線在 18.9，藍線貼在 9.3）。
   threshold ：β̂ 9.08 → 真值 9.65，α̂ 也對。這才是 Psi 能校準的世界。
```

α 兩種慣例都收斂到 63（決策界線不受 `a` 影響），只有 β（斜率）差一倍——跟 Decision A 的診斷一致。

### 1.2 校準 H(.99) / L(.90) 並回代 DDM

| 慣例 | 維度 | β̂ | β 真 | H | H 真 | H 回代 acc | L | L 真 | L 回代 acc | 範圍上限 |
|---|---|---|---|---|---|---|---|---|---|---|
| separation | 顏色 | 21.4 | 31.2 | 56.1 | 86.3 | 0.931 | 28.9 | 46.7 | 0.767 | 50 |
| separation | 方位 | 9.3 | 18.9 | 87.5 | 111.7 | 0.887 | 75.7 | 87.7 | 0.758 | 90 |
| threshold | 顏色 | 16.2 | 15.8 | 47.6 | 46.6 | 0.984 | 27.0 | 26.6 | 0.881 | 50 |
| threshold | 方位 | 9.2 | 9.7 | 86.9 | 87.8 | 0.977 | 75.2 | 75.6 | 0.868 | 90 |

- separation：真正的 H 在 86（顏色）/ 112（方位），範圍外；Psi 因 β 被釘住而給出範圍內的假 H，
  回代只有 0.93 / 0.89，L 只有 0.76。這正是 `psi Simulation_25JUNE2018.R:174` 的 101.6 那件事。
- threshold：H 真值 46.6 / 87.8 剛好在範圍上限之內（2019 版硬寫 50 / 90 的來源），估計值對得上，
  回代 0.98 / 0.98。10 位受試者中 6 位的顏色 H 估到 50–54，略出界，是估計噪音。

### 1.3 五種架構各一場 DFP（100 試/格，H/L 來自 1.2）

| 慣例 | COA | PAR-OR | PAR-AND | SER-OR | SER-AND |
|---|---|---|---|---|---|
| separation | D⁺ p=.012, MIC p<.001 → ParallelOR | 全不顯著 | 全不顯著 | 全不顯著 | 全不顯著 |
| threshold | 全不顯著 | MIC p=.031 → NA | 全不顯著 | 全不顯著 | 全不顯著 |

### 1.4 整場實驗（10 位 × 4 種架構，欄位同 `ParallelOR_Psi_Simulation_SFTresults.csv`）

| 慣例 | 架構 | D⁺ 顯著 | D⁻ 顯著 | MIC 顯著 | 判成正確架構 |
|---|---|---|---|---|---|
| separation | PAR-OR | 0/10 | 0/10 | 0/10 | 0/10 |
| separation | PAR-AND | 0/10 | 1/10 | 1/10 | 1/10 |
| separation | SER-OR | 1/10 | 1/10 | 1/10 | 8/10（＝什麼都不顯著的預設） |
| separation | SER-AND | 1/10 | 2/10 | 1/10 | 0/10 |
| threshold | PAR-OR | 1/10 | 0/10 | 3/10 | 1/10 |
| threshold | PAR-AND | 0/10 | 1/10 | 0/10 | 1/10 |
| threshold | SER-OR | 0/10 | 1/10 | 1/10 | 9/10（同上） |
| threshold | SER-AND | 0/10 | 1/10 | 0/10 | 0/10 |

**結論**：不管哪種慣例，`.99 / .90` 這組正確率目標配 a=1.45、v=1.6 的 DDM，給不出 SIC 看得見的
RT 分離（`plan_grtv3ada_psi_python.md` §5.3 的結論在完整規模下重現）。慣例決定的是 Psi
估不估得到真值（1.1、1.2），不是檢定力。

---

## 2. LNRM 路（`simulate_lnrm_ogival.py`）

### 2.1 反解 H / L（方位，`h_targ = 8.0`、`l_targ = 1.3`）

| 慣例 | H (scaled) | H 物理值 | H 回代 acc | L (scaled) | L 回代 acc | slope | midpoint | varZ | 丟掉的 draw |
|---|---|---|---|---|---|---|---|---|---|
| separation | 1.786 | 111° | 1.000 | 0.578 | 0.970 | 1.98 | 0.65 | 0.95 | 12.5% |
| threshold | 1.552 | 105° | 1.000 | 0.476 | 0.997 | 2.26 | 0.49 | 0.94 | 12.5% |

- 校準的 scaled 範圍是 [−0.67, 1.00]（物理 45°–90°）。**兩種慣例下 H 都在範圍外**（111° / 105°），
  L 的回代正確率 0.97 / 0.997 也遠高於 Psi 路的 .90。這是 Decision B 的問題（`h_targ = 8.0`
  在 L = 10 的尺度上是「幾乎滿格」），跟 `a` 慣例無關。
- 「丟掉 12.5%」= 8 鏈中剛好 1 鏈卡在 slope ≤ 0 的模態，反解無解。DEMetropolisZ 的已知弱點
  （`issue.md` M8），R-hat 警告有出現；R 的 `mean(na.rm=TRUE)` 會靜默吃掉。

### 2.2 五種架構各一場 DFP（250 試/格，H/L 來自 2.1）

| 慣例 | COA | PAR-OR | PAR-AND | SER-OR | SER-AND |
|---|---|---|---|---|---|
| separation | Coactive ✔ | ParallelOR ✔ | ParallelAND ✔ | NA（MIC p=.007） | Coactive ✘ |
| threshold | Coactive ✔ | ParallelOR ✔ | ParallelAND ✔ | NA（MIC p=.008） | SerialAND ✔ |

分離大到這種程度，SIC 幾乎都判對；SER-OR 的 MIC 用 ART 檢定在混合分布下假陽性
（`plan_grtv3ada_psi_python.md` §5.4 已記）。

### 2.3 參數收斂（取代遺失的 `post95.Rdata`；8 個 N、log 間距）

| N/層 | separation midpoint 5–95% | separation slope 5–95% | threshold midpoint 5–95% | threshold slope 5–95% |
|---|---|---|---|---|
| 3 | [0.81, 1.63] | [1.67, 4.11] | [0.93, 1.51] | [1.98, 4.36] |
| 20 | [1.18, 1.53] | [2.09, 3.21] | [0.89, 1.07] | [2.81, 3.83] |
| 80 | [1.23, 1.42] | [2.45, 3.05] | [1.04, 1.16] | [2.75, 3.33] |
| 300 | [1.22, 1.40] | [2.62, 3.03] | [1.07, 1.23] | [2.83, 3.19] |

圖：`results/p6/convergence/LNRM_parameter-convergence.png`（R :113-135 的形狀，`loess` 換 `lowess`）。

### 2.4 整場實驗（10 位、PAR-AND、100 試/格）

| 慣例 | 判成 ParallelAND | 選擇性影響 Pass |
|---|---|---|
| separation | 9/10（1 位 SerialOR） | 10/10 |
| threshold | 10/10 | 8/10 |

---

## 3. 對兩個決定的意義

```
   Decision A（a 的意義）        Psi 路：threshold 慣例才估得到真值；separation 慣例 β 撞網格上限、H 出界。
                                 LNRM 路：兩種慣例結果形狀相同（H 都出界），慣例不是主因。
   Decision B（targ 的尺度）     h_targ = 8.0 / L = 10 讓 H 落在校準範圍外、L 已有 0.97+ 的正確率；
                                 SIC 判得準是因為分離極大，不是校準得準。
   兩條路的檢定力差異            Psi（.99/.90）0–3/10 顯著；LNRM（8.0/1.3）9–10/10。
                                 同一個 DDM，差別只在 H/L 拉多開 —— 目標值該用哪個尺度、多大，
                                 才是要拍板的核心（decisions_for_author.md A.6 第 3 點、B.6 第 4 點）。
```

## 4. 移植上的發現（本輪修掉的）

- `diffIRT::simdiffT` 的拒絕抽樣在 a·|drift| ≳ 10 時級數收斂極慢、接受率趨近 0（R 會跑很久後
  `stop("Rejection algorithm failed")`）。`adaptivesft/ddm.py` 在 a·|drift| > 10 直接改用 Wiener 過程
  模擬（`tests/test_ddm.py` 對照解析式 P(correct) 與平均 RT）。R 的 `simulateLNRM_ogival.R` 用 a=3、
  v=2 配 COA 架構（drift 相加）一定會碰到這個。
- 後驗預測圖逐點 `integrate`（R :81-111）在 Python 要跑幾分鐘；改成網格累積積分
  `plognormalrace_curve`，一條曲線 1 ms，對 `quad` 誤差 1e−5。
