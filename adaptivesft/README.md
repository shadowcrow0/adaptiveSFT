# adaptivesft —— adaptiveSFT 的 Python 版

**這是什麼。** 這個資料夾是套件本體：從受試者的反應時間資料，一路算到「該用哪兩個刺激強度」
和「受試者是並行還是序列處理」的全部函式。不需要 R、Stan、PsychoPy。

**一個比喻：裁縫店。** 先量身（擬合模型），再依尺寸裁布（反解出高 / 低顯著度的強度），
接著試穿（用假受試者模擬一場實驗），最後評鑑版型（SIC 判斷處理架構）。每一步都是一個
函式，照順序接起來。

幾個名詞，第一次出現先解釋：

- **擬合（fit）**：把模型套到資料上，找出最合理的參數值。
- **後驗（posterior）**：貝氏模型看完資料後對參數的看法。不是一個數字，是一大堆
  「可能的參數值」，每一筆叫一個 **draw**。
- **先驗（prior）**：還沒看資料之前，對參數的預設看法。
- **NUTS**：PyMC 的取樣器，一種自動走遍可能參數值、產生後驗 draw 的方法。要能算
  **梯度**（參數往哪邊動會讓模型更合理）才能用。
- **DEMetropolisZ**：另一種取樣器，不需要梯度，但比較慢、比較容易卡住。
- **累積器（accumulator）**：LNRM 裡的「計時器」。一條代表答對、一條代表答錯，先到的贏。
- **漂移差 / 累積器分離**：兩條計時器的速度差 `z₂ − z₁`。差越大，答對越快越穩。
- **link**：「刺激強度 → 難度 d」的那條曲線。四種模型只差這條。
- **ogival**：S 形曲線（logistic），四種 link 之一。
- **numba**：把 Python 算式編譯成機器碼加速的工具。
- **反解**：已知模型參數，倒過來算出哪個強度會給你想要的難度。

遇到有爭議的地方，一律以本 repo 的設定為準：

- **Stan 字面參數化（`z = μ ∓ d`）**：答對累積器在 `μ − d`，答錯在 `μ + d`，照 `lnrm2.stan` 的寫法。
- **`varZ` 命名**：Stan 檔裡的 `varZ` 名字沿用，但它是標準差（SD），不是變異數（見下面 S3）。
- **`h_targ / l_targ` 是累積器分離（漂移差單位）**：高 / 低顯著度的目標是 `z₂ − z₁` 要多大，不是正確率。
- **ogival 的 `L = 10`**：S 形曲線的最大分離固定為 10。
- **`model_lnrm2.py` 的 numba 積木與 DEMetropolisZ**：沿用套件之前那個獨立版本的計算核心與取樣器。

流程圖（左邊是 Python 函式，右邊是它對應的原始 R / Stan 碼）：

```
   資料 (rt, correct, intensity)                 make_data()            ← dataframe2stan
         │
         ▼
   fit_lnrm(link=none|linear|quadratic|ogival)   models.py + race.py    ← lnrm0/1/2/2a.stan + rstan
         │  後驗 {mu, alpha, alpha2 | slope, midpoint, varZ, psi}
         ▼
   find_salience(h_targ, l_targ | acc_high, acc_low)   salience.py     ← find_salience_polynomial / _ogival
         │  high / low 的刺激強度（含 dropped 比例、90% 區間）
         ▼
   dfp_ddm(高, 低, 架構, 停止規則)                ddm.py                 ← diffIRT::simdiffT + dfp_ddm
         │  HH / HL / LH / LL 的 rt, correct
         ▼
   sic(HH, HL, LH, LL) → classify()               sic.py                 ← sft::sic / sicGroup

   另一條路：make_psi() → PsiObject.update() → salience_levels()   psi.py  ← psiSimulation_functions.R
```

一步一步讀：

1. `make_data()`：把三欄資料（反應時間 rt、答對與否 correct、刺激強度 intensity）整理成模型要的格式。
2. `fit_lnrm()`：擬合賽跑模型，拿到後驗。`mu` 是整體速度、`alpha / alpha2` 是強度的效果、
   `varZ` 是變異、`psi` 是不算在決策裡的固定延遲（例如按鍵的時間）。ogival 版用 `slope / midpoint`。
3. `find_salience()`：從後驗反解出高 / 低顯著度的強度。`dropped` 是有多少 draw 無解被丟掉。
4. `dfp_ddm()`：用 DDM（漂移擴散模型）當假受試者跑一場 DFP。HH / HL / LH / LL 是四個條件
   （兩個通道各高或低）。
5. `sic()` → `classify()`：算 SIC 曲線，判斷是並行（PAR）、序列（SER）還是共同激發（COA），
   停止規則是 OR 還是 AND。
6. 另一條路 `psi.py`：不擬合 LNRM，改用 Psi 適應程序。每題更新一次，最後直接給高 / 低強度。

## 用法

```python
import numpy as np
from adaptivesft import moc_ddm, fit_lnrm, find_salience, summarize, dfp_ddm, sic, classify

data = moc_ddm(100, a=3, v=2, ter=.1, sdv=.2, intensity_levels=np.linspace(-.4, .6, 10), rng=np.random.default_rng(1))
trace = fit_lnrm(data, link="ogival")                       # L = 10，同 simulateLNRM_ogival.R:26
res = find_salience(trace, h_targ=8.0, l_targ=1.3)          # 同 simulateLNRM_ogival.R:24-25
print(summarize(res))
res_acc = find_salience(trace, acc_high=0.9, acc_low=0.7)   # 或用正確率當目標

high, low = res["high"]["intensity"], res["low"]["intensity"]
cells = {}
for name, (c1, c2) in dict(HH=(high, high), HL=(high, low), LH=(low, high), LL=(low, low)).items():
    rt, cr = dfp_ddm(250, c1 * 2, c2 * 2, 3, .1, .2, "PAR", "OR")
    cells[name] = rt[cr == 1]
print(classify(sic(**cells)))
```

這段在做什麼：先用 `moc_ddm` 造一位假受試者的 100 題資料（10 個強度，定值刺激法）；
擬合 ogival 模型；反解目標分離 8.0 / 1.3 的強度；再用這兩個強度跑四個 DFP 條件，各 250 題，
只留答對的反應時間；最後判架構。`a=3, v=2, ter=.1, sdv=.2` 是 DDM 的四個參數：界線、
漂移率、非決策時間、漂移率的跨試次變異。

```bash
pip install -e ".[dev]"      # 或 pip install -r requirements-dev.txt（釘住的版本）
pytest                       # 約 3 分鐘；PyMC 的擬合測試用小樣本；CI 在 .github/workflows/tests.yml
```

「CI」＝每次把程式推上 GitHub 就自動跑一次測試。

## issue.md 的每一項在這裡怎麼定案

`issue.md` 是原始碼問題清單。下表說每一項在這個套件裡怎麼處理。先解釋表裡會用到的詞：

- **網格（grid）**：Psi 事先列出的一串候選參數值。受試者的真值如果不在這串裡，估計就會
  「被釘在邊緣」。
- **判別式**：解二次方程式時根號裡的那個數。小於 0 就無解。
- **鏈（chain）**：取樣器獨立跑的一趟。通常跑好幾條，互相比對。
- **模態**：後驗裡的一個「山峰」。有兩個山峰時，鏈可能卡在錯的那個。
- **KS**：Kolmogorov–Smirnov 檢定，比兩個分布像不像。
- **Op**：PyMC 裡自訂的計算節點。用 numba 寫的 Op 沒有梯度，所以不能用 NUTS。

| 編號 | 問題 | 定案 | 在哪 |
|---|---|---|---|
| R1 | `if (alpha2 < 0)` 長度 > 1 | `alpha2_rule` 參數，預設 `all_draws`（＝舊 R 成功路徑，數值不變）；另四種語意可選，`first_draw` 重現舊 R 字面行為 | `salience.py` |
| R2 / S1 | rstan / stanc 版本、`real x[N]` 舊語法 | 不呼叫 Stan | — |
| R3 | `permute=` | 不呼叫 rstan | — |
| R5 / S2 | `lnrm0 / lnrm1 / lnrm2a.stan` 遺失 | `link="none" / "linear" / "ogival"` 重建；每題 d 的公式在 `models.py` 檔頭；ogival 的先驗是猜的（TODO-A3） | `models.py` |
| R5 | `post95.Rdata`、輸入 csv 遺失 | 模擬腳本自己產生（scripts/ 尚未寫） | — |
| R6 | 2018 腳本的 bug（`sigmasqx`、`postOpt.diff`、`dp` 未初始化…） | 不移植那些腳本；函式層重寫 | — |
| M8 | Op 沒梯度，只能 DEMetropolisZ（8 鏈偶有 1 鏈卡在 slope ≤ 0） | 預設 `sampler="nuts"`：likelihood 用純 PyTensor 寫一次（與 numba Op 的 logp 差 < 1e−8），有梯度；`sampler="demetropolisz"` 保留 | `models.py` |
| B | `lnrm2a` 的 ½L / L、L 固定或估 | `ogival_offset=0.5|1.0`、`L=10|"estimate"`，反解跟著走 | `models.py`、`salience.py`、`scripts/demo_decisions.py` |
| S3 | `varZ` 是 SD 還是變異數 | **SD**。`dlognormalrace / plognormalrace` 改收 `varZ`，不開根號；R 的 `simulateLNRM_ogival.R:204/213/258/281` 把 SD 塞進變異數位是 R 自己的錯 | `race.py` |
| P1–P5 | `adaptive_sft2.py` | 作廢 | — |
| P4（plan A） | `mean(na.rm=TRUE)` 靜默丟 draw | 回報 `dropped` 比例，> 5% 進 warnings | `salience.py` |
| — | `DDM.pCorrect` 差 2 倍（`26MAR2019.R:117`） | `ddm_p_correct()` 照 `simdiffT.r:6`；測試證明 2 倍那條不是資料真相 | `ddm.py`、`tests/test_ddm.py` |
| — | `.99` 反解落在刺激範圍外（`25JUNE2018.R:174`） | `salience_levels(x_range=…)` 會警告 | `psi.py` |
| — | R 的 β 網格是寫死的常數（顏色 1–50、方位 1–10，`:17` / `:224`）；受試者的 β 在網格外就被釘在邊緣 | 網格照 R 當預設（`GRIDS`），`make_psi(b=…)` 可改；不用 AGRT 的公式 | `psi.py` |
| — | `Est.Trial.Psi.Color` 與 `.Orientation` 逐行重複 | 合成一個 `Psi` 類別，網格由呼叫端給 | `psi.py` |
| — | scipy `ks_2samp` 的單尾漸近 p 與 R 不同 | 手算 `exp(−2·n·D²)`，與 R `ks.test(exact=FALSE)` 同式 | `sic.py` |

幾個值得多講一句的：

- **R1**：原始 R 碼問 `alpha2 < 0`，但 `alpha2` 是幾千筆 draw，新版 R 會報錯。
  這裡把「要怎麼問」做成參數 `alpha2_rule`，預設和舊版 R 跑得通時算出的數字一模一樣。
- **S3**：Stan 檔裡 `varZ` 當 SD 用（`lnrm2.stan:38`）。R 的部分地方把它當變異數再開根號，
  是 R 端自己不一致。這裡統一當 SD。
- **M8**：原本只能用慢的取樣器。現在把 likelihood 用 PyTensor（PyMC 的計算引擎）重寫一遍，
  有梯度了，就能用 NUTS。兩種寫法在同一點算出的 logp 差不到 1e−8。
- **`:17` / `:224`** 指 `psiSimulation_functions.R` 的行號。「不用 AGRT 的公式」是指不用
  `Visual_AudioWM/AGRT.py` 那套算 β 網格的方式。
- **R5 那列「scripts/ 尚未寫」** 是定案當時的狀態。現在 `scripts/` 已經存在，見 `scripts/README.md`。

## 驗證

每個模組都有一支測試檔。「對什麼」欄說它拿什麼當標準答案。

| 測試 | 對什麼 |
|---|---|
| `test_race.py` | numba 逐題 logp vs scipy 封閉式（1e−10）；密度總積分 = 1；`plognormalrace` vs 蒙地卡羅 |
| `test_models.py` | Op = numba；四種 link 小樣本回收；`fix_params` |
| `test_salience.py` | 反解回代；五種 `alpha2_rule`；ogival 反解 = R `adaptiveSFT_functions.R:194-195` 公式；正確率介面 |
| `test_ddm.py` | 反應機率 = `simdiffT.r:6`；`sv = 0` 平均 RT = 解析式 |
| `test_sic.py` | KS 統計量 = scipy；五種架構簽名；`sicGroup` 決策表；**與 R `sft::sic` 逐位元比對**（`tests/data/sic_r_oracle.json`，由 `make_sic_oracle.R` 產生一次） |
| `test_psi.py` | 網格 = R 的 `seq()`；累積常態受試者回復；β 在網格外被釘住；`salience_levels` = `inv.pm.function` |
| `test_lnrm_vs_stan.py` | **LNRM 對 Stan**：`tests/data/lnrm_stan_oracle.json`（在有 rstan 的機器上由 `make_lnrm_oracle.R` 產生，見 `docs/stan_comparison_redhat.md`）vs `fit_lnrm(link="quadratic")`；oracle 已由 Arc 上的 cmdstanr 2.40.0 產生，五個參數的後驗平均差 ≤ 0.01 SD |
| `test_experiment.py` | Psi / LNRM 校準控制器在模擬受試者上跑通；DFP 試次表；單人分析判對 PAR-OR |
| `test_parity_demo.py` | **與 R 原碼逐位元 / 分布對照**：`sft::sic`（1e−12）、`diffIRT::simdiffT`（20000 試分布）、`psiSimulation_functions.R` 的 Psi 迴圈逐試相同（1e−10）；以及正確率 ↔ 分離、兩種 `a` 慣例的換算。`scripts/demo_parity.py` 印成表（英文） |

表裡的詞：「回收（recovery）」＝先用已知參數造假資料，再擬合，看能不能把參數找回來。
「oracle」＝用原版 R / Stan 跑一次、存成檔案的標準答案。「蒙地卡羅」＝用大量隨機模擬逼近
一個數值。「封閉式 / 解析式」＝有公式可以直接算的答案。「回代」＝把反解出的強度放回
模型，確認真的得到目標分離。「1e−10」等＝允許的誤差，小到可以當成零。

最重要的兩列是 `test_lnrm_vs_stan.py` 和 `test_parity_demo.py`：前者證明 Python 的 LNRM 和
Stan 原版擬合同一份資料時答案一樣，後者證明 SIC、DDM、Psi 三塊和 R 原碼逐位元一樣。

## 還沒做（plan_python_port.md P6–P7）

`simulateLNRM_ogival.R` 與 `psi Simulation_26MAR2019.R` 兩支腳本的迴圈與畫圖；`scripts/` 目錄尚未建立。

（註：這段是寫套件當時的狀態。`scripts/simulate_lnrm_ogival.py` 與 `scripts/simulate_psi.py`
現已存在，說明在 `scripts/README.md`。）

## 所以你要做什麼

1. 要校準真人：用 `experiment.py` 的 `LNRMCalibrator` 或 `PsiCalibrator`，流程見該檔檔頭。
   PsychoPy 腳本只負責畫面和按鍵，計算都在這裡。
2. 要分析一位受試者的 DFP 資料：`analyze_participant()`，或直接跑 `scripts/analyze_participant.py`。
3. 要用正確率而不是漂移差當目標：`find_salience(trace, acc_high=…, acc_low=…)`。
4. 改任何東西之前：`pytest`，確認「驗證」那張表全綠。
5. `a` 的意義與 `lnrm2a` 的 ½L / L：程式兩種都做成參數，決定權在研究者，見 `decisions_for_author.md`。
