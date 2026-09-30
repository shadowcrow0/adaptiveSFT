# adaptivesft —— adaptiveSFT 的 Python 版

以本 repo 的設定為準：Stan 字面參數化（`z = μ ∓ d`）、`varZ` 命名、`h_targ / l_targ` 是累積器分離
（漂移差單位）、ogival 的 `L = 10`、`model_lnrm2.py` 的 numba 積木與 DEMetropolisZ。
不需要 R、Stan、PsychoPy。

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

```bash
pip install -e ".[dev]"      # 或 pip install -r requirements-dev.txt（釘住的版本）
pytest                       # 約 3 分鐘；PyMC 的擬合測試用小樣本；CI 在 .github/workflows/tests.yml
```

## issue.md 的每一項在這裡怎麼定案

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

## 驗證

| 測試 | 對什麼 |
|---|---|
| `test_race.py` | numba 逐題 logp vs scipy 封閉式（1e−10）；密度總積分 = 1；`plognormalrace` vs 蒙地卡羅 |
| `test_models.py` | Op = numba；四種 link 小樣本回收；`fix_params` |
| `test_salience.py` | 反解回代；五種 `alpha2_rule`；ogival 反解 = R :194-195 公式；正確率介面 |
| `test_ddm.py` | 反應機率 = `simdiffT.r:6`；`sv = 0` 平均 RT = 解析式 |
| `test_sic.py` | KS 統計量 = scipy；五種架構簽名；`sicGroup` 決策表；**與 R `sft::sic` 逐位元比對**（`tests/data/sic_r_oracle.json`，由 `make_sic_oracle.R` 產生一次） |
| `test_psi.py` | 網格 = R 的 `seq()`；累積常態受試者回復；β 在網格外被釘住；`salience_levels` = `inv.pm.function` |
| `test_lnrm_vs_stan.py` | **LNRM 對 Stan**：`tests/data/lnrm_stan_oracle.json`（在有 rstan 的機器上由 `make_lnrm_oracle.R` 產生，見 `docs/stan_comparison_redhat.md`）vs `fit_lnrm(link="quadratic")`；沒有 oracle 時 skip |
| `test_parity_demo.py` | **與 R 原碼逐位元 / 分布對照**：`sft::sic`（1e−12）、`diffIRT::simdiffT`（20000 試分布）、`psiSimulation_functions.R` 的 Psi 迴圈逐試相同（1e−10）；以及正確率 ↔ 分離、兩種 `a` 慣例的換算。`scripts/demo_parity.py` 印成表（英文） |

## 還沒做（plan_python_port.md P6–P7）

`simulateLNRM_ogival.R` 與 `psi Simulation_26MAR2019.R` 兩支腳本的迴圈與畫圖；`scripts/` 目錄尚未建立。
