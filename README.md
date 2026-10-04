# adaptiveSFT

Code for adapting salience levels for the Double Factorial Paradigm (Houpt, 2018–2019; R + Stan),
and its Python port.

```
   原始碼（R / Stan，唯讀，2018–2019）              Python 版（adaptivesft/，2026）
   ─────────────────────────────────────           ─────────────────────────────────────────
   adaptiveSFT_functions.R                          race.py  salience.py  ddm.py
   lnrm2.stan（lnrm0/1/2a.stan 遺失）               models.py（四個 link，NUTS）
   psiSimulation_functions.R                        psi.py（逐行移植，網格照 R）
   psi Simulation_26MAR2019.R                       scripts/simulate_psi.py
   simulateLNRM_ogival.R                            scripts/simulate_lnrm_ogival.py
   sft::sic、diffIRT::simdiffT（CRAN 套件）         sic.py、ddm.py（移植，與 R 逐位元 / 分布比對）
```

## 用

```bash
pip install -e ".[dev]"          # 或 pip install -r requirements-dev.txt（釘住的版本）
pytest                           # 約 3 分鐘，不需要 R / Stan / PsychoPy
python scripts/demo_parity.py    # R 原碼算出的值 vs Python，並排
python scripts/demo_decisions.py # Decision A / B 兩種讀法並排
```

套件說明：`adaptivesft/README.md`。腳本說明：`scripts/README.md`。

## 文件索引

| 檔 | 內容 |
|---|---|
| `issue.md` | 原始碼的每個問題（R1–R6、S1–S3、P1–P5、M1–M9）；末段標各項最終狀態 |
| `decisions_for_author.md` | 兩個只有原作者能定的問題（英文）：`a` 的意義、`lnrm2a` 的 ½L / L / 先驗 |
| `plan_python_port.md` | 整體移植的盤點、規劃、進度 |
| `p6_results.md` | 兩支模擬腳本在兩種 `a` 慣例下的完整結果 |
| `docs/stan_comparison_redhat.md` | 在有 rstan 的機器上做 LNRM 對 Stan 的比對 |
| `docs/hpc_arc_tutorial.md`、`hpc/` | 在 Arc（Slurm）上建環境、送作業、把 oracle 與 demo 結果推回來 |
| `plan_grtv3ada_psi_python.md`、`plan_A_implementation.md`、`plan_r_modernization.md`、`bug.md`、`log.md`、`lnrm2_*.md` | 較早的評估與紀錄（有日期，保留） |
| `model_lnrm2.py`、`model_lnrm2a.py`、`lnrm2_pymc.py` | 套件之前的獨立 PyMC 版，留作對照 |
| `poc/` | 概念驗證腳本（含一份抄自 AGRT.py 的 Psi，非正式做法） |

## 自己學會寫這個模型：`learn/`

五個練習（Normal → 位移對數常態 → 賽跑 likelihood → 完整 lnrm2 → 反解 H/L），每題自己寫、
`python learn/check.py exN` 對答案（ex4 直接對 Stan 的後驗）。規則與進度表在 `learn/README.md`。

## 三年後要改東西，從哪個測試開始

| 模組 | 對應的測試 | 改之前先跑 |
|---|---|---|
| `adaptivesft/race.py`（賽跑 likelihood） | `tests/test_race.py` | 密度積分 = 1、與 R 的 `dlognormalrace` 同值 |
| `adaptivesft/models.py`（PyMC 擬合） | `tests/test_models.py`、`tests/test_lnrm_vs_stan.py` | 後者對 Stan 的後驗（`tests/data/lnrm_stan_oracle.json`） |
| `adaptivesft/salience.py`（反解 H/L） | `tests/test_salience.py` | 四種 `alpha2_rule` 的行為 |
| `adaptivesft/ddm.py`（DDM 模擬） | `tests/test_ddm.py` | 對 R `diffIRT::simdiffT` 的分布 |
| `adaptivesft/sic.py`（SIC / MIC） | `tests/test_sic.py` | 對 R `sft::sic` 逐位元 |
| `adaptivesft/psi.py`（Psi） | `tests/test_psi.py`、`tests/test_parity_demo.py` | 對 R 逐試 1e−13 |
| `adaptivesft/experiment.py`（真人實驗用） | `tests/test_experiment.py` | |

套件版本鎖在 `requirements-lock.txt`（Python 3.11）；三年後 `pip install -e .` 裝到新版 PyMC 跑不起來時，先用 lock 檔重建。

## 還沒有的

- `lnrm0 / lnrm1 / lnrm2a.stan`、`post95.Rdata`、兩個輸入 csv 不在 repo 裡；Python 的對應是重建，不是原檔。
- `decisions_for_author.md`：實驗端（Visual_AudioWM）已定 **B**——用 LNRM 分支、漂移差目標、本 repo 的 ogival 重建；A 只影響 DDM 模擬，維持未定。程式仍把兩種讀法都做成參數。
- LNRM 對 Stan 的比對已完成（Arc 上 cmdstanr 2.40.0 產生 `tests/data/lnrm_stan_oracle.json`）：五個參數的後驗平均差 ≤ 0.01 個後驗 SD，`tests/test_lnrm_vs_stan.py` 常駐。
