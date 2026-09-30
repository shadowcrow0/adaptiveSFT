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
| `plan_grtv3ada_psi_python.md`、`plan_A_implementation.md`、`plan_r_modernization.md`、`bug.md`、`log.md`、`lnrm2_*.md` | 較早的評估與紀錄（有日期，保留） |
| `model_lnrm2.py`、`model_lnrm2a.py`、`lnrm2_pymc.py` | 套件之前的獨立 PyMC 版，留作對照 |
| `poc/` | 概念驗證腳本（含一份抄自 AGRT.py 的 Psi，非正式做法） |

## 還沒有的

- `lnrm0 / lnrm1 / lnrm2a.stan`、`post95.Rdata`、兩個輸入 csv 不在 repo 裡；Python 的對應是重建，不是原檔。
- `decisions_for_author.md` 的兩個決定未定；程式把兩種讀法都做成參數，預設是 repo 字面。
- LNRM 對 Stan 的比對需要有 rstan 的機器（`docs/stan_comparison_redhat.md`）；oracle 進 repo 後 `pytest` 會自動比。
