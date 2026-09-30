# scripts/ —— 兩支 R 模擬腳本的 Python 版（plan_python_port.md P6）

| 腳本 | 對應 | 段落（`section`） |
|---|---|---|
| `simulate_lnrm_ogival.py` | `simulateLNRM_ogival.R` | `salience` `convergence` `ppc` `single-dfp` `full-experiment` `all` |
| `simulate_psi.py` | `psi Simulation_26MAR2019.R` | `convergence` `dfp` `full-experiment` `all` |
| `_common.py` | — | 共用：命令列、a 慣例迴圈、survivor + SIC 圖、csv |
| `demo_parity.py` | — | **英文 parity demo**：R 原碼算出的數值（`tests/data/*_r_oracle.json`）vs Python，加上兩條路、兩種 `a` 慣例之間的換算表。不需要 R；oracle 由 `tests/data/make_*_oracle.R` 產生一次 |

兩支都有 `--a-convention {separation,threshold,both}`（預設 `both`），對應 `decisions_for_author.md`
Decision A：`separation` 是 diffIRT 的定義（repo 字面），`threshold` 是 `psi Simulation_26MAR2019.R:117`
公式所隱含的定義（`simdiffT` 收到 `2a`）。每段都把兩種慣例的結果並排印出並存成 csv，
決定就從數字上選。

```bash
PY=venv/bin/python
$PY scripts/simulate_psi.py all --quick                 # 約 1 分鐘，只驗流程
$PY scripts/simulate_lnrm_ogival.py all --quick          # 約 5 分鐘
$PY scripts/simulate_psi.py dfp --trials 300             # R 的設定
$PY scripts/simulate_lnrm_ogival.py salience single-dfp  # 兩次 ogival 擬合（各約 30 s）
$PY scripts/simulate_lnrm_ogival.py convergence          # 8 個 N（log 間距）× 2 慣例，joblib 4 核
$PY scripts/simulate_lnrm_ogival.py convergence --full   # R 原設定 N = 1…300，很慢
```

輸出到 `output/`（gitignored）：每段一個子目錄，png + csv。

與 R 原腳本刻意不同的地方：

- `ppc`：`varZ` 當 SD 傳給賽跑 CDF（R `:204/:213` 把它塞進 `sigmasq` 多開一次根號）；CDF 用
  `plognormalrace_curve`（網格累積積分），不是逐點 `integrate`。
- `convergence`：預設 N 只取 8 個 log 間距的點（plan D8），`--full` 才 1…300；`loess` 換 `lowess`。
- `full-experiment`：每位受試者的 (a, v, ter, sdv) 照 `:496-513` 抽；R 的 `sft.allx`（用群體 H/L 的對照組）
  沒做。
- `simulate_psi.py`：`Est.Trial.Psi.Color` 與 `.Orientation` 合成一個 `run_psi(dim, …)`；
  R 讀進來的 `Psi_Simulation_SFTresults.csv` / `PsiDDM_Simulation_Pars.csv`（遺失）改成每位受試者現跑 Psi；
  「真」α/β 對每種慣例各算一條。
- 兩支的 H/L 若落在刺激範圍外只警告、不裁切（R 2019 版是手動改成範圍上限）。
