# scripts/ —— 兩支 R 模擬腳本的 Python 版（plan_python_port.md P6）

**這是什麼。** 這個資料夾是「用電腦模擬受試者，把整套校準 → DFP → SIC 流程從頭跑一遍」
的腳本，另外加上幾支給真人實驗用的工具。

**一個比喻：飛行模擬器。** 真的載客之前，先在模擬器裡飛幾趟。這裡的「假受試者」是
DDM（漂移擴散模型），參數可以自己設。你可以在真人進實驗室之前就知道：校準要幾題才穩、
目標設多少 SIC 判得準、流程哪裡會出錯。

幾個名詞：

- **校準（calibration）**：替每位受試者量出高 / 低顯著度的刺激強度。
- **DFP**：Double Factorial Paradigm，2×2 設計（兩個通道各高或低）的反應時間實驗。
- **SIC**：survivor interaction contrast，從四個條件的反應時間分布算出來的曲線，
  用來判斷兩個通道是並行、序列還是共同激發，停止規則是 OR 還是 AND。
- **架構（architecture）**：上面那五種組合，程式裡寫成 PAR-OR、PAR-AND、SER-OR、SER-AND、COA。
- **survivor 圖**：「到時間 t 還沒反應的比例」畫成曲線。SIC 就是四條 survivor 曲線的加減。
- **段落（section）**：每支腳本分成幾段，可以單獨跑一段。
- **`a` 慣例**：DDM 界線參數 `a` 的兩種讀法，見下面。

| 腳本 | 對應 | 段落（`section`） |
|---|---|---|
| `simulate_lnrm_ogival.py` | `simulateLNRM_ogival.R` | `salience` `convergence` `ppc` `single-dfp` `full-experiment` `all` |
| `simulate_psi.py` | `psi Simulation_26MAR2019.R` | `convergence` `dfp` `full-experiment` `all` |
| `_common.py` | — | 共用：命令列、a 慣例迴圈、survivor + SIC 圖、csv |
| `demo_decisions.py` | — | **英文 Decision A / B demo**：`a` 的兩種讀法、`lnrm2a` 的 ½L / L 與 L 固定 / 估，各自算出的 H / L、正確率、退化與否並排；輸出在 `results/decisions_demo.txt` |
| `power_scan.py` | — | 漂移差目標 × 每格試次 × 五種架構 → SIC 判對率（每人只擬合一次）；`--a --v --ter --sdv` 換成自己作業的 DDM 參數 |
| `analyze_participant.py` | — | 一位受試者的 DFP csv → SIC / MIC / dominance → 預測架構（`--json`、`--plot`） |
| `demo_parity.py` | — | **英文 parity demo**：R 原碼算出的數值（`tests/data/*_r_oracle.json`）vs Python，加上兩條路、兩種 `a` 慣例之間的換算表。不需要 R；oracle 由 `tests/data/make_*_oracle.R` 產生一次 |

段落名稱的意思：`salience`＝擬合一次、反解 H / L；`convergence`＝題數越多估計越穩嗎；
`ppc`＝posterior predictive check，後驗預測檢查，模型預測的反應時間分布和資料像不像；
`single-dfp`／`dfp`＝五種架構各跑一場乾淨的 DFP；`full-experiment`＝多位受試者、每人校準再 DFP，
整場實驗。

真人實驗用的是後三支：`power_scan.py` 幫你決定要幾題、目標設多少；`analyze_participant.py`
分析一位受試者；`demo_parity.py` 給你看 Python 和 R 原碼算的數字一樣。「oracle」＝用原版
R 跑出來存檔的標準答案。「漂移差」＝兩個累積器的速度差，LNRM 分支用它當校準目標。

兩支都有 `--a-convention {separation,threshold,both}`（預設 `both`），對應 `decisions_for_author.md`
Decision A：`separation` 是 diffIRT 的定義（repo 字面），`threshold` 是 `psi Simulation_26MAR2019.R:117`
公式所隱含的定義（`simdiffT` 收到 `2a`）。每段都把兩種慣例的結果並排印出並存成 csv，
決定就從數字上選。

白話版：DDM 的 `a` 可以指「兩條界線的總距離」（separation）或「起點到一條界線的距離」
（threshold），後者剛好是前者的一半。原作者的腳本兩種用法都出現過，程式無法替你決定，
所以每段都兩種各跑一次，並排給你看。

```bash
PY=venv/bin/python
$PY scripts/simulate_psi.py all --quick                 # 約 1 分鐘，只驗流程
$PY scripts/simulate_lnrm_ogival.py all --quick          # 約 5 分鐘
$PY scripts/simulate_psi.py dfp --trials 300             # R 的設定
$PY scripts/simulate_lnrm_ogival.py salience single-dfp  # 兩次 ogival 擬合（各約 30 s）
$PY scripts/simulate_lnrm_ogival.py convergence          # 8 個 N（log 間距）× 2 慣例，joblib 4 核
$PY scripts/simulate_lnrm_ogival.py convergence --full   # R 原設定 N = 1…300，很慢
```

`--quick` 是小樣本快速跑，只確認流程不會壞，數字不要拿來用。「joblib 4 核」＝同時用電腦的
四個核心平行算。「log 間距」＝題數 N 不是 1、2、3…一路取，而是像 2、4、8、16 這樣跳著取。

輸出到 `output/`（gitignored）：每段一個子目錄，png + csv。

「gitignored」＝這個目錄不進版本控制，隨時可以刪掉重跑。

`simulate_lnrm_ogival.py` 另有 `--sampler {nuts,demetropolisz}`（預設 nuts）、`--fit-separate`（R 的 `fit.separate`：每層各擬合一個 lnrm0，畫在 ogival 曲線上）；`full-experiment` 會同時跑 R 的 `sft.allx` 對照組（全體用同一組 H/L）。

`nuts` 和 `demetropolisz` 是兩種取樣器（產生後驗的方法）；nuts 快且穩，是預設。
`--fit-separate` 是把每個強度層各自擬合一個最簡單的模型，當作「不假設曲線形狀」的對照點
畫在 S 形曲線上。`sft.allx` 是「不替每個人校準，全體用同一組 H / L」的對照組。

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

這幾點的意思：`:204/:213`、`:496-513` 都是 `simulateLNRM_ogival.R` 的行號。「CDF」＝累積分布
函數，「到時間 t 為止已反應的比例」。`loess` / `lowess` 是兩種畫平滑曲線的方法，結果很接近。
「`(a, v, ter, sdv)`」是每位假受試者的 DDM 參數，從群體值附近隨機抽。「真 α/β」是 Psi 應該
收斂到的正確答案；兩種 `a` 慣例各算一條，慣例選對時 Psi 的估計才會收斂到它。
「不裁切」＝如果反解出的強度超出實驗能呈現的範圍，程式只警告，不偷偷改成範圍上限，
讓你自己看到並決定怎麼辦。

## 所以你要做什麼

1. 第一次接觸：跑 `simulate_psi.py all --quick`，一分鐘，確認環境沒問題。
2. 設計實驗前：用 `power_scan.py --a … --v … --ter … --sdv …` 代入自己作業的 DDM 參數，
   看要幾題、目標設多少才判得準。
3. 收到一位受試者的 DFP csv：`analyze_participant.py 檔名 --json … --plot …`。
4. 要向人說明 Python 版可信：跑 `demo_parity.py`，把那張表拿去。
5. `a` 的意義（Decision A）：看 `results/decisions_demo.txt` 或自己跑 `demo_decisions.py`，從數字上選。
