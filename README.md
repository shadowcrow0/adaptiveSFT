# adaptiveSFT

Code for adapting salience levels for the Double Factorial Paradigm (Houpt, 2018–2019; R + Stan),
and its Python port.

**這是什麼。** 這個 repo 是一套程式，替每位受試者量出「容易」和「困難」兩個刺激強度，
好拿去跑 Double Factorial Paradigm（DFP，一種 2×2 設計的反應時間實驗）。原版是 Houpt 在
2018–2019 年用 R 加 Stan 寫的；這裡把它整套改寫成 Python。

**一個比喻：驗光。** 驗光師不會給每個人同一副眼鏡。他先問幾輪「這樣清楚嗎？」，
把度數調到剛好。這個 repo 做的就是這件事：先用幾十到幾百題量出受試者對刺激的敏感度，
再從中算出兩個刺激強度——一個高顯著度（high salience，看得很清楚）、一個低顯著度
（low salience，勉強看得到）。正式的 DFP 實驗就用這兩個強度。

幾個會一直出現的詞，先認識一下：

- **R**：統計界常用的程式語言。原作者用它寫整個流程。
- **Stan**：專門寫貝氏統計模型的語言。原作者的模型 `lnrm2.stan` 用它寫。
- **Python / PyMC**：Python 是另一種程式語言。PyMC 是 Python 裡做貝氏模型的工具，
  可以想成 Python 版的 Stan。
- **移植（port）**：同一套算法用另一種語言重寫一遍。數字要對得上才算成功。
- **LNRM**：lognormal race model，「兩個計時器賽跑」的反應時間模型。「答對」計時器和
  「答錯」計時器各自跑，誰先到就是受試者的回答。
- **Psi**：另一條校準路線。每答一題就重算「下一題該出多難」的適應程序。
- **DDM**：drift diffusion model，漂移擴散模型。這裡用它當「假受試者」做模擬。
- **SIC**：survivor interaction contrast。DFP 實驗最後用來判斷「兩個訊息是並行還是序列
  處理」的統計量。
- **後驗（posterior）**：模型看完資料後對參數的看法，是一整組可能值，不是單一數字。

原始碼與 Python 版的對照：

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

左欄是原作者的檔案，只讀不改。右欄是本 repo 的 Python 對應。「遺失」表示原作者的檔案
不在 repo 裡，Python 版是照文件和其他檔案重建的。「逐位元比對」表示 Python 算出來的數字
和 R 算出來的一模一樣，連小數點後最後一位都相同。

## 用

```bash
pip install -e ".[dev]"          # 或 pip install -r requirements-dev.txt（釘住的版本）
pytest                           # 約 3 分鐘，不需要 R / Stan / PsychoPy
python scripts/demo_parity.py    # R 原碼算出的值 vs Python，並排
python scripts/demo_decisions.py # Decision A / B 兩種讀法並排
```

一行一行看：

| 指令 | 做什麼 |
|---|---|
| `pip install -e ".[dev]"` | 安裝這個套件和它需要的工具。`-e` 表示「直接用現場的檔案」，之後改程式不用重裝 |
| `pytest` | 跑全部自動測試。測試＝一堆小程式，自動檢查每個函式算出的數字對不對 |
| `python scripts/demo_parity.py` | 把 R 原碼算的數字和 Python 算的數字並排印出來 |
| `python scripts/demo_decisions.py` | 把 Decision A / B 兩種讀法的結果並排印出來（這兩個決定見下面的 `decisions_for_author.md`） |

「釘住的版本」是指把每個工具的版本號寫死。三年後重裝也會裝到同一組，不會因為工具更新而壞掉。

套件說明：`adaptivesft/README.md`。腳本說明：`scripts/README.md`。

## 文件索引

repo 裡的 `.md` 檔是決策紀錄和說明。先看前四個就夠。

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

編號的意思：R＝R 端問題、S＝Stan 端、P＝舊 Python 腳本、M＝PyMC 移植時遇到的。
「`a` 的意義」是 DDM 裡界線參數的兩種讀法；「½L / L」是 ogival（S 形曲線）模型的一個
係數該是多少。這兩題程式無法替你決定，所以寫成英文給原作者看。

「Arc（Slurm）」是學校的高效能運算叢集；Slurm 是它的排隊系統。「oracle」＝用原版 R / Stan
跑出來、存成檔案的標準答案，之後 Python 版每次都拿它對。

## 自己學會寫這個模型：`learn/`

五個練習（Normal → 位移對數常態 → 賽跑 likelihood → 完整 lnrm2 → 反解 H/L），每題自己寫、
`python learn/check.py exN` 對答案（ex4 直接對 Stan 的後驗）。規則與進度表在 `learn/README.md`。

「反解 H/L」＝已知模型參數，倒過來算出哪個刺激強度會給你想要的難度；H 是高顯著度、L 是低顯著度。

## 三年後要改東西，從哪個測試開始

想法很簡單：改任何一個模組之前，先跑它對應的測試，確認現在是綠的。改完再跑一次。
還是綠的就表示沒改壞。

| 模組 | 對應的測試 | 改之前先跑 |
|---|---|---|
| `adaptivesft/race.py`（賽跑 likelihood） | `tests/test_race.py` | 密度積分 = 1、與 R 的 `dlognormalrace` 同值 |
| `adaptivesft/models.py`（PyMC 擬合） | `tests/test_models.py`、`tests/test_lnrm_vs_stan.py` | 後者對 Stan 的後驗（`tests/data/lnrm_stan_oracle.json`） |
| `adaptivesft/salience.py`（反解 H/L） | `tests/test_salience.py` | 五種 `alpha2_rule` 的行為 |
| `adaptivesft/ddm.py`（DDM 模擬） | `tests/test_ddm.py` | 對 R `diffIRT::simdiffT` 的分布 |
| `adaptivesft/sic.py`（SIC / MIC） | `tests/test_sic.py` | 對 R `sft::sic` 逐位元 |
| `adaptivesft/psi.py`（Psi） | `tests/test_psi.py`、`tests/test_parity_demo.py` | 對 R 逐試 1e−13 |
| `adaptivesft/experiment.py`（真人實驗用） | `tests/test_experiment.py` | |

第三欄的意思：「密度積分 = 1」是機率分布的基本要求，總機率必須是 1；「與 R 同值」「逐位元」
「逐試 1e−13」都是在說 Python 和 R 的差距小到可以當成零（1e−13 ＝ 小數點後 13 位才有差）。
「likelihood」＝在某組參數下，看到這筆資料的機率。「MIC」＝mean interaction contrast，SIC 的
平均數版本。

套件版本鎖在 `requirements-lock.txt`（Python 3.11）；三年後 `pip install -e .` 裝到新版 PyMC 跑不起來時，先用 lock 檔重建。

## 還沒有的

- `lnrm0 / lnrm1 / lnrm2a.stan`、`post95.Rdata`、兩個輸入 csv 不在 repo 裡；Python 的對應是重建，不是原檔。
- `decisions_for_author.md`：實驗端（Visual_AudioWM）已定 **B**——用 LNRM 分支、漂移差目標、本 repo 的 ogival 重建；A 只影響 DDM 模擬，維持未定。程式仍把兩種讀法都做成參數。
- LNRM 對 Stan 的比對已完成（Arc 上 cmdstanr 2.40.0 產生 `tests/data/lnrm_stan_oracle.json`）：五個參數的後驗平均差 ≤ 0.01 個後驗 SD，`tests/test_lnrm_vs_stan.py` 常駐。

白話翻譯第三點：Python 版和 Stan 原版擬合同一份資料，五個參數的估計值幾乎相同。差距
不到後驗標準差的百分之一，也就是遠小於估計本身的不確定度。「常駐」＝這個比對已經是
固定測試，每次 `pytest` 都會跑。

## 所以你要做什麼

1. 要跑真人實驗的校準：看 `adaptivesft/README.md`，用 `experiment.py` 那三個零件。
2. 想知道要幾題、檢定力夠不夠：看 `scripts/README.md`，跑 `power_scan.py`。
3. 想自己弄懂模型：從 `learn/README.md` 的 ex1 開始。
4. 要改程式：先跑上面那張表對應的測試，確認綠的再動手。
5. 遇到 `a` 的意義或 `lnrm2a` 的 ½L / L 這兩題：讀 `decisions_for_author.md`，那是研究者要定的，程式兩種都做好了。
