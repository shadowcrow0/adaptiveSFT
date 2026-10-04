# 在 Red Hat 系機器上做 LNRM 對 Stan 的比對

這份文件教你在一台能裝 R 的 Red Hat 系 Linux 機器上，用原作者的工具算出一份「標準答案」，
讓本 repo 的 Python 模型從此能在任何機器上自動對答案。

**一個比喻。** 同一份考卷，原作者用 Stan 這支筆算過，我們用 PyMC 這支筆重算。
兩支筆不同，字跡不會一模一樣；我們要確認的只是**答案在合理誤差內相同**。
先用原來那支筆算出來、存成一張答案卡，之後誰都能拿這張卡來對——這張卡在這裡叫 **oracle**。

## 0. 先弄懂幾個名詞

| 名詞 | 白話 |
|---|---|
| **Stan / rstan** | Stan 是原作者用的貝氏統計引擎。rstan 是從 R 呼叫它的套件。`lnrm2.stan` 是原作者寫給 Stan 的模型檔。 |
| **PyMC** | Python 世界的同類引擎。`adaptivesft.fit_lnrm(link="quadratic")` 就是用 PyMC 重寫的 `lnrm2.stan`。 |
| **後驗（posterior）** | 模型看完資料後，對每個參數「可能是多少」的整個分布。我們比的是這個分布的平均、SD、分位數。 |
| **NUTS** | 兩邊引擎共用的抽樣演算法（No-U-Turn Sampler）。它靠亂數在後驗裡走路，所以兩次結果本來就不會一字不差。 |
| **蒙地卡羅誤差** | 抽樣亂數帶來的隨機誤差。「在蒙地卡羅誤差內一致」= 差異小到可以歸咎於亂數。 |
| **oracle** | 用原始工具算出來、存起來的標準答案。這裡是一個 json 檔。 |
| **json** | 純文字的資料格式。任何語言都讀得了，所以 Python 端不需要 R 就能拿到 Stan 的結果。 |
| **pytest** | Python 的自動測試工具。打 `pytest` 就是「自動對答案」。 |

## 這件事為什麼要做

`adaptivesft.fit_lnrm(link="quadratic")` 是 `lnrm2.stan` 的 PyMC 版。本 repo 移植的六個元件裡，
只剩它還沒跟原始碼對過。

做法分兩段。第一段在有 rstan 的機器上，用**原始 Stan 模型**擬合一份固定的資料，
把後驗摘要存成 json 帶回 repo。第二段在任何機器上跑 `pytest`，拿 PyMC 的後驗跟那份 json 比，
不需要 R。因為取樣器不同（Stan NUTS vs PyMC NUTS），比的是後驗平均、SD、分位數
在蒙地卡羅誤差內一致，不是逐位元。

```
   學校機器（有 rstan）                                   任何機器
   ─────────────────────────────────────────────          ────────────────────────────────
   tests/data/lnrm_oracle_input.csv（repo 內，固定）        tests/test_lnrm_vs_stan.py
          │  Rscript tests/data/make_lnrm_oracle.R                 │  pytest
          ▼                                                        ▼
   tests/data/lnrm_stan_oracle.json ── git add / push ──►  PyMC 後驗 vs json：|Δmean|/sd < .35 等
```

本文是**單機版**：一台你自己能裝東西的機器。
如果你用的是 Arc（學校的 Slurm 叢集），請直接看 `docs/hpc_arc_tutorial.md`，那裡有 `sbatch` 腳本。

## 1. 裝 R 與 rstan（RHEL / Rocky / Alma 8 或 9）

`dnf` 是 Red Hat 系 Linux 的套件安裝指令；EPEL 是一個額外的套件庫，R 放在那裡。
`sudo` 代表要有管理員權限。

```bash
sudo dnf install -y epel-release            # RHEL 本身要先 subscription-manager repos --enable codeready-builder-for-rhel-9-$(arch)-rpms
sudo dnf install -y R R-devel gcc-c++ make git
R --version                                  # 4.x 即可
```

接著要讓 R 能呼叫 Stan。有兩條路，二選一。腳本會自動偵測你裝了哪一條；
也可以用環境變數 `BACKEND=cmdstanr` 或 `BACKEND=rstan` 指定。

**路 1：已經有 CmdStan 的話用 cmdstanr。** CmdStan 是 Stan 的獨立命令列程式；
cmdstanr 是 R 從外面呼叫它的薄殼。好處是不用編 rstan 的 C++；只裝 R 套件，1–2 分鐘。

```r
install.packages(c("cmdstanr", "posterior", "jsonlite"),
                 repos = c("https://stan-dev.r-universe.dev", "https://cloud.r-project.org"))
cmdstanr::set_cmdstan_path("/path/to/cmdstan-2.36.0")   # 你裝 CmdStan 的目錄；或跑腳本時設環境變數 CMDSTAN=
cmdstanr::cmdstan_version()
```

**路 2：沒有 CmdStan 的話用 rstan。** rstan 要把 Stan 的 C++ 原始碼整個編譯成程式，
所以慢（10–20 分鐘），記憶體至少 4 GB。

```r
install.packages(c("rstan", "jsonlite"), repos = "https://cloud.r-project.org")
library(rstan); stan_version(); packageVersion("StanHeaders")
```

**CRAN 被擋的話用 conda。** CRAN 是 R 的官方套件庫；有些機構的網路會擋。
conda 是一個獨立的套件管理環境，不需要 root，也不用自己編，套件是預先編好的。

```bash
conda create -n rstan -c conda-forge r-base r-rstan r-jsonlite
conda activate rstan
```

## 2. 哪個 `.stan` 檔

Stan 的編譯器（stanc）在 2.33 版改了陣列的寫法，舊寫法會被拒絕。
所以 repo 裡有兩個檔，依你的 Stan 版本選一個。

| StanHeaders / stanc | 用哪個檔 | 怎麼指定 |
|---|---|---|
| ≥ 2.33（2023-09 之後 CRAN 裝的都是） | `stan/lnrm2_array.stan`（五行陣列語法改成 `array[N] real`，其餘逐字相同） | 預設，不用指定 |
| ≤ 2.32（舊環境、Ubuntu apt 的 `r-cran-rstan`） | 原始 `lnrm2.stan` 也編得過 | `STAN_FILE=lnrm2.stan` |

兩個檔的模型完全相同，差別只在語法（`issue.md` S1）。
想自己確認：`diff lnrm2.stan stan/lnrm2_array.stan` 只會有五行宣告不同。

## 3. 跑

```bash
git clone <本 repo> && cd adaptiveSFT
git checkout claude/grtv3-ada-adaptivesft-feasibility-btx6w8
Rscript tests/data/make_lnrm_oracle.R                 # 4 鏈 × 4000（2000 warmup），約 2–5 分鐘
#   或   STAN_FILE=lnrm2.stan Rscript tests/data/make_lnrm_oracle.R
git add tests/data/lnrm_stan_oracle.json && git commit -m "Add Stan oracle for lnrm2" && git push
```

「4 鏈 × 4000（2000 warmup）」的意思：取樣器從四個起點各走 4000 步（這四條路叫**鏈**），
每條前 2000 步是暖身、不算數，只留後 2000 步。

腳本會印出 rstan 的 `summary()` 表（mean / sd / 2.5% / 97.5% / n_eff / Rhat）。
看兩個數就好：

- **Rhat** 是收斂診斷。它問「四條鏈有沒有走到同一個答案」。應 ≤ 1.01。
- **divergent transitions** 是取樣器在陡峭地形上失足的警告。若 rstan 抱怨這個，
  把 `stan(...)` 加上 `control = list(adapt_delta = 0.95)` 再跑一次——意思是叫它走小步一點。

## 4. 回到任何有 Python 的機器

```bash
pip install -r requirements-dev.txt
pytest tests/test_lnrm_vs_stan.py -s               # -s 會印 Stan vs PyMC 的對照表
```

測試比的東西，一共兩組：

1. 五個參數的後驗：平均差 < 0.35 個後驗 SD、SD 差 < 30%、5/50/95% 分位數差 < 0.5 SD。
2. 「反解」的結果：R 端 `adaptiveSFT_functions.R:229-232` 那條二次式反解（`all_draws` 語意）
   在 json 裡的值，與 Python `find_salience_polynomial` 差 < 3%。
   「反解」是指把目標的分離量倒推回刺激強度——校準實驗最後拿去用的就是這個數。

## 5. 資料

`tests/data/lnrm_oracle_input.csv`：1000 試，從 `lnrm2` 自己的生成式抽（`make_lnrm_oracle_input.py`，
真值 mu 1.5、alpha 0.8、alpha2 −0.15、varZ 0.6、psi 0.12）。

**不要重新產生**，否則 oracle 就對不上——答案卡是對著這份考卷做的，考卷換了答案卡就作廢。
要換資料就兩邊一起重跑（csv 重產、R 重跑、json 重推）。

## 所以你要做什麼

1. 在一台能裝 R 的 Red Hat 系機器上，照 §1 裝 R，再從 cmdstanr / rstan / conda 三條路挑一條。
2. 照 §3 跑 `Rscript tests/data/make_lnrm_oracle.R`，確認印出的 Rhat ≤ 1.01。
3. 把 `tests/data/lnrm_stan_oracle.json` commit、push。
4. 回到任何有 Python 的機器跑 `pytest tests/test_lnrm_vs_stan.py -s`。綠燈就代表 PyMC 版與原始 Stan 模型一致。
5. 不要動 `tests/data/lnrm_oracle_input.csv`。
