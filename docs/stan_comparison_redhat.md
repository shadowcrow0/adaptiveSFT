# 在 Red Hat 系機器上做 LNRM 對 Stan 的比對

目的：`adaptivesft.fit_lnrm(link="quadratic")` 是 `lnrm2.stan` 的 PyMC 版，六個元件裡只剩它沒對過原碼。
做法是在有 rstan 的機器上用**原始 Stan 模型**擬合一份固定資料，把後驗摘要存成 json 帶回 repo，
之後 `pytest` 在任何機器上都能比（不需要 R）。取樣器不同（Stan NUTS vs PyMC NUTS），比的是後驗
平均、SD、分位數在蒙地卡羅誤差內一致，不是逐位元。

```
   學校機器（有 rstan）                                   任何機器
   ─────────────────────────────────────────────          ────────────────────────────────
   tests/data/lnrm_oracle_input.csv（repo 內，固定）        tests/test_lnrm_vs_stan.py
          │  Rscript tests/data/make_lnrm_oracle.R                 │  pytest
          ▼                                                        ▼
   tests/data/lnrm_stan_oracle.json ── git add / push ──►  PyMC 後驗 vs json：|Δmean|/sd < .35 等
```

在 Arc（Slurm）上請直接看 `docs/hpc_arc_tutorial.md`，那裡有 `sbatch` 腳本；本文是單機版。

## 1. 裝 R 與 rstan（RHEL / Rocky / Alma 8 或 9）

```bash
sudo dnf install -y epel-release            # RHEL 本身要先 subscription-manager repos --enable codeready-builder-for-rhel-9-$(arch)-rpms
sudo dnf install -y R R-devel gcc-c++ make git
R --version                                  # 4.x 即可
```

R 套件二選一（腳本會自動偵測，也可用 `BACKEND=cmdstanr` / `BACKEND=rstan` 指定）：

**已經有 CmdStan 的話用 cmdstanr**（不用編 rstan；只裝 R 套件，1–2 分鐘）：

```r
install.packages(c("cmdstanr", "posterior", "jsonlite"),
                 repos = c("https://stan-dev.r-universe.dev", "https://cloud.r-project.org"))
cmdstanr::set_cmdstan_path("/path/to/cmdstan-2.36.0")   # 你裝 CmdStan 的目錄；或跑腳本時設環境變數 CMDSTAN=
cmdstanr::cmdstan_version()
```

**沒有 CmdStan 的話用 rstan**（會編 C++，10–20 分鐘；記憶體至少 4 GB）：

```r
install.packages(c("rstan", "jsonlite"), repos = "https://cloud.r-project.org")
library(rstan); stan_version(); packageVersion("StanHeaders")
```

CRAN 被擋的話用 conda（不需要 root，也不用編）：

```bash
conda create -n rstan -c conda-forge r-base r-rstan r-jsonlite
conda activate rstan
```

## 2. 哪個 `.stan` 檔

| StanHeaders / stanc | 用哪個檔 | 怎麼指定 |
|---|---|---|
| ≥ 2.33（2023-09 之後 CRAN 裝的都是） | `stan/lnrm2_array.stan`（五行陣列語法改成 `array[N] real`，其餘逐字相同） | 預設，不用指定 |
| ≤ 2.32（舊環境、Ubuntu apt 的 `r-cran-rstan`） | 原始 `lnrm2.stan` 也編得過 | `STAN_FILE=lnrm2.stan` |

兩個檔的模型完全相同，差別只在語法（`issue.md` S1）。想確認：`diff lnrm2.stan stan/lnrm2_array.stan` 只會有五行宣告不同。

## 3. 跑

```bash
git clone <本 repo> && cd adaptiveSFT
git checkout claude/grtv3-ada-adaptivesft-feasibility-btx6w8
Rscript tests/data/make_lnrm_oracle.R                 # 4 鏈 × 4000（2000 warmup），約 2–5 分鐘
#   或   STAN_FILE=lnrm2.stan Rscript tests/data/make_lnrm_oracle.R
git add tests/data/lnrm_stan_oracle.json && git commit -m "Add Stan oracle for lnrm2" && git push
```

腳本會印出 rstan 的 `summary()` 表（mean / sd / 2.5% / 97.5% / n_eff / Rhat）。Rhat 應 ≤ 1.01；
若 rstan 抱怨 divergent transitions，把 `stan(...)` 加上 `control = list(adapt_delta = 0.95)` 再跑一次。

## 4. 回到任何有 Python 的機器

```bash
pip install -r requirements-dev.txt
pytest tests/test_lnrm_vs_stan.py -s               # -s 會印 Stan vs PyMC 的對照表
```

測試比的東西：五個參數的後驗平均差 < 0.35 個後驗 SD、SD 差 < 30%、5/50/95% 分位數差 < 0.5 SD；
以及 R 端 `adaptiveSFT_functions.R:229-232` 那條二次式反解（`all_draws` 語意）在 json 裡的值，
與 Python `find_salience_polynomial` 差 < 3%。

## 5. 資料

`tests/data/lnrm_oracle_input.csv`：1000 試，從 `lnrm2` 自己的生成式抽（`make_lnrm_oracle_input.py`，
真值 mu 1.5、alpha 0.8、alpha2 −0.15、varZ 0.6、psi 0.12）。**不要重新產生**，否則 oracle 就對不上；
要換資料就兩邊一起重跑。
