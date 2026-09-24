# bug.md — 這台機器現在能跑什麼、不能跑什麼（2026-09-24 實測）

> 一句話：**Python 端全部能跑；R 端不能，因為三個 CRAN 套件沒裝，而 CRAN 被這台
> 機器的網路政策擋住。** 不是 R 壞、不是程式碼壞、不是版本問題，是網路。
> 逐條指令與原始輸出見 `env_report.md`（subagent 寫，關鍵幾條我重跑過）。

```
                 R 端                                   Python 端
   ┌─────────────────────────────────┐      ┌──────────────────────────────────┐
   │ R 4.3.3            ✔            │      │ venv Python 3.11 + PyMC 5.28.5 ✔ │
   │ g++ / make / node  ✔            │      │ model_lnrm2.py    跑完 16 s   ✔ │
   │ rstan / diffIRT / sft  ✘ 沒裝    │      │ model_lnrm2a.py   跑完  8 s   ✔ │
   │ install.packages()               │      │ lnrm2_pymc.py     跑完 21 s   ✔ │
   │     └──► cloud.r-project.org     │      │ adaptive_sft2.py  ✘ 要 pystan   │
   │           ✘ proxy 回 403         │      │   pystan 2.19 在 3.11 編不起來  │
   └─────────────────────────────────┘      └──────────────────────────────────┘
```

---

## 1. R 端：發生了什麼

### 1.1 三個套件不在機器上

`adaptiveSFT_functions.R:1-3` 字面：

```r
require(rstan)
require(diffIRT)
require(sft)
```

實測 `requireNamespace()` 三個都是 `FALSE`。機器上的 R 只有 base + recommended
共 29 個套件，沒有任何 CRAN 附加套件。

### 1.2 為什麼 `source()` 第 4 行就死

`require()` 找不到套件只會警告不會停，但緊接著 `adaptiveSFT_functions.R:4`：

```r
rstan_options(auto_write = TRUE)
```

這個函式來自 rstan，沒裝就是：

```
Error in rstan_options(auto_write = TRUE) :
  could not find function "rstan_options"
```

所以任何 `source("adaptiveSFT_functions.R")` 的腳本（`simulateLNRM_ogival.R:1`、
兩個 `psi Simulation_*.R`）都在這一行停。五個 R 檔語法都 `parse()` 得過，
問題純粹是套件。

### 1.3 為什麼裝不起來

```
Rscript -e 'install.packages("rstan", repos="https://cloud.r-project.org")'
```

```
Warning: unable to access index for repository https://cloud.r-project.org/src/contrib:
  cannot open URL 'https://cloud.r-project.org/src/contrib/PACKAGES'
Warning message:
package 'rstan' is not available for this version of R
```

第二行的「not available for this version of R」是**誤導**：R 根本沒拿到套件清單，
所以什麼都「not available」。真正的原因：

```
   R ──► HTTPS_PROXY ──► 這台機器的出口 gateway ──► cloud.r-project.org:443
                                     │
                                     └── 回 403：CONNECT 被拒（政策）
```

```
curl https://cloud.r-project.org/src/contrib/PACKAGES
→ curl: (56) CONNECT tunnel failed, response 403
```

proxy 狀態頁記錄：`"kind": "connect_rejected", "detail": "gateway answered 403 to
CONNECT (policy denial or upstream failure)", "host": "cloud.r-project.org:443"`。
試過的 CRAN 鏡像（cloud.r-project.org、cran.r-project.org、cran.rstudio.com、
packagemanager.posit.co）全部一樣。

**這是這個 session 的網路政策設定，不是 R 的問題。** 解法在 §3。

### 1.4 其餘 R 端前置條件其實都在

| 東西 | 狀態 | 為什麼重要 |
|---|---|---|
| g++ 13.3、make、gfortran | ✔ | rstan 要編 C++ |
| node v22 | ✔ | rstan ≥ 2.26 用 JavaScript 跑 Stan 編譯器 |
| StanHeaders 的 `stanc.js` | ✘ 找不到 | 隨 StanHeaders 套件來，套件裝不了它就沒有 |
| cmdstan / `stanc` 執行檔 | ✘ | 沒裝 |

也就是說：**一旦 CRAN 通了，rstan 應該裝得起來**。裝起來之後會撞的是
`issue.md` R2 / S1（新 StanHeaders 拒絕 `lnrm2.stan` 的舊陣列語法）。

---

## 2. Python 端：發生了什麼

### 2.1 能跑的

venv（`scratchpad/venv`，Python 3.11.15）：pymc 5.28.5、pytensor 2.38.3、
numba 0.65.1、arviz 0.23.4、scipy 1.17.1、numpy 2.4.6。

| 指令 | 結果 |
|---|---|
| `python model_lnrm2.py` | 跑完，16 s，R-hat 1.01 |
| `python model_lnrm2a.py --n 300 --tune 300 --draws 300 --chains 4` | 跑完，8 s（這麼小 R-hat 當然差，不是 bug） |
| `python lnrm2_pymc.py` | 跑完，21 s，NUTS，0 divergences |

唯一的警告：pytensor 說找不到 BLAS（`could not link to a BLAS installation`），
只是慢一點，不影響結果。

### 2.2 不能跑的

**系統 python** (`/usr/local/bin/python3`) 什麼都沒裝，`import numpy` 就掛。
一定要用 venv 那個 python。

**`adaptive_sft2.py`**：

```
  File "/home/user/adaptiveSFT/adaptive_sft2.py", line 2, in <module>
    import pystan
ModuleNotFoundError: No module named 'pystan'
```

而且裝不起來：PyPI 通（`pip download pystan==2.19.1.1` 抓得到 16 MB 原始碼），
但 pystan 2.19 沒有 Python 3.11 的 wheel，從原始碼編：

```
Cython>=0.22 and NumPy are required.
error: metadata-generation-failed
```

pystan 2 已停止維護，跟新 NumPy / Cython 3 不相容。這個檔案本來就有 P1–P5
五個 bug（`issue.md` §3），不值得救；它的 Stan 部分已由 `model_lnrm2.py` 取代。

**`cmdstanpy`** 抓得到 wheel，但 CmdStan 本體要從 github releases 下載並編譯，沒試。

---

## 3. 怎麼解

| 問題 | 誰能解 | 做法 |
|---|---|---|
| CRAN 被擋 | **你**（環境設定） | 這個 cloud 環境的 Network access 設定：改成較寬的等級，或把 `cloud.r-project.org` 加進允許清單。位置：session 標題列的環境選單 → Edit。改完新開 session |
| 裝了 rstan 之後 `lnrm2.stan` 編不過 | 你 / 原作者 | `issue.md` S1 那五行，或不用 rstan 改走 PyMC |
| `adaptive_sft2.py` 要 pystan | 不用解 | 已被 `model_lnrm2.py` 取代 |
| 系統 python 沒 numpy | 已解 | 用 venv |

---

## 4. 資源

```
磁碟 29 GB 可用   CPU 4 核   RAM 15 GB（用 1.2 GB）
```

夠跑。PyMC 腳本目前把 chains 限制在 2 個 job，可以開到 4。
