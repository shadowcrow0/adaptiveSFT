# 在 Arc（UTSA 的 HPC，Slurm + Lmod）上跑這個 repo

這份文件教你在學校的共用計算叢集 Arc 上，把本 repo 的 Stan 比對與整套測試跑完、把結果推回 repo。

**一個比喻。** Arc 像一間大餐廳。你走進門看到的是**點餐櫃檯**（登入節點）：
你在這裡填單、排隊、查進度，但不能自己進廚房炒菜。真正的運算在**廚房**（計算節點）做。
**Slurm** 是負責收單、排隊、分配廚房的領班；`sbatch` 就是把你的單子交給領班。

## 0. 先弄懂幾個名詞

| 名詞 | 白話 |
|---|---|
| **HPC** | High-Performance Computing，很多台電腦接在一起的共用叢集。 |
| **登入節點** | 你 `ssh` 進去後落腳的那台機器。只能做輕工作：git、設定環境、送單。 |
| **計算節點** | 真正跑運算的機器。你不直接登入，而是透過 Slurm 把工作送過去。 |
| **Slurm** | 叢集的工作排程系統。負責排隊、分配機器、限制時間和記憶體。 |
| **`sbatch`** | 把一個腳本檔交給 Slurm 排隊執行，你可以離線。 |
| **`srun --pty bash`** | 向 Slurm 要一台計算節點，開一個互動式終端機給你用。適合除錯。 |
| **`squeue`** | 看排隊與執行中的工作。 |
| **分區（partition）** | Slurm 把機器分成幾群，每群有自己的名字、時限、核心數。送單時要指定一個。 |
| **module / Lmod** | Arc 預裝了很多軟體版本，`module load` 把你要的那個版本放進可用路徑。 |
| **conda** | 一個套件管理工具。它把 Python 或 R 和所有套件裝在一個獨立資料夾裡，不需要管理員權限，也不用自己編譯。 |
| **`/work`** | Arc 給每個人的大容量工作區。家目錄（`~`）配額小，環境要放 `/work`。 |

## 流程總覽

要做兩件事，順序固定：

```
   ① 一次性：登入 → 拿 repo → 建環境（R + rstan、Python）        約 30 分鐘，大多在等編譯
   ② 每次：  sbatch 送 R 作業（Stan oracle）→ sbatch 送 Python 作業（pytest + demo）→ git push 結果
```

登入節點只能做 git、module、建環境這類輕工作；**所有計算都要 `sbatch` / `srun`**。
在登入節點跑 Stan 或 PyMC 會被系統砍掉。

Arc 的細節（分區名、module 版本、`/work` 路徑）以 `sinfo`、`module avail`、Arc 的文件為準。
下面用 `<…>` 標的都是登入後要自己查一次、再填進去的。

---

## ① 一次性設定

### 1. 登入、拿 repo

```bash
ssh <abc123>@arc.utsa.edu                  # 校內或 VPN
cd /work/<abc123>                          # 工作區放 /work（家目錄配額小）；沒有的話 cd ~
git clone <repo 的 URL> adaptiveSFT
cd adaptiveSFT
git checkout claude/grtv3-ada-adaptivesft-feasibility-btx6w8
```

`<abc123>` 是你的 Arc 帳號。

### 2. 看有什麼 module

這一步是「看菜單」：Arc 有哪些版本的 R、anaconda、gcc，以及有哪些分區可以送單。

```bash
module avail R          # 例：R/4.3.x
module avail anaconda   # 例：anaconda3/2023.x
module avail gcc        # rstan 編 C++ 用；通常 R module 已經帶
sinfo -o "%P %l %c %m"  # 分區名、時限、核心數、記憶體；記下一個一般 CPU 分區的名字
```

### 3. R 端的 Stan 套件 —— 三條路擇一

R 要能呼叫 Stan（原作者的貝氏統計引擎）才能產生標準答案。三條路，挑你的情況。

**路 0：已經自己裝好 R 與 CmdStan，只缺套件**（最快）

CmdStan 是 Stan 的獨立命令列程式；cmdstanr 是 R 從外面呼叫它的薄殼，不用編 C++。

```bash
Rscript -e 'install.packages(c("cmdstanr","posterior","jsonlite"), repos=c("https://stan-dev.r-universe.dev","https://cloud.r-project.org"))'
export CMDSTAN=/path/to/cmdstan-2.36.0        # 你的 CmdStan 目錄；之後 make_lnrm_oracle.R 會自動用 cmdstanr
```

**路 A：Arc 的 R module + 自己的套件庫**（rstan 編 10–20 分鐘，要在計算節點編）

rstan 會把 Stan 的 C++ 原始碼整個編譯成程式，吃多核與記憶體，所以要先用 `srun` 借一台計算節點。
`R_LIBS_USER` 告訴 R 把套件裝到你自己的資料夾（系統資料夾你沒有寫入權限）。

```bash
module load R/<版本> gcc/<版本>
mkdir -p ~/R/lib
echo 'R_LIBS_USER=~/R/lib' >> ~/.Renviron
# 編譯要多核與記憶體，不要在登入節點做：
srun -p <分區> -c 4 --mem=8G -t 01:00:00 --pty bash
  module load R/<版本> gcc/<版本>
  Rscript -e 'install.packages(c("rstan","jsonlite"), repos="https://cloud.r-project.org", Ncpus=4)'
  Rscript -e 'library(rstan); cat(stan_version(), "\n")'    # 2.32.x 或 2.36.x 都可以
  exit
```

**路 B：conda（不用編，最省事；Arc 若擋 CRAN 也不受影響）**

CRAN 是 R 的官方套件庫。conda-forge 是 conda 的套件庫，裡面的 rstan 已經編好了。

```bash
module load anaconda3/<版本>
conda create -y -p /work/<abc123>/envs/rstan -c conda-forge r-base=4.3 r-rstan r-jsonlite
conda activate /work/<abc123>/envs/rstan
Rscript -e 'library(rstan); cat(stan_version(), "\n")'
conda deactivate
```

`hpc/arc_setup.sh` 把路 A 與路 B 都寫好了：`bash hpc/arc_setup.sh conda` 或 `bash hpc/arc_setup.sh module`。
它同時也會建好第 4 步的 Python 環境。檔案開頭有四個變數（anaconda / R / gcc 的 module 名、`ENV_ROOT`），
跑之前改成 `module avail` 看到的名字。

### 4. Python 環境（PyMC）

PyMC 是 Python 世界的貝氏統計引擎；本 repo 的模型就是用它寫的。

```bash
module load anaconda3/<版本>
conda create -y -p /work/<abc123>/envs/asft python=3.11
conda activate /work/<abc123>/envs/asft
pip install -r requirements-dev.txt        # 釘住的版本：pymc 5.28.5、pytensor 2.38.3、numba …
pip install -e .
conda deactivate
```

「釘住的版本」= 指定確切版本號，確保每台機器裝到一樣的東西。

---

## ② 每次要跑的作業

### 5. 送 R 作業：產生 Stan oracle

**oracle** 是「標準答案」：用原始 Stan 模型擬合一份固定資料，把後驗摘要存成
`tests/data/lnrm_stan_oracle.json`。之後 Python 端的 pytest 就拿它來對答案，不需要 R。

```bash
sbatch hpc/stan_oracle.sbatch              # 先打開檔案改 <分區>、<abc123>、選 conda 或 module
squeue -u <abc123>                         # 看狀態
tail -f slurm-<job-name>-<jobid>.out       # 結束時會印 rstan 的 summary 表與 "wrote tests/data/lnrm_stan_oracle.json"
```

打開 `hpc/stan_oracle.sbatch` 要改的地方：

- `--partition=`：換成 `sinfo` 看到的分區名（檔內預設 `compute1`）。
- `R_MODE`：`own`（預設，自己裝的 R + CmdStan，對應路 0）、`conda`（路 B）、`module`（路 A）。
  也可以不改檔案，送單時用 `R_MODE=conda sbatch hpc/stan_oracle.sbatch`。
- `ENV_ROOT` 預設 `/work/$USER/envs`；`$USER` 會自動帶入你的帳號，所以 `<abc123>` 只在路徑不同時才要改。
- `module load anaconda3` / `module load R gcc`：改成 `module avail` 看到的實際名字。

輸出檔名由 `--output=slurm-%x-%j.out` 決定：`%x` 是作業名（這支是 `lnrm2-stan-oracle`），`%j` 是作業編號。

4 鏈 × 4000 iter，1000 試，約 3–8 分鐘（含編譯 1–2 分鐘）。
「鏈」是取樣器從一個起點走的一條路；四條鏈走到同一個答案才算收斂。**Rhat** 就是這個收斂診斷，應 ≤ 1.01。

### 6. 送 Python 作業：整套測試 + 兩個 demo

```bash
sbatch hpc/pytest.sbatch
tail -f slurm-<job-name>-<jobid>.out
```

這支的作業名是 `adaptivesft-tests`。跑的東西：`pytest -q`（現在會包含 `test_lnrm_vs_stan.py`，因為 oracle 已經在）、
`scripts/demo_parity.py`、`scripts/demo_decisions.py`，輸出寫進 `results/`。約 15 分鐘。

- `pytest -q` = 把 `tests/` 底下的自動測試全部跑一遍。
- `demo_parity.py` = 把原始 R / Stan 算出的數字和 Python 移植的數字並排印出來。
- `demo_decisions.py` = 把 `decisions_for_author.md` 兩個待決問題的各種讀法，在同一份模擬資料上並排算一次。

### 7. 把結果推回 repo

```bash
git add tests/data/lnrm_stan_oracle.json results/
git commit -m "Add Stan oracle from Arc; parity and decision demos"
git push
```

之後任何機器 `pytest` 都會直接比 Stan 的後驗，不再需要 R。

---

## 常見問題

| 現象 | 原因 / 解法 |
|---|---|
| `sbatch: error: invalid partition` | 分區名不對，`sinfo` 查 |
| rstan 編譯時 `virtual memory exhausted` | `--mem` 開到 8G 以上，或改走 conda |
| `install.packages` 連不到 CRAN | 走 conda（路 B），conda-forge 通常有開 |
| Stan 抱怨 `real x[N]` 語法 | 你用了根目錄的 `lnrm2.stan` 配新 stanc；預設就是 `stan/lnrm2_array.stan`，不要設 `STAN_FILE` |
| PyMC 說找不到 BLAS、跑很慢 | `conda install -c conda-forge "blas=*=openblas"`；或忽略，只是慢 |
| `pytest` 在登入節點被砍 | 正常，用 `sbatch hpc/pytest.sbatch` |
| 想互動式除錯 | `srun -p <分區> -c 4 --mem=8G -t 01:00:00 --pty bash`，再手動 `conda activate` |

名詞補充：**stanc** 是 Stan 的編譯器，2.33 版改了陣列寫法，所以 repo 另備了 `stan/lnrm2_array.stan`（只改五行語法）。
**BLAS** 是矩陣運算的底層函式庫；缺它只是慢，結果不變。

## 所以你要做什麼

1. 第一次：登入 Arc，clone repo，用 `module avail` 和 `sinfo` 查名字，然後 `bash hpc/arc_setup.sh conda`（最省事）。
2. 打開 `hpc/stan_oracle.sbatch` 和 `hpc/pytest.sbatch`，把 `--partition=` 和 module 名改成你查到的。
3. `sbatch hpc/stan_oracle.sbatch`，等它印出 `wrote tests/data/lnrm_stan_oracle.json`，確認 Rhat ≤ 1.01。
4. `sbatch hpc/pytest.sbatch`，等約 15 分鐘。
5. `git add tests/data/lnrm_stan_oracle.json results/` → commit → push。
6. 絕對不要在登入節點直接跑 R 或 pytest。
