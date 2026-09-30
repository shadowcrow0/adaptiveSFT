# 在 Arc（UTSA 的 HPC，Slurm + Lmod）上跑這個 repo

要做兩件事，順序固定：

```
   ① 一次性：登入 → 拿 repo → 建環境（R + rstan、Python）        約 30 分鐘，大多在等編譯
   ② 每次：  sbatch 送 R 作業（Stan oracle）→ sbatch 送 Python 作業（pytest + demo）→ git push 結果
```

登入節點只能做 git、module、建環境這類輕工作；**所有計算都要 `sbatch` / `srun`**，在登入節點跑 Stan 或 PyMC 會被砍。
Arc 的細節（分區名、module 版本、`/work` 路徑）以 `sinfo`、`module avail`、Arc 的文件為準；下面用 `<…>` 標的
都是登入後要自己查一次的。

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

### 2. 看有什麼 module

```bash
module avail R          # 例：R/4.3.x
module avail anaconda   # 例：anaconda3/2023.x
module avail gcc        # rstan 編 C++ 用；通常 R module 已經帶
sinfo -o "%P %l %c %m"  # 分區名、時限、核心數、記憶體；記下一個一般 CPU 分區的名字
```

### 3. R 與 rstan —— 兩條路擇一

**路 A：Arc 的 R module + 自己的套件庫**（rstan 編 10–20 分鐘，要在計算節點編）

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

```bash
module load anaconda3/<版本>
conda create -y -p /work/<abc123>/envs/rstan -c conda-forge r-base=4.3 r-rstan r-jsonlite
conda activate /work/<abc123>/envs/rstan
Rscript -e 'library(rstan); cat(stan_version(), "\n")'
conda deactivate
```

`hpc/arc_setup.sh` 把兩條路都寫好了：`bash hpc/arc_setup.sh conda` 或 `bash hpc/arc_setup.sh module`。

### 4. Python 環境（PyMC）

```bash
module load anaconda3/<版本>
conda create -y -p /work/<abc123>/envs/asft python=3.11
conda activate /work/<abc123>/envs/asft
pip install -r requirements-dev.txt        # 釘住的版本：pymc 5.28.5、pytensor 2.38.3、numba …
pip install -e .
conda deactivate
```

---

## ② 每次要跑的作業

### 5. 送 R 作業：產生 Stan oracle

```bash
sbatch hpc/stan_oracle.sbatch              # 先打開檔案改 <分區>、<abc123>、選 conda 或 module
squeue -u <abc123>                         # 看狀態
tail -f slurm-<jobid>.out                  # 結束時會印 rstan 的 summary 表與 "wrote tests/data/lnrm_stan_oracle.json"
```

4 鏈 × 4000 iter，1000 試，約 3–8 分鐘（含編譯 1–2 分鐘）。Rhat 應 ≤ 1.01。

### 6. 送 Python 作業：整套測試 + 兩個 demo

```bash
sbatch hpc/pytest.sbatch
tail -f slurm-<jobid>.out
```

跑的東西：`pytest -q`（現在會包含 `test_lnrm_vs_stan.py`，因為 oracle 已經在）、`scripts/demo_parity.py`、
`scripts/demo_decisions.py`，輸出寫進 `results/`。約 15 分鐘。

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
