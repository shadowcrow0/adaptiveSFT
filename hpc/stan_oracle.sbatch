#!/usr/bin/env bash
#SBATCH --job-name=lnrm2-stan-oracle
#SBATCH --partition=compute1          # sinfo 查實際分區名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # rstan 4 鏈平行
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
# 產生 tests/data/lnrm_stan_oracle.json：用原始 lnrm2 模型（stan/lnrm2_array.stan，只改五行語法）
# 擬合 tests/data/lnrm_oracle_input.csv。之後 pytest 在任何機器上都能比 Stan 的後驗。
#   sbatch hpc/stan_oracle.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

R_MODE="${R_MODE:-conda}"                 # conda 或 module，與 hpc/arc_setup.sh 一致
ENV_ROOT="${ENV_ROOT:-/work/$USER/envs}"
module load anaconda3                     # 改成 module avail anaconda 看到的名字

if [ "$R_MODE" = "conda" ]; then
  RSCRIPT=(conda run -p "$ENV_ROOT/rstan" Rscript)
else
  module load R gcc                       # 改成實際版本
  RSCRIPT=(Rscript)
fi

echo "host $(hostname)  cpus $SLURM_CPUS_PER_TASK  $(date)"
"${RSCRIPT[@]}" -e 'library(rstan); cat("stan", stan_version(), "rstan", as.character(packageVersion("rstan")), "\n")'
# 用新語法檔（預設）。舊 stanc（<= 2.32）想用原檔：STAN_FILE=lnrm2.stan sbatch hpc/stan_oracle.sbatch
"${RSCRIPT[@]}" tests/data/make_lnrm_oracle.R
ls -l tests/data/lnrm_stan_oracle.json
echo "done $(date)"
