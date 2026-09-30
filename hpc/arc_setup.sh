#!/usr/bin/env bash
# 一次性環境設定（在 Arc 的登入節點或 srun 互動節點執行）。
#   bash hpc/arc_setup.sh conda     # R + rstan 與 Python 都用 conda（推薦，不編譯）
#   bash hpc/arc_setup.sh module    # R 用 Arc 的 module + 自己編 rstan；Python 用 conda
# 改下面四個變數再跑。
set -euo pipefail
MODE="${1:-conda}"
ANACONDA_MODULE="anaconda3"          # module avail anaconda  看實際名字
R_MODULE="R"                         # module avail R         例 R/4.3.2（MODE=module 才用）
GCC_MODULE="gcc"                     # module avail gcc       （MODE=module 才用）
ENV_ROOT="/work/$USER/envs"          # /work 沒有就改 $HOME/envs

mkdir -p "$ENV_ROOT"
module load "$ANACONDA_MODULE"

echo "== Python 環境：$ENV_ROOT/asft"
if [ ! -d "$ENV_ROOT/asft" ]; then
  conda create -y -p "$ENV_ROOT/asft" python=3.11
fi
conda run -p "$ENV_ROOT/asft" pip install -q -r requirements-dev.txt
conda run -p "$ENV_ROOT/asft" pip install -q -e .
conda run -p "$ENV_ROOT/asft" python -c "import pymc, pytensor; print('pymc', pymc.__version__, 'pytensor', pytensor.__version__)"

if [ "$MODE" = "conda" ]; then
  echo "== R 環境（conda）：$ENV_ROOT/rstan"
  if [ ! -d "$ENV_ROOT/rstan" ]; then
    conda create -y -p "$ENV_ROOT/rstan" -c conda-forge r-base=4.3 r-rstan r-jsonlite
  fi
  conda run -p "$ENV_ROOT/rstan" Rscript -e 'library(rstan); cat("rstan", as.character(packageVersion("rstan")), "stan", stan_version(), "\n")'
else
  echo "== R 環境（module $R_MODULE + 使用者套件庫）"
  module load "$R_MODULE" "$GCC_MODULE"
  mkdir -p "$HOME/R/lib"
  grep -q R_LIBS_USER "$HOME/.Renviron" 2>/dev/null || echo "R_LIBS_USER=$HOME/R/lib" >> "$HOME/.Renviron"
  Rscript -e 'if (!requireNamespace("rstan", quietly=TRUE) || !requireNamespace("jsonlite", quietly=TRUE))
                install.packages(c("rstan","jsonlite"), repos="https://cloud.r-project.org", Ncpus=4)
              library(rstan); cat("rstan", as.character(packageVersion("rstan")), "stan", stan_version(), "\n")'
fi
echo "== done. 接著：sbatch hpc/stan_oracle.sbatch，然後 sbatch hpc/pytest.sbatch"
