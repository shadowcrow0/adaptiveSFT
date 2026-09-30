#!/usr/bin/env bash
#SBATCH --job-name=adaptivesft-tests
#SBATCH --partition=compute1          # sinfo 查實際分區名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # PyMC 4 鏈平行
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
# 整套 pytest（有 Stan oracle 時包含 test_lnrm_vs_stan.py）+ 兩個英文 demo，輸出寫進 results/。
#   sbatch hpc/pytest.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ENV_ROOT="${ENV_ROOT:-/work/$USER/envs}"
module load anaconda3                     # 改成 module avail anaconda 看到的名字
PY=(conda run -p "$ENV_ROOT/asft" --no-capture-output)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # 讓 PyMC 的鏈平行而不是 BLAS 搶核

echo "host $(hostname)  cpus $SLURM_CPUS_PER_TASK  $(date)"
"${PY[@]}" python -c "import pymc, pytensor, numba; print('pymc', pymc.__version__, 'pytensor', pytensor.__version__, 'numba', numba.__version__)"
"${PY[@]}" pytest -q -s tests/test_lnrm_vs_stan.py || true     # 先單獨印 Stan vs PyMC 的表（沒 oracle 會 skip）
"${PY[@]}" pytest -q
mkdir -p results
"${PY[@]}" python scripts/demo_parity.py    > results/parity_demo.txt
"${PY[@]}" python scripts/demo_decisions.py > results/decisions_demo.txt
echo "done $(date)"
echo "接著：git add tests/data/lnrm_stan_oracle.json results/ && git commit && git push"
