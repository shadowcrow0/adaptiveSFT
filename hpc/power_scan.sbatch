#!/usr/bin/env bash
#SBATCH --job-name=asft-power-scan
#SBATCH --partition=compute1          # Arc：compute1（65 節點，常滿）、compute3（6 節點）；送時可用 -p 蓋掉
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # PyMC 4 鏈平行
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
# 檢定力細掃：目標（H/L 漂移差）× 每格試次 × 五種架構 → sicGroup 判對率。
#   sbatch hpc/power_scan.sbatch
#   sbatch --export=ALL,NP=100 hpc/power_scan.sbatch        # 每個目標 100 位模擬受試者（預設 50）
# 每位模擬受試者擬合一次 lnrm2（Arc 上約 15–30 s），之後每個 n 只是 DDM 模擬 + SIC 檢定（很快）。
# 預設 3 個目標 × 50 人 = 150 次擬合 ≈ 1–1.5 小時。跑完看 results/power_scan_fine.csv，
# 或 python scripts/power_table.py results/power_scan_fine.csv 印成表。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ENV_ROOT="${ENV_ROOT:-/work/$USER/envs}"
ASFT_ENV="${ASFT_ENV:-$ENV_ROOT/asft}"   # 環境在別處：sbatch --export=ALL,ASFT_ENV=/work/<id>/envs/sim ...
NP="${NP:-50}"
TARGETS="${TARGETS:-2.0:0.5 4.0:1.0 8.0:1.3}"
NCELL="${NCELL:-84 100 120 150 200 250}"
module load anaconda3                     # 改成 module avail anaconda 看到的名字
PY=(conda run -p "$ASFT_ENV" --no-capture-output)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "host $(hostname)  cpus $SLURM_CPUS_PER_TASK  $(date)"
echo "targets: $TARGETS   n-cell: $NCELL   participants per target: $NP"
mkdir -p results
# shellcheck disable=SC2086
"${PY[@]}" python scripts/power_scan.py --targets $TARGETS --n-cell $NCELL --n-participants "$NP" \
    --out results/power_scan_fine.csv
"${PY[@]}" python scripts/power_table.py results/power_scan_fine.csv | tee results/power_scan_fine.txt
echo "done $(date)"
echo "接著：git add results/power_scan_fine.csv results/power_scan_fine.txt && git commit && git push"
