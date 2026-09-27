"""產生 tests/data/sic_input.csv：固定種子的 PAR-OR DFP 資料，給 make_sic_oracle.R（R）與 test_sic.py 共用。
    python tests/data/make_sic_input.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from adaptivesft.ddm import dfp_ddm  # noqa: E402

rng = np.random.default_rng(20260927)
rows = []
for name, (d1, d2) in dict(HH=(2.5, 2.5), HL=(2.5, 1.0), LH=(1.0, 2.5), LL=(1.0, 1.0)).items():
    rt, cr = dfp_ddm(120, d1, d2, 3.0, 0.1, 0.2, "PAR", "OR", rng=rng)
    rows += [(name, float(f"{t:.6f}")) for t in rt[cr == 1]]   # 六位小數：R 與 Python 讀到同一個 double
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sic_input.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cell", "rt"])
    w.writerows(rows)
print("wrote", out, len(rows), "rows")
