"""Pre-drawn uniforms for the Psi oracle: the same u[t] drives the simulated observer in R
(make_psi_oracle.R) and in Python (test_parity_demo.py), so both see identical responses.
    python tests/data/make_psi_oracle_input.py
"""
import csv
import os

import numpy as np

rng = np.random.default_rng(20260930)
u = rng.uniform(size=300)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "psi_oracle_input.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "u"])
    w.writerows((i + 1, f"{v:.15f}") for i, v in enumerate(u))   # 15 decimals: identical double in R and Python
print("wrote", out, len(u), "rows")
