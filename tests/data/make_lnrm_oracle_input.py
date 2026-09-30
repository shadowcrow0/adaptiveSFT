"""Fixed data set for the Stan-vs-PyMC comparison of lnrm2 (quadratic link).
Drawn from the model itself (adaptivesft.race.lnrm_random) at the truth used by model_lnrm2.py,
so both fits should recover mu 1.5, alpha 0.8, alpha2 -0.15, varZ 0.6, psi 0.12.
    python tests/data/make_lnrm_oracle_input.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from adaptivesft.models import d_numpy  # noqa: E402
from adaptivesft.race import lnrm_random  # noqa: E402

TRUTH = dict(mu=1.5, alpha=0.8, alpha2=-0.15, varZ=0.6, psi=0.12)
rng = np.random.default_rng(20260930)
x = rng.uniform(0.0, 3.0, 1000)
d = d_numpy("quadratic", x, TRUTH)
rt, correct = lnrm_random(d, TRUTH["mu"], TRUTH["varZ"], TRUTH["psi"], rng=rng)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lnrm_oracle_input.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["intensity", "rt", "correct"])
    w.writerows((f"{a:.12f}", f"{b:.12f}", int(c)) for a, b, c in zip(x, rt, correct))
print("wrote", out, len(x), "rows; accuracy", correct.mean().round(3), "min rt", rt.min().round(4))
