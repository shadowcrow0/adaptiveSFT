"""
Parity demo: what the original R / Stan code computes, what the Python port computes, and how the
numbers of the two branches (Psi accuracy targets vs. LNRM drift-separation targets) and of the two
readings of the DDM boundary parameter map onto each other.

Every "R" column below is a number produced by the original code (sft::sic, diffIRT::simdiffT,
the Psi loop of psiSimulation_functions.R) and stored in tests/data/*_r_oracle.json by the
make_*_oracle.R scripts. Nothing in this file calls R.

    python scripts/demo_parity.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
DATA = os.path.join(os.path.dirname(HERE), "tests", "data")

from adaptivesft.ddm import ddm_p_correct, simdiffT                      # noqa: E402
from adaptivesft.psi import GRIDS, inv_pm_function, make_psi, pm_function, salience_levels  # noqa: E402
from adaptivesft.salience import accuracy_to_targ, targ_to_accuracy      # noqa: E402
from adaptivesft.sic import mic_test, sic                                # noqa: E402


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def header(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def row(label, r_val, py_val, tol=None, note=""):
    diff = abs(r_val - py_val)
    ok = "" if tol is None else ("  OK" if diff <= tol else "  MISMATCH")
    print(f"  {label:<44} R: {r_val:>14.10g}   Python: {py_val:>14.10g}   |diff| = {diff:.1e}{ok}  {note}")


# ----------------------------------------------------------------------------------------------
def section_sic():
    header("1. sft::sic (R) vs adaptivesft.sic (Python) on the same 473 correct-trial RTs")
    o = load("sic_r_oracle.json")
    cells = {k: np.asarray(o["cells"][k], float) for k in ("HH", "HL", "LH", "LL")}
    s = sic(**cells)
    print("  Same data, same statistics, differences at machine precision:")
    row("SIC(t): max |R - Python| over all t", 0.0, float(np.max(np.abs(np.asarray(o["SIC_values"]) - s["SIC"][1]))), 1e-12)
    row("D+ (Houpt-Townsend KS-SIC)", o["Dplus"], s["SICtest"]["positive"][0], 1e-12)
    row("p(D+) = exp(-2 N D+^2)", o["p_Dplus"], s["SICtest"]["positive"][1], 1e-12)
    row("D-", o["Dminus"], s["SICtest"]["negative"][0], 1e-12)
    row("p(D-)", o["p_Dminus"], s["SICtest"]["negative"][1], 1e-12)
    for (name, stat, p), r_stat, r_p in zip(s["Dominance"], o["dom_stat"], o["dom_p"]):
        row(f"dominance {name}: p", r_p, p, 1e-10)
    row("MIC = (LL - LH) - (HL - HH)", o["MIC"], s["MICtest"]["statistic"], 1e-12)
    row("p(MIC), adjusted rank transform", o["p_MIC_art"], s["MICtest"]["p_value"], 1e-8)
    row("p(MIC), ANOVA", o["p_MIC_anova"], mic_test(**cells, method="anova")["p_value"], 1e-8)


# ----------------------------------------------------------------------------------------------
def section_ddm():
    header("2. diffIRT::simdiffT (R) vs adaptivesft.ddm.simdiffT (Python): 20000 trials each, "
           "independent random streams")
    o = load("ddm_r_oracle.json")
    rng = np.random.default_rng(1)
    print("  Distributional comparison (Monte-Carlo error ~0.005 on P, ~1% on mean RT):")
    for name, c in o.items():
        rt, x = simdiffT(20000, c["a"], c["mv"], c["sv"], c["ter"], rng=rng)
        print(f"\n  case {name}: a={c['a']}, mean drift={c['mv']:.3f}, sd drift={c['sv']}, ter={c['ter']}")
        row("P(upper boundary)", c["p_upper"], float(x.mean()), 0.012)
        row("closed form 1/(1+exp(-a*drift)) (simdiffT.r:6)", c["closed_form_p"], float(ddm_p_correct(c["a"], c["mv"])), 1e-12)
        row("mean RT", c["rt_mean"], float(rt.mean()), 0.03 * c["rt_mean"])
        q = np.quantile(rt, [.1, .25, .5, .75, .9])
        row("median RT", c["rt_q"][2], float(q[2]), 0.03 * c["rt_q"][2])


# ----------------------------------------------------------------------------------------------
def section_psi():
    header("3. The Psi loop of psiSimulation_functions.R (R, verbatim) vs adaptivesft.psi.Psi (Python), "
           "same observer, same uniforms")
    o = load("psi_r_oracle.json")
    u = np.loadtxt(os.path.join(DATA, "psi_oracle_input.csv"), delimiter=",", skiprows=1)[:, 1]
    n = len(o["intensity"])
    sim = o["sim"]
    psi = make_psi("colour")
    intens, resp, alpha, beta = [], [], [], []
    for t in range(n):
        intens.append(psi.next_intensity)
        r = int(u[t] < pm_function(psi.next_intensity, sim["a"], sim["b"], sim["d"]))
        resp.append(r)
        psi.update(r)
        a_hat, b_hat = psi.estimate()
        alpha.append(a_hat)
        beta.append(b_hat)
    same_x = int(np.sum(np.asarray(o["intensity"]) == np.asarray(intens)))
    same_r = int(np.sum(np.asarray(o["response"]) == np.asarray(resp)))
    print(f"  simulated observer: alpha={sim['a']}, beta={sim['b']}, lapse={sim['d']}  (R :10-12);  {n} trials")
    print(f"  stimulus chosen on every trial identical in R and Python: {same_x}/{n}")
    print(f"  response identical on every trial:                      {same_r}/{n}")
    row(f"alpha-hat after trial {n} (posterior mean, R :154-163)", o["alpha"][-1], alpha[-1], 1e-10)
    row(f"beta-hat  after trial {n}", o["beta"][-1], beta[-1], 1e-10)
    row("max |alpha-hat diff| over all trials", 0.0, float(np.max(np.abs(np.asarray(o["alpha"]) - alpha))), 1e-10)
    row("max |beta-hat diff|  over all trials", 0.0, float(np.max(np.abs(np.asarray(o["beta"]) - beta))), 1e-10)
    (H, L), _ = salience_levels(alpha[-1], beta[-1], sim["d"], [0.99, 0.90])
    row("H = inv.pm.function(.99, alpha-hat, beta-hat) (R :183)", o["high"], H, 1e-10)
    row("L = inv.pm.function(.90, ...) (R :184)", o["low"], L, 1e-10)
    print("\n  trial   x presented (R)   x (Py)   r   alpha-hat (R)   alpha-hat (Py)   beta-hat (R)   beta-hat (Py)")
    for t in list(range(0, min(n, 5))) + [n - 1]:
        print(f"  {t + 1:5d}   {o['intensity'][t]:>15.1f}   {intens[t]:>6.1f}   {resp[t]}   {o['alpha'][t]:>13.6f}"
              f"   {alpha[t]:>14.6f}   {o['beta'][t]:>12.6f}   {beta[t]:>13.6f}")


# ----------------------------------------------------------------------------------------------
def section_conversions():
    header("4. Converting between the quantities the two branches and the two conventions use")

    # 4a. the two readings of a, evaluated at the original scripts' numbers
    a, v, thres50, xmax = 1.45, 1.6, 6.0, 50.0                       # psiSimulation_functions.R:15, 93-97
    print("\n  4a. DDM boundary parameter: 'threshold' (26MAR2019.R:117, exponent 2*a*drift) vs "
          "'separation' (diffIRT, exponent a*drift)")
    print(f"      colour range [-55, 50], thres50 = {thres50:g}, a = {a}, v = {v};  scaled x = (x - thres50)/(50 - thres50)")
    print(f"      {'x (colour units)':>18} {'scaled':>8} {'drift':>7} {'P(correct) threshold':>22} {'P(correct) separation':>23}")
    for x in (50.0, 53.64586, 101.6397):
        sc = (x - thres50) / (xmax - thres50)
        print(f"      {x:>18.3f} {sc:>8.3f} {sc * v:>7.3f} {ddm_p_correct(a, sc * v, a_is_separation=False):>22.4f}"
              f" {ddm_p_correct(a, sc * v, a_is_separation=True):>23.4f}")
    print("      (50 is the range top; 53.6 and 101.6 are the L and H values hard-coded in psi Simulation_25JUNE2018.R:177,174)")
    print(f"      {'target accuracy':>18} {'x needed, threshold':>22} {'x needed, separation':>23}")
    for p in (0.99, 0.90):
        lg = np.log(p / (1 - p))
        x_thr = thres50 + lg / (2 * a) / v * (xmax - thres50)
        x_sep = thres50 + lg / a / v * (xmax - thres50)
        print(f"      {p:>18.2f} {x_thr:>22.1f} {x_sep:>23.1f}")
    print("      -> under 'threshold' the 99% point is the range top (which the 2019 script hard-codes as 50);")
    print("         under 'separation' it is at x = 93, outside the range (which the 2018 run found: 101.6).")

    # 4b. Psi accuracy targets <-> LNRM separation targets, through the LNRM noise varZ
    print("\n  4b. Accuracy target (Psi branch, .99 / .90) <-> accumulator separation target "
          "(LNRM branch, h_targ 8.0 / l_targ 1.3)")
    print("      P(correct) = Phi( (z2 - z1) / (varZ * sqrt 2) )   so   targ = varZ * sqrt2 * Phi^-1(p)")
    for varZ in (0.6, 0.95):
        print(f"      varZ = {varZ}  (0.6 = model_lnrm2.py test value, 0.95 = fitted in results/p6):")
        print(f"        {'accuracy':>10} -> {'separation':>11}        {'separation':>11} -> {'accuracy':>10}")
        for p, t in zip((0.99, 0.90, 0.75), (8.0, 1.3, 0.5)):
            print(f"        {p:>10.2f} -> {float(accuracy_to_targ(p, varZ)):>11.3f}        {t:>11.1f} -> {float(targ_to_accuracy(t, varZ)):>10.4f}")
    print("      -> Houpt's h_targ = 8.0 is accuracy 1.0000 at any plausible varZ; l_targ = 1.3 is 0.83-0.94.")
    print("         The Psi targets .99/.90 correspond to separations of about 2.0/1.1 at varZ 0.6.")

    # 4c. Psi alpha/beta -> H/L in stimulus units -> DDM accuracy (both conventions)
    print("\n  4c. From a Psi estimate to what the DDM observer actually does at H and L")
    o = load("psi_r_oracle.json")
    a_hat, b_hat, d = o["alpha"][-1], o["beta"][-1], o["sim"]["d"]
    print(f"      R's own final estimate on the oracle run: alpha-hat = {a_hat:.3f}, beta-hat = {b_hat:.3f}")
    print(f"      {'level':>6} {'target p':>9} {'x = inv.pm.function':>20} {'scaled':>8} {'DDM P, threshold':>17} {'DDM P, separation':>18}")
    for lab, p in (("H", 0.99), ("L", 0.90)):
        x = float(inv_pm_function(p, a_hat, b_hat, d))
        sc = (x - thres50) / (xmax - thres50)
        print(f"      {lab:>6} {p:>9.2f} {x:>20.3f} {sc:>8.3f} {ddm_p_correct(a, sc * v, a_is_separation=False):>17.4f}"
              f" {ddm_p_correct(a, sc * v, a_is_separation=True):>18.4f}")
    print("      (the oracle observer is a cumulative normal, not a DDM; this shows the mapping, not a recovery)")

    # 4d. R's own beta grids vs the beta of the two DDM readings
    print("\n  4d. R's hard-coded beta grids (psiSimulation_functions.R:17, :224) vs the observer's true beta")
    for dim, sep_true, thr_true in (("colour", 31.2, 15.8), ("orientation", 18.9, 9.65)):
        lo, hi, _ = GRIDS[dim]["b"]
        print(f"      {dim:>12}: grid [{lo:g}, {hi:g}]   true beta under separation = {sep_true}"
              f"{'  (outside grid)' if not lo <= sep_true <= hi else ''}   under threshold = {thr_true}"
              f"{'  (outside grid)' if not lo <= thr_true <= hi else ''}")
    print("      -> only the 'threshold' reading keeps both true betas inside the grids the author wrote.")


if __name__ == "__main__":
    section_sic()
    section_ddm()
    if os.path.exists(os.path.join(DATA, "psi_r_oracle.json")):
        section_psi()
    else:
        print("\n(psi_r_oracle.json not found: run Rscript tests/data/make_psi_oracle.R)")
    section_conversions()
    print()
