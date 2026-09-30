"""
Decision A / Decision B demo (decisions_for_author.md): what each reading of the ambiguous
quantity produces, computed side by side on the same simulated data, in English.

  Decision A  the DDM boundary parameter `threshold` (a): diffIRT's boundary *separation*
              (P = 1/(1+exp(-a*drift))) or the distance from the start point to the boundary
              (P = 1/(1+exp(-2*a*drift)), the formula on psi Simulation_26MAR2019.R:117).
  Decision B  the ogival LNRM (lnrm2a.stan, file missing): is the per-accumulator offset
              (1/2)*L*inv_logit(...) or L*inv_logit(...); is L fixed at 10 or estimated; and what
              the separation targets h_targ = 8.0 / l_targ = 1.3 mean in accuracy.

    python scripts/demo_decisions.py [--quick]

Runtime ~4 min (--quick ~1 min): six PyMC fits for Decision B, two Psi runs for Decision A.
"""
import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
warnings.filterwarnings("ignore")

from adaptivesft.ddm import ddm_p_correct, moc_ddm, simdiffT               # noqa: E402
from adaptivesft.models import d_numpy, fit_lnrm, make_data                # noqa: E402
from adaptivesft.psi import inv_pm_function, make_psi                      # noqa: E402
from adaptivesft.race import lnrm_random                                   # noqa: E402
from adaptivesft.salience import find_salience_ogival, targ_to_accuracy    # noqa: E402


def header(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ==============================================================================================
def decision_a(quick):
    header("DECISION A - what does `threshold` (a) mean?   psiSimulation_functions.R:96-99, 26MAR2019.R:117")
    a, v, sdv, ter = 1.45, 1.6, 0.25, 0.1
    thres50, xmax = 6.0, 50.0
    sc = lambda x: (x - thres50) / (xmax - thres50)
    print("\n  A1. The same script parameters, the two readings, evaluated at the colour range top (x = 50):")
    print(f"      {'reading':<12}{'separation given to simdiffT':>30}{'P(correct) closed form':>24}{'x for 99%':>11}{'x for 90%':>11}")
    for name, sep in (("separation", True), ("threshold", False)):
        p_top = ddm_p_correct(a, v, a_is_separation=sep)
        k = a if sep else 2 * a
        x99 = thres50 + np.log(99) / k / v * (xmax - thres50)
        x90 = thres50 + np.log(9) / k / v * (xmax - thres50)
        print(f"      {name:<12}{(a if sep else 2 * a):>30.2f}{p_top:>24.4f}{x99:>11.1f}{x90:>11.1f}")
    print("      author's numbers: 2018 run H = 101.6 (25JUNE2018.R:174); 2019 revision H = 50 (26MAR2019.R:208)")

    n_trials = 60 if quick else 300
    print(f"\n  A2. Psi calibration ({n_trials} trials, R grids) of a DDM observer under each reading, colour dimension:")
    print(f"      {'reading':<12}{'alpha-hat':>10}{'beta-hat':>10}{'H=inv.pm(.99)':>15}{'DDM acc at H':>13}{'L=inv.pm(.90)':>15}{'DDM acc at L':>13}{'in range?':>10}")
    for name, sep in (("separation", True), ("threshold", False)):
        rng = np.random.default_rng(1)
        psi = make_psi("colour")
        for _ in range(n_trials):
            _, r = simdiffT(1, a, sc(psi.next_intensity) * v, sdv, ter, rng=rng, a_is_separation=sep)
            psi.update(int(r[0]))
        al, be = psi.estimate()
        H, L = (float(inv_pm_function(p, al, be, psi.d)) for p in (0.99, 0.90))
        accH = simdiffT(2000, a, sc(H) * v, sdv, ter, rng=rng, a_is_separation=sep)[1].mean()
        accL = simdiffT(2000, a, sc(L) * v, sdv, ter, rng=rng, a_is_separation=sep)[1].mean()
        print(f"      {name:<12}{al:>10.2f}{be:>10.2f}{H:>15.1f}{accH:>13.3f}{L:>15.1f}{accL:>13.3f}{'yes' if H <= xmax else 'NO':>10}")
    print("      -> Both estimates are accurate for their own observer (DDM accuracy at H is ~.99 either way).")
    print("         What differs is whether H is a presentable stimulus: under 'threshold' the 99% point is at")
    print("         the range top (x ~ 50, the value the 2019 script hard-codes); under 'separation' it is at")
    print("         x ~ 93, outside the colour range (the 101.6 the 2018 run found).")


# ==============================================================================================
def decision_b(quick):
    header("DECISION B - the ogival LNRM: (1/2)L or L offset, L fixed or estimated, and what 8.0 / 1.3 mean")
    kw = dict(tune=300, draws=300, chains=2) if quick else dict(tune=800, draws=800, chains=4)
    truth = dict(mu=1.5, slope=2.0, midpoint=1.5, varZ=0.6, psi=0.12)
    L = 10.0
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 3.0, 400 if quick else 800)

    print("\n  B1. Simulate from each reading of the offset, fit each reading: does the model degenerate?")
    print("      data: 800 trials, slope 2, midpoint 1.5, mu 1.5, varZ 0.6, psi 0.12, L = 10")
    print(f"      {'simulated with':<16}{'accuracy':>9}{'median RT':>10} | {'fitted with':<14}{'slope-hat':>10}{'midpoint-hat':>13}{'varZ-hat':>9}{'divergences':>12}{'slope<=0 draws':>15}")
    fits = {}
    for gen_off in (0.5, 1.0):
        d = d_numpy("ogival", x, truth, L) * (2 * gen_off)          # d_numpy already has the 1/2
        rt, corr = lnrm_random(d, truth["mu"], truth["varZ"], truth["psi"], rng=rng)
        data = make_data(x, rt, corr)
        for fit_off in (0.5, 1.0):
            tr = fit_lnrm(data, link="ogival", L=L, ogival_offset=fit_off, random_seed=3, **kw)
            fits[(gen_off, fit_off)] = tr
            post = tr.posterior
            print(f"      {'(1/2)L' if gen_off == 0.5 else 'L':<16}{corr.mean():>9.3f}{np.median(rt):>10.3f} | "
                  f"{'(1/2)L' if fit_off == 0.5 else 'L':<14}{float(post['slope'].values.mean()):>10.3f}"
                  f"{float(post['midpoint'].values.mean()):>13.3f}{float(post['varZ'].values.mean()):>9.3f}"
                  f"{int(tr.sample_stats['diverging'].sum()):>12d}{float((post['slope'].values <= 0).mean()):>15.3f}")
    print("      -> with the full-L offset the simulated task is at ceiling (accuracy ~1, RT collapsed to psi),")
    print("         which is the degeneracy log.md reports; the (1/2)L reading gives an ordinary data set.")

    print("\n  B2. L fixed at 10 vs L estimated, on the (1/2)L data, and what the targets 8.0 / 1.3 become:")
    d = d_numpy("ogival", x, truth, L)
    rt, corr = lnrm_random(d, truth["mu"], truth["varZ"], truth["psi"], rng=rng)
    data = make_data(x, rt, corr)
    print(f"      {'L':<12}{'L-hat':>8}{'slope-hat':>10}{'varZ-hat':>9} | {'H (targ 8.0)':>13}{'reachable':>10}{'acc at H':>9} | {'L (targ 1.3)':>13}{'reachable':>10}{'acc at L':>9}")
    for Lopt in (10.0, "estimate"):
        tr = fit_lnrm(data, link="ogival", L=Lopt, random_seed=3, **kw)
        post = tr.posterior
        res = find_salience_ogival(tr, h_targ=8.0, l_targ=1.3)
        Lhat = float(post["L"].values.mean()) if "L" in post else 10.0
        varZ = post["varZ"].values.ravel()
        print(f"      {str(Lopt):<12}{Lhat:>8.2f}{float(post['slope'].values.mean()):>10.3f}{varZ.mean():>9.3f} | "
              f"{res['high']['intensity']:>13.3f}{1 - res['high']['dropped']:>10.1%}{float(np.median(targ_to_accuracy(8.0, varZ))):>9.4f} | "
              f"{res['low']['intensity']:>13.3f}{1 - res['low']['dropped']:>10.1%}{float(np.median(targ_to_accuracy(1.3, varZ))):>9.4f}")
    print("      calibrated intensity range: [0, 3].  targ 8.0 = z2 - z1 of 8 out of a maximum L = 10.")

    print("\n  B3. Separation targets <-> accuracy, at the varZ values seen in practice:")
    print(f"      {'varZ':>6}{'targ 8.0 -> acc':>17}{'targ 1.3 -> acc':>17}{'acc .99 -> targ':>17}{'acc .90 -> targ':>17}")
    from adaptivesft.salience import accuracy_to_targ
    for vz in (0.6, 0.95, 1.5):
        print(f"      {vz:>6.2f}{float(targ_to_accuracy(8.0, vz)):>17.4f}{float(targ_to_accuracy(1.3, vz)):>17.4f}"
              f"{float(accuracy_to_targ(0.99, vz)):>17.3f}{float(accuracy_to_targ(0.90, vz)):>17.3f}")
    print("      -> h_targ = 8.0 is ceiling accuracy whatever varZ is; the Psi branch's .99 / .90 are separations")
    print("         of about 2 / 1 - four times smaller than 8.0. The two branches are calibrating different things.")

    print("\n  B4. DDM observer of simulateLNRM_ogival.R (a = 3, v = 2): where do H and L land under (1/2)L, L = 10?")
    rng = np.random.default_rng(11)
    lv = np.linspace(-0.667, 1.0, 10)                                  # orientation, scaled (R :35-39)
    dat = moc_ddm(100, 3.0, 2.0, 0.1, 0.2, lv, rng=rng)
    tr = fit_lnrm(dat, link="ogival", L=L, random_seed=5, **kw)
    res = find_salience_ogival(tr, h_targ=8.0, l_targ=1.3)
    for name in ("high", "low"):
        xi = res[name]["intensity"]
        print(f"      {name}: scaled intensity {xi:.3f} (calibrated range [-0.67, 1.00]{', OUTSIDE' if not -0.667 <= xi <= 1 else ''}), "
              f"DDM P(correct) there = {ddm_p_correct(3.0, xi * 2.0):.4f}, dropped draws {res[name]['dropped']:.1%}")
    print("      -> the LNRM branch's H is outside the calibrated range for the same reason the Psi branch's .99 is:")
    print("         both ask for a level the observer only reaches beyond the stimulus range.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    decision_a(args.quick)
    decision_b(args.quick)
    print()
