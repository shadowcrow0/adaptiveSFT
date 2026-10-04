"""
對答案：python learn/check.py ex1 [ex2 ...] | all
只呼叫你在 exN_*.py 裡寫的函式、比數字。要看哪裡沒對上就讀印出來的表。
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTENSOR_FLAGS", "warn__ignore_bug_before=all")


def ok(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    return bool(cond)


def sample(model, **kw):
    import pymc as pm
    with model:
        return pm.sample(progressbar=False, random_seed=1, **kw)


def mean(trace, name):
    return float(trace.posterior[name].values.mean())


def as_numpy(v):
    """pytensor 變數就 .eval()，numpy 就原樣。"""
    if hasattr(v, "eval"):
        return np.asarray(v.eval(), dtype=float)
    return np.asarray(v, dtype=float)


# ------------------------------------------------------------------------------------ ex1
def check_ex1():
    from learn.ex1_normal import build_model
    rng = np.random.default_rng(0)
    y = rng.normal(2.0, 1.5, 200)
    tr = sample(build_model(y), draws=500, tune=500, chains=2)
    print(f"  mu    posterior mean {mean(tr, 'mu'):.3f}   truth 2.0   sample mean {y.mean():.3f}")
    print(f"  sigma posterior mean {mean(tr, 'sigma'):.3f}   truth 1.5   sample sd   {y.std(ddof=1):.3f}")
    return ok(abs(mean(tr, "mu") - y.mean()) < 0.15, "mu 在樣本平均 ±0.15") & \
        ok(abs(mean(tr, "sigma") - y.std(ddof=1)) < 0.15, "sigma 在樣本 SD ±0.15")


# ------------------------------------------------------------------------------------ ex2
def check_ex2():
    from learn.ex2_shifted_lognormal import build_model
    rng = np.random.default_rng(1)
    psi, mu, sigma = 0.2, -0.5, 0.4
    rt = psi + np.exp(rng.normal(mu, sigma, 400))
    tr = sample(build_model(rt), draws=800, tune=800, chains=2)
    res = True
    for name, truth, tol in (("psi", psi, 0.05), ("mu", mu, 0.1), ("sigma", sigma, 0.08)):
        m = mean(tr, name)
        print(f"  {name:6s} posterior mean {m:.3f}   truth {truth}")
        res &= ok(abs(m - truth) < tol, f"{name} 在真值 ±{tol}")
    return res


# ------------------------------------------------------------------------------------ ex3
def check_ex3():
    from adaptivesft.race import lnrm_pointwise_loglik
    from learn.ex3_race import race_logp
    rng = np.random.default_rng(2)
    rt = 0.3 + np.exp(rng.normal(1.0, 0.5, 50))
    correct = (rng.uniform(size=50) < 0.7).astype("int32")
    d = rng.uniform(0, 2, 50)
    res = True
    for mu, sigma, psi in ((1.5, 0.6, 0.12), (0.5, 0.3, 0.25), (2.0, 1.0, 0.0)):
        mine = as_numpy(race_logp(rt, correct, mu - d, mu + d, sigma, psi))
        ref = lnrm_pointwise_loglik(rt, correct, d, mu, sigma, psi)
        err = float(np.max(np.abs(mine - ref)))
        print(f"  mu={mu} sigma={sigma} psi={psi}: max |yours − reference| = {err:.2e}   (sum yours {mine.sum():.4f}, ref {ref.sum():.4f})")
        res &= ok(err < 1e-6, "逐題 logp 差 < 1e−6")
    # 尾端：很慢的 RT 不能變成 -inf / nan
    big = as_numpy(race_logp(np.array([50.0, 200.0]), np.array([1, 0], dtype="int32"),
                             np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.3, 0.0))
    res &= ok(np.all(np.isfinite(big)), f"尾端有限：{big}")
    return res


# ------------------------------------------------------------------------------------ ex4
_EX4_TRACE = {}


def check_ex4():
    import arviz as az
    from learn.ex4_lnrm2 import build_model
    d = np.genfromtxt(os.path.join(ROOT, "tests", "data", "lnrm_oracle_input.csv"), delimiter=",", names=True)
    rt, correct, x = d["rt"], d["correct"].astype("int32"), d["intensity"]
    with open(os.path.join(ROOT, "tests", "data", "lnrm_stan_oracle.json")) as f:
        oracle = json.load(f)
    stan = {s["name"]: s for s in oracle["summary"]}
    t0 = time.time()
    tr = sample(build_model(rt, correct, x), draws=1000, tune=1000, chains=4, target_accept=0.9)
    print(f"  sampled in {time.time() - t0:.0f} s; divergences {int(tr.sample_stats['diverging'].sum())}")
    _EX4_TRACE["trace"] = tr
    rhat = az.rhat(tr)
    res = True
    print(f"  {'param':7s} {'yours':>8s} {'Stan':>8s} {'Stan sd':>8s} {'|diff|/sd':>9s} {'rhat':>6s}")
    for name in ("mu", "alpha", "alpha2", "varZ", "psi"):
        m, s_m, s_sd = mean(tr, name), stan[name]["mean"], stan[name]["sd"]
        z = abs(m - s_m) / s_sd
        r = float(rhat[name].values)
        print(f"  {name:7s} {m:8.4f} {s_m:8.4f} {s_sd:8.4f} {z:9.3f} {r:6.3f}")
        res &= ok(z < 0.35 and r < 1.05, f"{name}: 與 Stan 的差 < 0.35 SD 且 rhat < 1.05")
    return res


# ------------------------------------------------------------------------------------ ex5
def check_ex5():
    from learn.ex5_salience import invert
    with open(os.path.join(ROOT, "tests", "data", "lnrm_stan_oracle.json")) as f:
        oracle = json.load(f)
    stan = {s["name"]: s["mean"] for s in oracle["summary"]}
    sal = oracle["salience"]
    res = True
    # 1. 閉式：在 (α, α₂) = (1, −0.1)，2(x − 0.1x²) = 1.6 的較小根 = (10 − √(100 − 32)) / 2 … 直接代數驗證
    x = invert(np.array([1.0]), np.array([-0.1]), 1.6)
    back = 2 * (1.0 * x - 0.1 * x ** 2)
    print(f"  invert(1, −0.1, 1.6) = {float(x[0]):.4f}; 代回 2·d(x) = {float(back[0]):.4f}（應為 1.6）")
    res &= ok(abs(float(back[0]) - 1.6) < 1e-9 and float(x[0]) < 5.0, "代回去等於 targ，而且取較小的根")
    # 2. 對 R 在 Stan draws 上算的平均
    tr = _EX4_TRACE.get("trace")
    if tr is not None:
        a = tr.posterior["alpha"].values.ravel()
        a2 = tr.posterior["alpha2"].values.ravel()
        hi, lo = np.nanmean(invert(a, a2, sal["h_targ"])), np.nanmean(invert(a, a2, sal["l_targ"]))
        print(f"  你的 trace：high {hi:.4f} low {lo:.4f}；R 在 Stan draws 上：high {sal['high']:.4f} low {sal['low']:.4f}")
        res &= ok(abs(hi - sal["high"]) < 0.05 and abs(lo - sal["low"]) < 0.05, "H / L 與 R 的差 < 0.05")
    else:
        hi = float(invert(np.array([stan["alpha"]]), np.array([stan["alpha2"]]), sal["h_targ"])[0])
        print(f"  在 Stan 的後驗平均上：high {hi:.4f}（R 逐 draw 平均 {sal['high']:.4f}；先跑 ex4 才有逐 draw 的比較）")
        res &= ok(abs(hi - sal["high"]) < 0.1, "後驗平均上的反解接近 R 的逐 draw 平均")
    return res


CHECKS = {"ex1": check_ex1, "ex2": check_ex2, "ex3": check_ex3, "ex4": check_ex4, "ex5": check_ex5}


def main(argv):
    names = list(CHECKS) if (not argv or argv == ["all"]) else argv
    results = {}
    for n in names:
        if n not in CHECKS:
            print(f"unknown: {n}; choose from {list(CHECKS)}")
            return 2
        print(f"== {n}")
        try:
            results[n] = CHECKS[n]()
        except NotImplementedError as e:
            print(f"  TODO  {e}")
            results[n] = False
    print("\n" + "  ".join(f"{n}: {'PASS' if v else 'FAIL'}" for n, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
