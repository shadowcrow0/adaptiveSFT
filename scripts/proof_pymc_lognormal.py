"""
proof_pymc_lognormal.py —— 證明 adaptivesft/models_v2.py 的賽跑 likelihood 是用 PyMC 內建的
pm.LogNormal（logp / logcdf）拼出來的，不是手寫公式。

    PyMC 沒有 "LogNormalRace" 分布。賽跑模型 = 贏家的 LogNormal log-density
                                        + 輸家的 LogNormal log-survival
    v2 兩項都從 pm.LogNormal.dist 取得（pm.logp、pm.logcdf + pm.math.log1mexp），再包進 pm.CustomDist。

五種證據（a 原始碼、b 計算圖、c 數值、d 模型物件、e 端到端擬合）；任一數值檢查失敗就以非零碼結束。
用法（repo 根目錄）：  python scripts/proof_pymc_lognormal.py [--skip-fit]
"""
import argparse
import inspect
import io
import json
import logging
import os
import re
import sys
import time
import warnings
from contextlib import redirect_stdout

warnings.filterwarnings("ignore")                      # 把 PyMC / ArviZ 的 warning 擋在報告外
import numpy as np                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pymc as pm                                      # noqa: E402
import pytensor                                        # noqa: E402
from pytensor.graph import ancestors                   # noqa: E402
from scipy.stats import lognorm                        # noqa: E402

from adaptivesft import models as v1                   # noqa: E402
from adaptivesft import models_v2 as v2                # noqa: E402
from adaptivesft.models_v2 import build_model, fit_lnrm_v2, make_data, race_logp  # noqa: E402

logging.getLogger("pymc").setLevel(logging.ERROR)
V2_PATH = os.path.join(ROOT, "adaptivesft", "models_v2.py")
V1_PATH = os.path.join(ROOT, "adaptivesft", "models.py")
DATA = os.path.join(ROOT, "tests", "data")
TOL = 1e-6
FAILURES = []


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def check(ok, msg):
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        FAILURES.append(msg)


# ---------------------------------------------------------------- (a) 原始碼
def source_evidence():
    section("(a) SOURCE EVIDENCE — adaptivesft/models_v2.py, read at runtime")
    with open(V2_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    tokens = ["pm.LogNormal.dist", "pm.logp(", "pm.logcdf(", "pm.math.log1mexp", "pm.CustomDist"]
    print(f"  file: {V2_PATH}")
    found = {t: [] for t in tokens}
    for i, line in enumerate(lines, 1):
        for t in tokens:
            if t in line and not line.lstrip().startswith(("#", "log 密度", "log 存活", "接進模型")):
                found[t].append(i)
    for t in tokens:
        print(f"\n  {t!r} at line(s) {found[t]}")
        for i in found[t]:
            print(f"    {i:4d}: {lines[i - 1].rstrip()}")
        check(len(found[t]) > 0, f"models_v2.py calls {t}")
    code_lines = [l for l in lines[18:] if not l.lstrip().startswith("#")]  # 跳過 docstring（它刻意對照 v1 的公式）
    code = "\n".join(code_lines)
    v1_code = open(V1_PATH, encoding="utf-8").read()
    hand = {"erfc": r"\berfc\b", "½·ln(2π) literal 0.9189…": r"0\.9189", "√2 literal 1.41421…": r"1\.4142",
            "log(2*pi)": r"log\(2\s*\*\s*(np\.)?pi\)"}
    print("\n  Negative evidence — hand-written lognormal pieces present in v1 but absent in v2 code:")
    for name, pat in hand.items():
        in1, in2 = bool(re.search(pat, v1_code)), bool(re.search(pat, code))
        print(f"    {name:<28} v1(models.py): {'yes' if in1 else 'no ':<3}   v2(models_v2.py code): {'yes' if in2 else 'no'}")
        check(not in2, f"v2 code has no hand-written {name}")
    print("\n  Where the formula actually lives (PyMC's own source, not this repo):")
    print(f"    {inspect.getsourcefile(pm.LogNormal)}")
    for fn in (pm.LogNormal.logp, pm.LogNormal.logcdf):
        src, start = inspect.getsourcelines(fn)
        print(f"    pm.LogNormal.{fn.__name__}  (lines {start}-{start + len(src) - 1}):")
        for j, l in enumerate(src):
            print(f"      {start + j:5d}: {l.rstrip()}")


# ---------------------------------------------------------------- 小例子
def example(n=10, seed=11):                       # seed=11 給 10 題裡 3 題答錯，兩個分支都測到
    rng = np.random.default_rng(seed)
    x = rng.choice([0.1, 0.3, 0.6, 1.0, 1.5], n)
    mu, sigma, psi, alpha, alpha2 = 1.5, 0.6, 0.12, 0.8, -0.15
    d = alpha * x + alpha2 * x ** 2
    t1 = psi + np.exp(rng.normal(mu - d, sigma))
    t2 = psi + np.exp(rng.normal(mu + d, sigma))
    rt = np.minimum(t1, t2)
    correct = (t1 <= t2).astype(float)
    return dict(rt=rt, correct=correct, d=d, mu=mu, sigma=sigma, psi=psi, x=x)


# ---------------------------------------------------------------- (b) 計算圖
def graph_evidence(ex):
    section("(b) GRAPH EVIDENCE — race_logp(...) graph and instrumented pm.LogNormal.dist")
    counts = {"pm.LogNormal.dist": 0, "pm.logp": 0, "pm.logcdf": 0, "pm.math.log1mexp": 0}
    dist_mu = []
    orig_dist = pm.LogNormal.__dict__["dist"]          # classmethod 物件
    orig_logp, orig_logcdf, orig_l1me = pm.logp, pm.logcdf, pm.math.log1mexp

    def dist_wrapper(cls, *a, **k):
        counts["pm.LogNormal.dist"] += 1
        dist_mu.append(k.get("mu", a[0] if a else None))
        return orig_dist.__func__(cls, *a, **k)

    def count(name, f):
        def w(*a, **k):
            counts[name] += 1
            return f(*a, **k)
        return w

    pm.LogNormal.dist = classmethod(dist_wrapper)
    pm.logp, pm.logcdf, pm.math.log1mexp = count("pm.logp", orig_logp), count("pm.logcdf", orig_logcdf), count("pm.math.log1mexp", orig_l1me)
    try:
        v = race_logp(ex["rt"], ex["mu"], ex["d"], ex["sigma"], ex["psi"], ex["correct"])
    finally:
        pm.LogNormal.dist = orig_dist
        pm.logp, pm.logcdf, pm.math.log1mexp = orig_logp, orig_logcdf, orig_l1me
    print("  Calls made inside ONE race_logp(...) call:")
    for k_, n_ in counts.items():
        print(f"    {k_:<20} {n_}")
    check(counts["pm.LogNormal.dist"] == 2, "pm.LogNormal.dist called exactly twice (winner, loser)")
    check(counts["pm.logp"] == 1 and counts["pm.logcdf"] == 1 and counts["pm.math.log1mexp"] == 1,
          "pm.logp, pm.logcdf, pm.math.log1mexp each called once")
    c = ex["correct"]
    z_win = ex["mu"] - ex["d"] * (2 * c - 1)
    z_lose = ex["mu"] + ex["d"] * (2 * c - 1)
    mu1, mu2 = (np.asarray(m.eval()) for m in dist_mu)
    print(f"    1st LogNormal.dist mu (first 4) = {np.round(mu1[:4], 4)}  == z_win  : {np.allclose(mu1, z_win)}")
    print(f"    2nd LogNormal.dist mu (first 4) = {np.round(mu2[:4], 4)}  == z_lose : {np.allclose(mu2, z_lose)}")
    check(np.allclose(mu1, z_win) and np.allclose(mu2, z_lose), "1st dist is the winner, 2nd dist is the loser")

    print("\n  Op classes in the graph (pytensor.graph.ancestors), with defining module:")
    ops = {}
    for a in ancestors([v]):
        if a.owner is not None:
            op = a.owner.op
            key = f"{type(op).__name__:<22} {type(op).__module__}"
            if type(op).__name__ == "Elemwise":
                key = f"{'Elemwise(' + type(op.scalar_op).__name__ + ')':<22} {type(op.scalar_op).__module__}"
            ops[key] = ops.get(key, 0) + 1
    for k_ in sorted(ops):
        print(f"    {ops[k_]:3d} × {k_}")
    pymc_ops = [a.owner.op for a in ancestors([v]) if a.owner is not None and type(a.owner.op).__module__.startswith("pymc")]
    desc = [f"{type(o).__name__}(msg={getattr(o, 'msg', None)!r})" for o in pymc_ops]
    print(f"\n  PyMC-specific ops: {desc}")
    check(len(pymc_ops) == 2 and all(getattr(o, "msg", "") == "sigma > 0" for o in pymc_ops),
          "two Check{sigma > 0} nodes (pymc.logprob.utils.CheckParameterValue) = LogNormal.logp + LogNormal.logcdf")
    names = {k_.split()[0] for k_ in ops}
    for want in ("Elemwise(Log1mexp)", "Elemwise(Erfc)", "Elemwise(Erfcx)", "Elemwise(Log)"):
        check(want in names, f"graph contains {want}")
    print("\n  pytensor.dprint(race_logp(...)):")
    buf = io.StringIO()
    with redirect_stdout(buf):
        pytensor.dprint(v, print_type=False)
    for line in buf.getvalue().splitlines():
        print("    " + line)


# ---------------------------------------------------------------- (c) 數值
def numeric_evidence(ex):
    section("(c) NUMERICAL EVIDENCE — five implementations on 10 trials")
    rt, c, d, mu, s, psi = ex["rt"], ex["correct"], ex["d"], ex["mu"], ex["sigma"], ex["psi"]
    t = rt - psi
    z_win = mu - d * (2 * c - 1)
    z_lose = mu + d * (2 * c - 1)
    # (i) v2
    a = race_logp(rt, mu, d, s, psi, c).eval()
    # (ii) 直接用 PyMC 內建物件（本腳本自己寫）
    b = (pm.logp(pm.LogNormal.dist(mu=z_win, sigma=s), t)
         + pm.math.log1mexp(pm.logcdf(pm.LogNormal.dist(mu=z_lose, sigma=s), t))).eval()
    # (iii) scipy（獨立函式庫）
    sc = lognorm(s=s, scale=np.exp(z_win)).logpdf(t) + lognorm(s=s, scale=np.exp(z_lose)).logsf(t)
    # (iv) v1 手寫 erfc
    v1_ = v1._race_logp_pt(rt, c.astype("int32"), d, mu, s, psi).eval()
    # (v) Stan 語句逐字翻譯：z[1]=mu-d, z[2]=mu+d, if (correct) ... else ...
    z1, z2 = mu - d, mu + d
    lpdf = lambda y, m: lognorm(s=s, scale=np.exp(m)).logpdf(y)     # lognormal_lpdf(y | m, varZ)
    lccdf = lambda y, m: lognorm(s=s, scale=np.exp(m)).logsf(y)     # lognormal_lccdf(y | m, varZ)
    st = np.where(c == 1, lpdf(rt - psi, z1) + lccdf(rt - psi, z2), lpdf(rt - psi, z2) + lccdf(rt - psi, z1))
    print(f"  mu={mu} sigma={s} psi={psi}; d = 0.8x - 0.15x^2")
    hdr = f"  {'rt':>8} {'corr':>4} {'x':>4} | {'(i) v2':>12} {'(ii) PyMC':>12} {'(iii) scipy':>12} {'(iv) v1':>12} {'(v) Stan':>12}"
    print(hdr)
    for i in range(len(rt)):
        print(f"  {rt[i]:8.4f} {int(c[i]):4d} {ex['x'][i]:4.1f} | {a[i]:12.6f} {b[i]:12.6f} {sc[i]:12.6f} {v1_[i]:12.6f} {st[i]:12.6f}")
    print()
    for name, other in (("(ii) direct PyMC", b), ("(iii) scipy", sc), ("(iv) v1 erfc", v1_), ("(v) Stan/scipy", st)):
        diff = float(np.max(np.abs(a - other)))
        print(f"  max |(i) v2 − {name:<17}| = {diff:.3e}")
        check(diff <= TOL, f"v2 vs {name} within {TOL:g}")
    neg = race_logp(np.array([0.05]), mu, np.array([0.1]), s, psi, np.array([1.0])).eval()
    print(f"  rt <= psi guard: race_logp(rt=0.05, psi={psi}) = {neg[0]}")
    check(np.isneginf(neg[0]), "rt <= psi gives -inf (switch guard)")


# ---------------------------------------------------------------- (d) 模型
def model_evidence(data):
    section("(d) MODEL EVIDENCE — build_model(...)")
    model, info = build_model(data)
    print(f"  model.basic_RVs    = {model.basic_RVs}")
    print(f"  model.free_RVs     = {model.free_RVs}")
    print(f"  model.observed_RVs = {model.observed_RVs}")
    obs = model.observed_RVs[0]
    op = obs.owner.op
    mro = [f"{k.__module__}.{k.__name__}" for k in type(op).__mro__[:3]]
    print(f"  obs.owner.op class = {type(op).__name__}; MRO = {mro}")
    check(any("CustomDistRV" in m for m in mro), "observed RV 'obs' is a pymc CustomDistRV")
    val_var = model.rvs_to_values[obs]
    print(f"  obs observed data: {type(val_var).__name__}, shape = {np.asarray(getattr(val_var, 'data', val_var.eval())).shape}")
    logp_fn = model.compile_logp()
    ip0 = model.initial_point()
    print(f"  logp at PyMC default initial point      = {logp_fn(ip0):.4f}   (varZ=0.05 → loser's logcdf → 0, log1mexp(0) = -inf)")
    for k_, v_ in info["initvals"].items():
        model.set_initval(model[k_], v_)
    ip = model.initial_point()
    val = float(logp_fn(ip))
    print(f"  initvals used by fit_lnrm_v2            = {info['initvals']}")
    print(f"  transformed initial point               = { {k: float(v) for k, v in ip.items()} }")
    print(f"  model.logp() at that point              = {val:.4f}")
    check(np.isfinite(val), "model.logp() compiles and is finite at the sampling initial point")
    print(f"  per-variable: {model.point_logps(ip)}")
    try:
        import graphviz  # noqa: F401
        print("  (graphviz importable; pm.model_to_graphviz skipped on purpose)")
    except ImportError:
        print("  (graphviz not importable; skipped, nothing installed)")


# ---------------------------------------------------------------- (e) 端到端
def end_to_end(data):
    section("(e) END-TO-END — fit_lnrm_v2 on tests/data/lnrm_oracle_input.csv vs Stan oracle")
    with open(os.path.join(DATA, "lnrm_stan_oracle.json"), encoding="utf-8") as f:
        o = json.load(f)
    t0 = time.time()
    idata = fit_lnrm_v2(data, tune=500, draws=500, chains=2, random_seed=1)
    secs = time.time() - t0
    import arviz as az
    rhat = az.rhat(idata, var_names=[r["name"] for r in o["summary"]]).to_array().values
    print(f"  NUTS tune=500 draws=500 chains=2: {secs:.0f} s; divergences={int(idata.sample_stats['diverging'].sum())}; max r̂={rhat.max():.3f}")
    print(f"  Stan oracle: {o['package']} / Stan {o['stan_version']}, {o['chains']} chains × {o['iter']} iter, N={o['n_trials']}")
    print(f"  {'param':<7} {'truth':>8} {'v2 mean':>10} {'v2 sd':>8} {'Stan mean':>10} {'Stan sd':>8} {'|diff|/sd':>10}")
    for row in o["summary"]:
        v = idata.posterior[row["name"]].values.ravel()
        z = abs(v.mean() - row["mean"]) / row["sd"]
        print(f"  {row['name']:<7} {o['truth'][row['name']]:8.3f} {v.mean():10.4f} {v.std():8.4f} {row['mean']:10.4f} {row['sd']:8.4f} {z:10.3f}")
        check(z < 0.5, f"{row['name']}: |v2 mean − Stan mean| / Stan sd = {z:.3f} < 0.5")
    check("obs" in idata.log_likelihood, "idata.log_likelihood['obs'] present (PyMC built-in, from CustomDist logp)")


def main():
    global TOL
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--skip-fit", action="store_true", help="skip section (e), the ~1 min NUTS fit")
    ap.add_argument("--tol", type=float, default=TOL, help="tolerance for section (c); e.g. --tol 1e-12 to see a non-zero exit")
    args = ap.parse_args()
    TOL = args.tol
    print(f"PyMC {pm.__version__}, PyTensor {pytensor.__version__}, numpy {np.__version__}")
    print("Claim: models_v2.race_logp = pm.logp(pm.LogNormal.dist(z_win, σ), t) + log1mexp(pm.logcdf(pm.LogNormal.dist(z_lose, σ), t))")
    print("       (PyMC has no race distribution; only LogNormal is built in — the race is assembled from it.)")
    source_evidence()
    ex = example()
    graph_evidence(ex)
    numeric_evidence(ex)
    raw = np.loadtxt(os.path.join(DATA, "lnrm_oracle_input.csv"), delimiter=",", skiprows=1)
    data = make_data(raw[:, 0], raw[:, 1], raw[:, 2])
    model_evidence(data)
    if not args.skip_fit:
        end_to_end(data)
    section("RESULT")
    if FAILURES:
        print(f"  {len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print("   - " + f)
        sys.exit(1)
    print("  all checks passed")


if __name__ == "__main__":
    main()
