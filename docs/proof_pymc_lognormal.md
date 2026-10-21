# Proof: `models_v2.py` builds the race likelihood from PyMC's built-in `LogNormal`

**Claim.** `adaptivesft/models_v2.py` does not hand-write the lognormal race likelihood.
Both terms of the per-trial log-likelihood come from `pm.LogNormal.dist(...)`:

```
ln L(trial) = ln f(t; z_win, σ)        ← pm.logp(   pm.LogNormal.dist(z_win,  σ), t)
            + ln S(t; z_lose, σ)       ← pm.math.log1mexp(pm.logcdf(pm.LogNormal.dist(z_lose, σ), t))
t = rt − ψ,  z_win/z_lose = μ ∓ d (correct) or μ ± d (error)
```

The result is handed to `pm.CustomDist(..., logp=race_logp, random=race_random, observed=rt)`. Script: `scripts/proof_pymc_lognormal.py` (exit 1 if any numerical check fails). Output below is from one run
(PyMC 5.28.5, PyTensor 2.38.3), trimmed; the script suppresses PyMC/ArviZ warnings.

## (a) Source evidence — read from the file at runtime

```
  'pm.LogNormal.dist' at line(s) [36, 37]
      36:     log_pdf = pm.logp(pm.LogNormal.dist(mu=z_win, sigma=sigma), t)
      37:     log_sf = pm.math.log1mexp(pm.logcdf(pm.LogNormal.dist(mu=z_lose, sigma=sigma), t))
  'pm.CustomDist' at line(s) [113]
     113:         pm.CustomDist("obs", p["mu"], d, p["varZ"], p["psi"], c_t, logp=race_logp, random=race_random, observed=rt)

  Negative evidence — hand-written lognormal pieces present in v1 but absent in v2 code:
    erfc / ½·ln(2π) literal 0.9189… / √2 literal 1.41421…     v1(models.py): yes   v2(models_v2.py code): no

  Where the formula actually lives (PyMC's own source, not this repo):
    .../site-packages/pymc/distributions/continuous.py
    pm.LogNormal.logp   (lines 1734-1746): -0.5*((log(value)-mu)/sigma)^2 - 0.5*log(2π) - log(sigma) - log(value) + check_parameters(sigma > 0)
    pm.LogNormal.logcdf (lines 1748-1759): switch(value <= 0, -inf, normal_lcdf(mu, sigma, log(value)))      + check_parameters(sigma > 0)
```

## (b) Graph evidence — instrumented calls and the PyTensor graph

`pm.LogNormal.dist`, `pm.logp`, `pm.logcdf`, `pm.math.log1mexp` were wrapped with counters for one `race_logp(...)` call:

```
  Calls made inside ONE race_logp(...) call:
    pm.LogNormal.dist    2
    pm.logp              1
    pm.logcdf            1
    pm.math.log1mexp     1
    1st LogNormal.dist mu (first 4) = [1.4215 1.4215 0.85   1.074 ]  == z_win  : True
    2nd LogNormal.dist mu (first 4) = [1.5785 1.5785 2.15   1.926 ]  == z_lose : True
```

Ops in the graph (`pytensor.graph.ancestors`): the only PyMC-defined op is `CheckParameterValue` (`Check{sigma > 0}`,
`pymc.logprob.utils`, 2 of them), inserted by `check_parameters` inside `LogNormal.logp` and `LogNormal.logcdf`;
`Erfc`/`Erfcx`/`Log1p` come from PyMC's `normal_lcdf`, `Log1mexp` from `pm.math.log1mexp`.
`pytensor.dprint(race_logp(...))`, trimmed to the structural lines:

```
    Switch [id A]                              ← pt.switch(t > 0, ..., -inf)   (models_v2.py:38)
     ├─ Add [id C]                             ← log_pdf + log_sf
     │  ├─ Check{sigma > 0} [id D]             ← pm.logp(LogNormal.dist(z_win, σ), t)
     │  │  ├─ Switch [id E]  ... Pow / Log / Mul 0.5·Log(6.283…) ...   (LogNormal.logp body)
     │  └─ Log1mexp [id BZ]                    ← pm.math.log1mexp(...)
     │     └─ Check{sigma > 0} [id CA]         ← pm.logcdf(LogNormal.dist(z_lose, σ), t)
     │        ├─ Switch [id CB] ── Le (value <= 0 → -inf)
     │        │     ├─ Sub ── Log ── True_div ── Erfcx [id DF]   (normal_lcdf, far-tail branch)
     │        │     └─ Log1p ── True_div ── Neg ── Erfc [id DU]  (normal_lcdf, main branch)
```

## (c) Numerical evidence — five implementations, 10 trials (3 errors)

(i) v2 `race_logp`; (ii) the same PyMC expression written directly in the script; (iii) `scipy.stats.lognorm` logpdf + logsf;
(iv) v1 `_race_logp_pt` (hand-written erfc); (v) Stan's `lognormal_lpdf(rt−ψ | z, varZ) + lognormal_lccdf(...)` with its `if (correct)` branching, via scipy.

```
        rt corr    x |       (i) v2    (ii) PyMC  (iii) scipy      (iv) v1     (v) Stan
    3.1394    1  0.1 |    -1.894373    -1.894373    -1.894373    -1.894373    -1.894373
    5.2126    0  1.0 |    -4.743260    -4.743260    -4.743260    -4.743260    -4.743260
    1.6516    0  0.1 |    -2.728074    -2.728074    -2.728074    -2.728074    -2.728074
  max |(i) v2 − (ii) direct PyMC | = 0.000e+00      max |(i) v2 − (iii) scipy    | = 3.921e-08
  max |(i) v2 − (iv) v1 erfc     | = 3.730e-08      max |(i) v2 − (v) Stan/scipy | = 3.921e-08
  rt <= psi guard: race_logp(rt=0.05, psi=0.12) = -inf
```

## (d) Model evidence — `build_model(...)`

```
  model.basic_RVs    = [mu, varZ, psi, alpha, alpha2, obs]
  model.observed_RVs = [obs]
  obs.owner.op class = CustomDist_obs; MRO = ['abc.CustomDist_obs', 'pymc.distributions.custom.CustomDistRV', ...]
  logp at PyMC default initial point      = -inf   (varZ=0.05 → loser's logcdf → 0, log1mexp(0) = -inf)
  initvals used by fit_lnrm_v2            = {'mu': 1.0, 'varZ': 0.5, 'psi': 0.1112359499196, 'alpha': 0.5, 'alpha2': -0.1}
  model.logp() at that point              = -2167.1072
```

The `-inf` at PyMC's *default* point is expected (σ = 0.05 rounds the loser's CDF to 1); `fit_lnrm_v2` always passes `initvals`.

## (e) End-to-end — `fit_lnrm_v2` on `tests/data/lnrm_oracle_input.csv` vs the Stan oracle

```
  NUTS tune=500 draws=500 chains=2: 16 s; divergences=0; max r̂=1.018   (Stan oracle: cmdstanr 0.9.0, 4 × 4000 iter, N=1000)
  param      truth    v2 mean    v2 sd  Stan mean  Stan sd  |diff|/sd
  mu         1.500     1.5481   0.0294     1.5472   0.0292      0.032
  alpha      0.800     0.7467   0.0515     0.7429   0.0526      0.073
  alpha2    -0.150    -0.1186   0.0195    -0.1170   0.0196      0.081
  varZ       0.600     0.6248   0.0240     0.6239   0.0241      0.038
  psi        0.120     0.1132   0.0510     0.1137   0.0508      0.008
  all checks passed
```

## What is and is not built in

PyMC ships no race / "LogNormalRace" distribution, so the race itself cannot be a single built-in call. What *is* built in,
and what v2 uses, is `pymc.LogNormal` (https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LogNormal.html):
its `logp` (the density, incl. the `value > 0` and `sigma > 0` guards) and its `logcdf` (via PyMC's `normal_lcdf`, which
switches between `erfc` and the `erfcx` tail expansion). v2 adds only the race algebra — choose `z_win`/`z_lose` from `correct`,
shift `t = rt − ψ`, survival = `log1mexp(logcdf)`, sum the two terms — and registers the result with `pm.CustomDist`, which is
also what gives `idata.log_likelihood["obs"]` and `pm.sample_posterior_predictive` for free. v1 (`models.py:_race_logp_pt`)
writes the same two terms by hand (`−ln t − ln σ − ½ln2π − …` and `ln(½·erfc(…))`); (c) shows they agree to ~4e−8.

## Re-run

```
python scripts/proof_pymc_lognormal.py              # from the repo root; full report ≈1 min (NUTS fit); --skip-fit for (a)-(d) only
python scripts/proof_pymc_lognormal.py --tol 1e-12 --skip-fit; echo $?   # demonstrates the non-zero exit
```
