# Issue: `model_lnrm2a.py` degenerates at `L = 10`

Status: **resolved in code, pending the author's confirmation.**
Question 1 below (TODO-A1) turned out to be the whole problem. `ogival_d`
now returns `0.5 · L · inv_logit(...)` (`model_lnrm2a.py`), matching
`simulateLNRM_ogival.R:206-209`. Recovery at `L = 10`, N = 1000,
DEMetropolisZ 8 chains × 3000:

```
          true   mean     sd  hdi_3%  hdi_97%  ess_bulk  r_hat
mu        1.50  1.392  0.035   1.325    1.456    1305.0   1.01
slope     2.00  1.931  0.045   1.844    2.017    1394.0   1.01
midpoint  1.50  1.538  0.022   1.497    1.580    1266.0   1.00
varZ      0.60  0.613  0.015   0.587    0.641    1386.0   1.00
psi       0.12  0.121  0.001   0.118    0.123    1477.0   1.00
```

(`d ∈ [0.24, 4.76]`, accuracy 96.5 %, median RT 0.50 s.) The guess in
question 1 that halving "alone may not close the gap" was wrong: it does.
Questions 2–5 remain open but no longer block anything. Everything below is
the record as written before the fix; line numbers refer to the pre-fix file
where they cite `model_lnrm2a.py`.

## Issue

`model_lnrm2a.py` is a PyMC rebuild of the missing `lnrm2a.stan`
(`model_lnrm2a.py:4` — the file "does not exist since the repo's first
commit"), built by taking the verified `model_lnrm2.py` (port of
`lnrm2.stan`) and replacing only the intensity→difficulty curve, quadratic
`d = α·x + α₂·x²` → ogival `d = L/(1+exp(−slope·(x−midpoint)))`, `L = 10`,
while keeping the race structure `z₁ = μ − d`, `z₂ = μ + d` from
`lnrm2.stan:23-24` unchanged (`model_lnrm2a.py:76-78`). With `L = 2` this
recovers all five parameters cleanly. With `L = 10` — the value the R side
actually uses (`simulateLNRM_ogival.R:26`) — the simulated data are
degenerate (99% accuracy, RT collapsed near `psi`) and the sampler cannot
recover `slope`/`midpoint` at all (R-hat > 2). So the "swap only the `d`
curve" hypothesis in the docstring (`model_lnrm2a.py:13-16`) is not
sufficient by itself to explain `lnrm2a.stan`; something else about the
`z = μ ∓ d` construction must differ too.

## Reproduction

Script: `scratchpad/logagent/repro.py` (this session's scratch dir), calling
`lnrm2a_random` + `build_model_lnrm2a` from `model_lnrm2a.py` with
`{mu: 1.5, slope: 2.0, midpoint: 1.5, varZ: 0.6, psi: 0.12}`, `N = 1000`,
seed 42, `tune=2000, draws=2000, chains=8`. RuntimeWarnings about overflow
in `exp` (from `ogival_d`/`lnrm2a_random` evaluating `exp(9.5)` etc. at
`L=10`) are expected and ignored.

```
L = 2.0
  d min/max = 0.096 / 1.905
  accuracy  = 0.898
  median rt = 1.6321   min rt = 0.2380

          true   mean     sd  hdi_3%  hdi_97%  ess_bulk  r_hat
mu        1.50  1.422  0.028   1.370    1.475     624.0   1.02
slope     2.00  1.845  0.119   1.624    2.067     195.0   1.03
midpoint  1.50  1.550  0.052   1.461    1.654     266.0   1.03
varZ      0.60  0.618  0.021   0.578    0.657     112.0   1.06
psi       0.12  0.127  0.030   0.073    0.180      84.0   1.07
```

```
L = 10.0
  d min/max = 0.478 / 9.523
  accuracy  = 0.990
  median rt = 0.1502   min rt = 0.1201

          true   mean     sd  hdi_3%  hdi_97%  ess_bulk  r_hat
mu        1.50  2.015  1.289   0.180    4.387      10.0   2.35
slope     2.00  1.543  0.614   0.133    2.020      10.0   2.49
midpoint  1.50  0.854  2.122  -5.020    2.922      10.0   2.11
varZ      0.60  0.988  0.771   0.544    3.111      12.0   1.81
psi       0.12  0.115  0.016   0.098    0.120      12.0   1.80
```

Only `psi` is close to recovered at `L=10` (it's pinned by `min(rt)`); the
other four are junk (ESS ≈ 10-12 out of 16000 draws).

## Why

For `d = 8` (near the top of the observed `[0.478, 9.523]` range at `L=10`),
with `mu = 1.5`, `varZ = 0.6` (used as the log-space SD, per
`model_lnrm2.py:182` "名字叫 varZ...這裡照樣當標準差用"):

```
correct accumulator:  log-mean = μ − d = 1.5 − 8  = −6.5   exp(−6.5) ≈ 0.0015 s
error accumulator:     log-mean = μ + d = 1.5 + 8  =  9.5   exp( 9.5) ≈ 13 000 s

P(correct) = Φ( √2·d / varZ ) = Φ( √2·8 / 0.6 ) = Φ(18.9) ≈ 1
```

(`Φ(√2·d/varZ)` because `correct ⇔ z₁<z₂`, and `z₂−z₁ ~ N(2d, 2·varZ²)`.)

```
 d = 1  (μ∓d = 0.5 / 2.5, still overlapping on a log scale)

   density
     ┤        ╭──╮ correct (~exp(0.5)≈1.6s)
     ┤       ╭╯  ╰╮         ╭──────╮ error (~exp(2.5)≈12s)
     ┤      ╭╯    ╰╮      ╭─╯      ╰─╮
     ┼──────╯───────╰────╯───────────╰──► t (s)
            1        5         12

 d = 8  (μ∓d = −6.5 / 9.5, off the map)

   density
     ┤ ▐ correct: point mass at t≈ψ+0.0015s
     ┤ ▐   (rt is essentially always this)
     ┼─▐───────────────────────────·  ·  ·  ·  ·──► t (s)
       ψ                                        error ~13000s
                                          (never wins the race, never seen)
```

No real RT data collapse to a spike at `ψ` with a second accumulator
finishing in the next geological era. Once `d` exceeds roughly 3-4, changing
`slope` or `midpoint` further changes nothing observable (accuracy is
already ≈1, RT is already ≈ψ), so the likelihood surface is flat in
`slope`/`midpoint` there — which is exactly the ESS ≈ 10-12 / R-hat > 2
degeneracy seen above, not a sampler-tuning problem.

## Questions for the author

Each ties to a `TODO-A#` in `model_lnrm2a.py`'s docstring (verified at the
lines below):

1. **TODO-A1** (`model_lnrm2a.py:31`) — Is `d = L·inv_logit(slope·(x−midpoint))`
   really the *per-accumulator* offset fed into `z₁ = μ−d, z₂ = μ+d`? Or is
   it the **total separation** between accumulators, i.e. should it be
   `z₁ = μ − d/2, z₂ = μ + d/2`? Evidence for the latter reading: the
   posterior-predictive code that reconstructs accumulator means from a
   fitted `lnrm2a` posterior does exactly this —
   `simulateLNRM_ogival.R:206-209` and `:242-251`:
   ```r
   mux <- mu[rsamp] + c(-.5, .5)*L * inv_logit(slope[rsamp]*(i - midpoint[rsamp]))
   ```
   i.e. accumulator means are `μ ∓ 0.5·L·inv_logit(...)`, not `μ ∓ L·inv_logit(...)`.
   That also matches `L`'s own name, "max separation"
   (`simulateLNRM_ogival.R:26`): `z₂−z₁` maxes out at `L` only if each side
   moves by `L/2`. If this is right, the current code is using twice the
   intended offset — `d ∈ [0.5, 9.5]` should be `d ∈ [0.25, 4.75]`, which is
   still far from the working `L=2` regime, so this alone may not close the
   gap, but it is a concrete, code-verified error to fix first.
2. **TODO-A2** (`model_lnrm2a.py:32`) — Is the symmetric `z = μ ∓ d`
   structure from `lnrm2.stan:23-24` even carried over to `lnrm2a`, or does
   `lnrm2a` use something else (e.g. only one accumulator's mean moves, or
   `d` enters through a rate/scale parameter instead of the log-mean)?
3. **TODO-A4** (`model_lnrm2a.py:34`) — Is `L` a fixed constant (`10`,
   `L_MAX_SEPARATION` at `model_lnrm2a.py:58`), passed in as `data`, or
   estimated? `adaptiveSFT_functions.R:11` (`x <- L * inv_logit(...)`)
   and `:194-195` (`l_targ.dist = logit(l_targ/10.)/slope + midpoint`) both
   treat `L=10` as a known constant, consistent with the current code, but
   neither line proves it isn't also a free parameter in `lnrm2a.stan`
   itself.
4. **TODO-A3** (`model_lnrm2a.py:33`) — `slope`/`midpoint` priors are
   currently guessed as `Normal(0,2)` each (`model_lnrm2a.py:148-149`).
   Given the actual scale of `d` (target range `[l_targ, h_targ] =
   [1.3, 8.0]`, `simulateLNRM_ogival.R:24-25`), what were the real priors?
5. **TODO-A5** (`model_lnrm2a.py:35`) — Are `mu`/`varZ`/`psi` priors really
   identical to `lnrm2.stan:30-31,17`, or did they change alongside the `d`
   curve (e.g. tighter `varZ` prior to prevent the runaway-separation
   regime above)?

## What is verified — not the problem

The `pt.Op` (`LNRM2A_PointwiseOp`, `model_lnrm2a.py:105-122`) itself is
correct: its output matches a plain numpy/scipy closed-form race likelihood
computed independently, for **both** PyMC assemblies (A: `pm.Potential`,
`model_lnrm2a.py:162-164`; B: `pm.CustomDist`, `:154-160`), at `L=10`,
`N=500`, `true_params` above:

```
scipy/numpy closed-form total logp   = 1378.6184450046246
Assembly A (pm.Potential)  logp()    = 1378.6184450046246
Assembly B (pm.CustomDist) logp()    = 1378.6184450046246
|A - scipy| = 0.0      |B - scipy| = 0.0
```

(bitwise-identical here, well inside the ~1e-12 tolerance asked for —
script: `scratchpad/logagent/verify_op.py`). So the bug, whatever it is, is
in the model specification (which curve, which offset, which priors), not
in the numba log-density code or the PyMC wiring.

## Files

- `/home/user/adaptiveSFT/model_lnrm2a.py` — model under test, TODO-A1..A5 at lines 31-35
- `/home/user/adaptiveSFT/model_lnrm2.py` — verified reference this was built from
- `/home/user/adaptiveSFT/lnrm2.stan` — original Stan model (`z` at :23-24, priors at :17,30-33)
- `/home/user/adaptiveSFT/simulateLNRM_ogival.R` — `L` at :26, `mux` reconstruction at :206-209, :242-251
- `/home/user/adaptiveSFT/adaptiveSFT_functions.R` — `getPr_ogival` (`x <-` at :11), `find_salience_ogival` (`pars` at :186, `l_targ.dist`/`h_targ.dist` at :194-195), `inv_logit` at :178
- scratch scripts (this session): `scratchpad/logagent/repro.py`, `scratchpad/logagent/verify_op.py`
