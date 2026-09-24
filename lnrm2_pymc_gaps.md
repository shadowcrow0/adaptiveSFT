# What in `lnrm2.stan` Has No Direct PyMC Equivalent

This document lists the parts of `lnrm2.stan` that **cannot be rewritten
line-for-line in PyMC**. For each one: the Stan code, the math it computes,
why there is no direct translation, and what PyMC forces you to write instead.

Everything marked *verified* was run in this environment:
**PyMC 5.28.5 / PyTensor 2.38.3 / Python 3.11**.

For the parts that *do* port directly, see the last section.
For the full porting notes, see `lnrm2_pymc_notes.md`.

---

## Summary

| # | Stan construct | Location | Direct PyMC equivalent? | Severity |
|---|---|---|---|---|
| 1 | `lognormal_lccdf` | `lnrm2.stan:39,43` | **No** | Model dies at init if done naively |
| 2 | `target +=` with data shifted by a parameter | `lnrm2.stan:38-43` | **No** | `CustomDist` rejects it; must use `Potential` |
| 3 | Bounded declaration as the prior | `lnrm2.stan:17` | **No** | Must write the prior explicitly |
| 4 | Per-trial `if / else` in the model | `lnrm2.stan:37-44` | **No** | Must vectorize with `pt.switch` |
| 5 | `transformed parameters` block | `lnrm2.stan:19-28` | Partial | `pm.Deterministic` is opt-in, not a block |
| 6 | Float precision in the tails | implicit | Partial | Must force `floatX = "float64"` |

Items 1 and 2 are the ones that will actually break a port. Items 3 to 6 are
translation differences that need a deliberate rewrite but are not dangerous
once you know about them.

---

## 1. `lognormal_lccdf` — no safe direct equivalent

### Stan

`lnrm2.stan:38-39` (and the mirror image at `:42-43`):

```stan
target += lognormal_lpdf (rt[tr] - psi | z[1,tr], varZ);
target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);
```

### Math

`lognormal_lccdf(y | m, s)` returns the **log survival function** of the
log-normal distribution:

```
      ln S(y; m, s)  =  ln [ 1 − F(y; m, s) ]

                                      (     ln y − m  )
      where   S(y; m, s)  =  Φ  ( − ──────────  )
                                      (        s      )

      Φ  =  standard normal CDF
```

Stan computes this **directly** from the tail, so it stays finite even when
`F(y)` is numerically 1.

### Why there is no direct translation

PyMC has `pm.logp` and `pm.logcdf` but **no `logsf` / `logccdf`**. The standard
advice is to compose them:

```python
pm.math.log1mexp(pm.logcdf(pm.LogNormal.dist(mu=m, sigma=s), y))   # WRONG
```

The sign convention is fine (*verified*: in this version `log1mexp(x)` computes
`log(1 − exp(x))` for `x ≤ 0`, so feeding `logcdf` directly is correct).
The problem is saturation:

```
   small s  ──►  F(y) rounds to exactly 1.0 in float64
            ──►  logcdf returns −0.0
            ──►  log1mexp(−0.0) = log(1 − 1) = log(0) = −inf
```

*Verified* on `y = 20, m = 0.5`:

```
   s        true ln S (scipy)     log1mexp route
   0.6         −11.046460         −11.046460       ok
   0.2         −81.307775         −81.307778       ok
   0.05      −1250.565570              −inf        WRONG
   0.033     −2865.061096              −inf        WRONG
   0.01    −31149.836613              −inf        WRONG
```

The true value is a large finite negative number; the composed route returns
`−inf`. NUTS `jitter+adapt_diag` initialization randomly lands on small `s`,
so the model fails **before sampling starts** (*verified*):

```
pymc.exceptions.SamplingError: Initial evaluation of model at starting point failed!
Starting values: {..., 'varZ_log__': array(-3.41), ...}      # varZ ≈ 0.033
Logp initial evaluation results: {..., 'race': -inf}
```

### What PyMC forces you to write

Bypass `logcdf` and evaluate the survival function through the normal CDF,
which PyTensor implements with `erfcx` and which stays accurate in the tail:

```python
def lognormal_lccdf(y, m, s):
    # ln S(y; m, s) = ln Φ( −(ln y − m) / s )
    return pm.logcdf(pm.Normal.dist(0.0, 1.0), -(pt.log(y) - m) / s)
```

*Verified* to match `scipy.stats.lognorm.logsf` at every row of the table above,
including `s = 0.01` where the value is `−31149.84`.

---

## 2. `target +=` on data shifted by a parameter — `CustomDist` rejects it

### Stan

`lnrm2.stan:38`:

```stan
target += lognormal_lpdf(rt[tr] - psi | z[1,tr], varZ);
                          ^^^^^^^^^^^^
                          observed data minus a free parameter
```

### Math

The likelihood is evaluated at `yₙ = tₙ − ψ`, where `tₙ` is data and `ψ` is a
parameter being sampled:

```
      Lₙ  =  ln f(tₙ − ψ; wₙ, s)  +  ln S(tₙ − ψ; lₙ, s)
```

Stan does not care that the argument mixes data and parameters; `target +=`
accepts any scalar expression.

### Why there is no direct translation

The natural PyMC idiom for "observed data with a custom likelihood" is
`pm.CustomDist(..., observed=...)`. But `observed=` **must be a constant**.
Passing the shifted quantity fails (*verified*):

```python
pm.CustomDist("rt_obs", ..., logp=race_logp, observed=rt - psi)
# TypeError: Variables that depend on other nodes cannot be used for
#            observed data. The data variable was: Sub.0
```

Stan has no such restriction, so this line cannot be translated as written.

### What PyMC forces you to write

Two options.

**Option A — `pm.Potential`** (closest to `target +=`):

```python
u = pt.clip(pt.constant(rt) - psi, 1e-12, np.inf)
pm.Potential("race", pt.sum(logp_winner + logccdf_loser))
```

Cost: `Potential` is not a random variable, so **no posterior predictive
sampling** and no `observed` bookkeeping. This is the trade-off Stan's
`target +=` never had to make because Stan does not do posterior predictive
sampling from the model block either.

**Option B — `pm.CustomDist` with the shift moved inside `logp`:**

```python
def race_logp(value, z_win, z_lose, s, psi):
    u = pt.clip(value - psi, 1e-12, np.inf)      # shift happens HERE
    ...
pm.CustomDist("rt_obs", z_win, z_lose, varZ, psi, logp=race_logp, observed=rt)
                                                                    #  ^^ raw rt
```

`psi` becomes an argument of `logp` instead of being subtracted before
`observed`. This is a structural rearrangement, not a translation.

---

## 3. Bounded declaration as the prior — no "type is prior" in PyMC

### Stan

`lnrm2.stan:17`:

```stan
real<lower=0,upper=minRT> psi;
```

and `lnrm2.stan:35`, the only thing the model block says about `psi`:

```stan
// psi has improper flat prior on positive reals
```

There is **no `~` statement for `psi`**. The declaration *is* the prior.

### Math

Stan maps a bounded parameter to an unconstrained one and adds the Jacobian
automatically. With no `~` statement, the density in the unconstrained space
is flat, so the induced density on `ψ` is:

```
      p(ψ)  ∝  1        for  0 < ψ < t_min
      p(ψ)  =  0        otherwise
```

i.e. a **uniform distribution on (0, t_min)** — proper, not improper, because
the interval is finite. (The comment in the source says "improper flat on
positive reals"; that describes the intent, not what the bounds actually
produce.)

### Why there is no direct translation

PyMC has no mechanism where declaring a variable's support implies its prior.
Every variable must be created from a distribution. `pm.Flat` exists but is
unbounded and has no `lower` / `upper` arguments, so "Flat plus bounds" is not
a one-liner.

### What PyMC forces you to write

```python
psi = pm.Uniform("psi", lower=0.0, upper=minRT)
```

This is mathematically identical to what the Stan declaration produces, and
PyMC adds the interval-transform Jacobian automatically just as Stan does.
But it is a rewrite: the Stan code says nothing about a uniform distribution;
you have to know that is what the bounds imply.

The same applies to `lnrm2.stan:16`, `real<lower=0> varZ;` — but there the
`~ inv_gamma(1, .1)` on line 30 already has support `(0, ∞)`, so
`pm.InverseGamma("varZ", alpha=1, beta=0.1)` carries the bound for free.

---

## 4. Per-trial `if / else` inside the model — must be vectorized

### Stan

`lnrm2.stan:36-45`:

```stan
for ( tr in 1:N) {
   if ( correct[tr] ) {
      target += lognormal_lpdf (rt[tr] - psi | z[1,tr], varZ);
      target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);
   }
   else {
      target += lognormal_lpdf (rt[tr] - psi | z[2,tr], varZ);
      target += lognormal_lccdf(rt[tr] - psi | z[1,tr], varZ);
   }
}
```

### Math

The branch selects which accumulator is the winner:

```
              ⎧ z₁ₙ   if cₙ = 1                  ⎧ z₂ₙ   if cₙ = 1
      wₙ  =   ⎨                          lₙ  =   ⎨
              ⎩ z₂ₙ   if cₙ = 0                  ⎩ z₁ₙ   if cₙ = 0

      Lₙ  =  ln f(yₙ; wₙ, s)  +  ln S(yₙ; lₙ, s)
```

### Why there is no direct translation

Stan's model block is imperative code executed once per gradient evaluation.
PyMC builds a static PyTensor graph; a Python `for` loop with an `if` on data
would either build `N` separate subgraphs (slow to compile, slow to run) or
fail outright because `correct[tr]` is a tensor, not a Python bool.

### What PyMC forces you to write

```python
ok     = pt.constant(correct == 1)
z_win  = pt.switch(ok, z1, z2)
z_lose = pt.switch(ok, z2, z1)
```

followed by a single vectorized `logp` / `lccdf` call over all `N` trials.
Same math, different shape of code.

---

## 5. `transformed parameters` block — no equivalent block

### Stan

`lnrm2.stan:19-28`:

```stan
transformed parameters {
   real z[2,N];
   for (tr in 1:N) {
      z[1,tr] = mu - alpha * intensity[tr] - alpha2 * square_intensity[tr];
      z[2,tr] = mu + alpha * intensity[tr] + alpha2 * square_intensity[tr];
   }
}
```

### Math

```
      dₙ   =  α·xₙ + α₂·xₙ²
      z₁ₙ  =  μ − dₙ
      z₂ₙ  =  μ + dₙ
```

### Why there is no direct translation

In Stan, anything declared in `transformed parameters` is **automatically
saved in the output** for every draw (here `2 × N` values per draw). PyMC has
no block with that semantics. An intermediate expression is just a tensor;
it is saved only if you wrap it in `pm.Deterministic`.

### What PyMC forces you to write

Either keep `z1`, `z2` as plain tensors (not saved — equivalent to the R caller
passing `pars=` to exclude `z`), or opt in explicitly:

```python
z1 = pm.Deterministic("z1", mu - d)
z2 = pm.Deterministic("z2", mu + d)
```

Not a correctness issue, but the default behaviour is inverted: Stan saves
unless told not to; PyMC does not save unless told to.

---

## 6. Tail precision — must force float64

### Stan

Implicit. Stan computes in `double` throughout.

### Math

The survival term in the tail:

```
      ln S(y)  ≈  −1250  to  −31149     (from the table in section 1)
```

### Why there is no direct translation

PyTensor's default `floatX` is `float32`. The log-survival values above, and
the `logcdf → log1mexp` composition, lose precision or saturate earlier in
`float32`. Stan never exposes this choice.

### What PyMC forces you to write

```python
import pytensor
pytensor.config.floatX = "float64"   # before importing pymc
```

*Verified*: with `float64`, per-trial log-likelihood matches a scipy reference
with maximum absolute error `4e-8` to `2e-6` (relative `≈ 3e-8`), i.e. pure
floating-point rounding.

---

## What ports directly

For honesty, the parts that translate one-to-one with no surprises:

| Stan | PyMC | Note |
|---|---|---|
| `varZ ~ inv_gamma(1, .1)` (`:30`) | `pm.InverseGamma("varZ", alpha=1, beta=0.1)` | Same (shape, scale) parameterization |
| `mu ~ normal(0,1)` (`:31`) | `pm.Normal("mu", 0, 1)` | |
| `alpha ~ normal(0,2)` (`:32`) | `pm.Normal("alpha", 0, 2)` | |
| `alpha2 ~ normal(0,1)` (`:33`) | `pm.Normal("alpha2", 0, 1)` | |
| `lognormal_lpdf(y \| m, s)` (`:38`) | `pm.logp(pm.LogNormal.dist(mu=m, sigma=s), y)` | Pass `varZ` as `sigma`, not squared |
| `square_intensity = square(intensity)` (`:10`) | `intensity**2` in numpy | Precompute outside the model |
| Automatic Jacobian for bounds | Automatic in PyMC too | Both add it; nothing to write |

One naming trap carries over unchanged: `varZ` sits in the **standard deviation**
slot of `lognormal_lpdf`, so in PyMC it must be passed as `sigma=varZ`, never
`sigma=varZ**0.5`. See `lnrm2_stan_explained.md` section 8.
