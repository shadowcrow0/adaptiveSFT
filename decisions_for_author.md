# Two open decisions in adaptiveSFT — what the code says, what it produces, and what has to be decided

Date: 2026-09-27. Line numbers refer to the files in this repository as of commit `445f6ec`;
`diffIRT` and `sft` line numbers refer to the CRAN sources (github.com/cran). Everything quoted
under "original code" is verbatim. Numbers under "actual output" were computed in this session
(Python port in `adaptivesft/`, or the R scripts' own arithmetic); numbers under "expected output"
are what the surrounding code and comments assume.

**Status (2026-10-02).** The author chose **B** for the real SFT experiment (Visual_AudioWM,
`VAWM_calibrate.py`, `METHOD = "lnrm"`): salience is calibrated on the LNRM branch with targets in
**drift-rate separation** (`h_targ` / `l_targ`, A.6 item 3), and the ogival model follows this
repository's rebuild (B.6 item 4: `½·L·inv_logit`, `L = 10` fixed, `Normal(0, 2)` priors, raw
intensity). The quadratic `lnrm2` (the file that exists, verified against Stan) is the default
link; `LINK = "ogival"` switches to `lnrm2a` under the conventions above. Decision A is left as
documented: it only affects the DDM simulations and the accuracy-target Psi branch, which the
experiment no longer uses for salience. B.6 items 1–3 remain unknowable without the lost files.

---

## Decision A — the DDM boundary parameter, and why the .99 salience level lands outside the stimulus range

### A.1 Original code

The Psi simulation drives a Ratcliff DDM observer through `diffIRT::simdiffT`
(`psiSimulation_functions.R:93-105`):

```r
  #estimated parameters from human data using psychometric function
  #thres50 = estimated intensity value to obtain 50% accuracy
  thres50 = 6

  # Ratcliff parameters
  threshold=1.45
  v= 1.6
  ter=.1
  sdv=.25

  ### Simulate the trials
  for (trial in 1:nTrials) {

    scaled.intensity <- (next.intensity-thres50)/(x.range[2]-thres50)
    DDMresult <- simdiffT(1,threshold,scaled.intensity*v,sdv,ter)
```

with the colour stimulus range `x.range <- c(-55,50)` (`psiSimulation_functions.R:15`).

The "true" psychometric function that the estimated Psi parameters are compared against is
computed in `psi Simulation_26MAR2019.R:117`:

```r
DDM.pCorrect = exp(threshold*scaled.intensity*v) / (exp(-1*threshold*scaled.intensity*v) + exp(threshold*scaled.intensity*v))
```

which simplifies to

```
   P(correct) = 1 / (1 + exp(−2 · threshold · scaled.intensity · v))
```

The two salience levels are then read off the fitted Psi parameters at 99% and 90%
(`psiSimulation_functions.R:171, 183-184`):

```r
inv.pm.function <- function (y,a,b,d) qnorm((y-.5*d)/(1-d), a, b)
  ...
  highSalience.color <- inv.pm.function(.99, result.color$alpha[...], result.color$beta[...], sim.d)
  lowSalience.color  <- inv.pm.function(.90, result.color$alpha[...], result.color$beta[...], sim.d)
```

What `simdiffT` actually does with its second argument (`diffIRT/R/simdiffT.r:5-6`,
documented in `diffIRT/man/simdiffT.Rd:16` as "`a`: boundary separation"):

```r
        drift=rnorm(1,mv,sv)
        p[jj]=exp(a*drift)/(1+exp(a*drift))
```

i.e.

```
   P(upper boundary) = 1 / (1 + exp(−a · drift))
```

### A.2 Purpose

`threshold`, `v`, `thres50` and `x.range` together define the simulated observer's psychometric
function in stimulus units. The design intent visible in the code is: the 50% point sits at
`thres50`, and the top of the stimulus range (`x.range[2]`) is where the observer is at
ceiling, so that Psi can locate a 99% level (H) and a 90% level (L) inside the range.

### A.3 Expected output vs actual output

Expected (from the comment on `:92-93` and from the fact that `.99` is requested): H and L both
inside `[-55, 50]`, with H near the top of the range.

Actual, as hard-coded by the author after running the simulation
(`psi Simulation_25JUNE2018.R:174, 177`):

```r
highSalience.color = 101.63970 #COMMENT THIS LINE IF YOU RERAN A SIMULATION ABOVE
lowSalience.color = 53.64586 #COMMENT THIS LINE IF YOU RERAN A SIMULATION ABOVE
```

Both are above the range maximum of 50. In the 2019 revision the values were replaced by hand
(`psi Simulation_26MAR2019.R:208, 211, 234, 237`):

```r
highSalience.color = 50 #COMMENT THIS LINE IF YOU RERAN A SIMULATION ABOVE
lowSalience.color = 16.45 #DATA FROM SIMULATED PARTICIPANT 1 OF DFP ...
highSalience.orientation = 90 #75.50087 #COMMENT THIS LINE IF YOU RERAN A SIMULATION ABOVE
lowSalience.orientation = 70.81 #DATA FROM SIMULATED PARTICIPANT 1 OF DFP ...
```

i.e. H was clipped to the range maximum (50 and 90 are exactly `x.range[2]` for colour and
orientation).

### A.4 Where the difference comes from

The two formulas differ by a factor of 2 in the exponent. Evaluated at the parameter values in
the script (`a = 1.45`, `v = 1.6`, `thres50 = 6`, colour range top 50):

| x | P(correct), `simdiffT` (`1/(1+e^{−a·drift})`) | P(correct), line 117 (`1/(1+e^{−2a·drift})`) |
|---|---|---|
| 50 (range top) | 0.9105 | **0.9904** |
| 53.65 (author's L) | 0.9250 | 0.9935 |
| 101.64 (author's H) | 0.9936 | 1.0000 |

| target | x under `simdiffT` | x under line 117 |
|---|---|---|
| 0.99 | **93.1** | **49.6** |
| 0.90 | 47.7 | 26.8 |

Under the line-117 formula the design is self-consistent: 99% falls at x = 49.6, essentially the
range top, which is why the 2019 revision hard-codes 50 and 90. Under what `simdiffT` actually
simulates, 99% needs x ≈ 93 for colour, far outside the range, which is exactly the 101.6 the
2018 run produced (Psi's estimate of a shallower curve, plus estimation noise). The same holds
for orientation: at x = 90 the observer is at 0.9105, not 0.99.

So the anomaly is not in Psi and not in `inv.pm.function`; it is that the observer being
simulated (`simdiffT.r:6`) is half as sensitive as the observer the script believes it is
simulating (`26MAR2019.R:117`).

### A.5 What the variable relates to

In the Ratcliff model with boundary separation `a`, starting point `a/2` and diffusion
coefficient 1,

```
   P(upper) = 1 / (1 + exp(−a · v))
```

If instead `a` denotes the distance from the starting point to either boundary (the usual meaning
of the word *threshold*, and the name the script uses: `threshold=1.45`), the separation is `2a`
and

```
   P(upper) = 1 / (1 + exp(−2 · a · v))
```

which is line 117. `diffIRT` uses the first convention (`a` = separation, `simdiffT.Rd:16`).
The script's variable name and its line 117 use the second. Everything downstream depends on
which one is meant: the psychometric slope of the simulated observer, whether 99% is reachable
inside `x.range`, the Psi β estimate (and in `AGRT.py` whether β hits the grid ceiling derived
from the range), the H/L intensities, and the accuracy/RT separation that the DFP and SIC
stage sees.

### A.6 What has to be decided

1. **Which convention is intended for `threshold` (= `a`)?**
   - *Separation* (diffIRT's): then line 117 is wrong and should read
     `1/(1+exp(-threshold*scaled.intensity*v))`; the intended "99% at range top" design requires
     either `threshold ≈ 2.9` or `v ≈ 3.2`, or a wider `x.range`.
   - *Distance to boundary* (the script's): then every `simdiffT(..., threshold, ...)` call
     (`psiSimulation_functions.R:105, 200-201, 310, 405-406`, `adaptiveSFT_functions.R:120-125,
     159`) should pass `2*threshold`, and the published simulations were run with half the
     intended sensitivity.
2. **What the H level should be when 99% is unreachable inside the range**: clip to `x.range[2]`
   (what the 2019 revision does by hand), lower the H target (e.g. .95), or widen the range.
3. **Whether the salience targets for the real experiment are defined in accuracy** (this Psi
   branch: .99/.90) **or in drift-rate separation** (the LNRM branch: `h_targ`/`l_targ`, Decision B).
   The two are only interchangeable through the observer's noise parameter (`varZ` in the LNRM,
   `a`·`v` in the DDM), which is exactly the quantity in question here.

The Python port (`adaptivesft/ddm.py`) follows `simdiffT` (`ddm_p_correct`) and reports a warning
when a requested level falls outside the range (`adaptivesft/psi.py::salience_levels`); it does
not silently apply either fix.

---

## Decision B — the ogival LNRM (`lnrm2a.stan`): per-accumulator offset vs total separation, and the priors

### B.1 Original code

`lnrm2a.stan` is referenced but not in the repository (it has never been in any commit). Its
behaviour has to be inferred from how the R code uses its posterior. Three places:

Difficulty curve and the constant `L` (`adaptiveSFT_functions.R:9-11`, `simulateLNRM_ogival.R:24-26`):

```r
getPr_ogival <- function(intensity, target, range, postSamps,
                    plotDensity=FALSE, ...) {
    x <- with(postSamps, L * inv_logit(slope * (intensity - midpoint)))
```

```r
l_targ <- 1.3#.7
h_targ <- 8.0#1.4
L <- 10 # max separation
```

Inversion from target to intensity (`adaptiveSFT_functions.R:180-199`):

```r
find_salience_ogival <- function(dat, h_targ, l_targ, fitModel=NA) {
  ...
    fitModel <- stan(file="lnrm2a.stan", data=standatDiff,
                     pars=c("slope", "midpoint", "mu", "varZ", "psi"),
                     open_progress=TRUE)
  ...
  l_targ.dist = logit(l_targ / 10.) / slope + midpoint
  h_targ.dist = logit(h_targ / 10.) / slope + midpoint
```

Reconstruction of the two accumulator means from the posterior, used for the posterior-predictive
plots (`simulateLNRM_ogival.R:206-209`):

```r
     mux <- with(post.diff, t(
                 matrix(c(1,1), 2, 1) %*%mu[rsamp]  +
                 matrix(c(-.5,.5)*L, 2, 1) %*%
                 inv_logit(slope[rsamp] * (i - midpoint[rsamp]))))
```

For comparison, the accumulator means in the model that *does* exist (`lnrm2.stan:23-24`):

```stan
      z[1,tr] = mu - alpha * intensity[tr] - alpha2 * square_intensity[tr];
      z[2,tr] = mu + alpha * intensity[tr] + alpha2 * square_intensity[tr];
```

### B.2 Purpose

`lnrm2a` replaces the quadratic difficulty curve of `lnrm2` with a logistic (ogival) one so that
the separation between the correct and incorrect accumulators saturates at `L` instead of growing
without bound. `find_salience_ogival` inverts that curve to find the intensities at which the
separation equals `h_targ` and `l_targ`.

### B.3 Expected output vs actual output

Expected: with the difficulty curve swapped and everything else as in `lnrm2.stan`, simulated data
at `L = 10` should be recoverable, and `h_targ = 8.0`, `l_targ = 1.3` should invert to sensible
intensities.

Actual (`log.md`, N = 1000, 8 chains × 3000):

| construction of z | simulated accuracy | recovery |
|---|---|---|
| `z = mu ∓ L·inv_logit(...)` (offset = full L) | 0.99, median RT collapsed to ψ + 0.15 s | slope / midpoint not recovered, R-hat > 2, ESS ≈ 10 |
| `z = mu ∓ ½·L·inv_logit(...)` (offset = L/2) | 0.965 | all five parameters recovered, R-hat ≤ 1.01 |

And, separately, the second rebuild in `Visual_AudioWM/adaptivesft` (which estimates `L` as a free
parameter `D` instead of fixing 10) finds that on its own simulated data `h_targ = 8.0` and
`l_targ = 1.3` are unreachable for > 99% of posterior draws (`salience.py:24-27` there), because
the estimated `D` is ≈ 1, not 10.

### B.4 Where the difference comes from

Two independent things, both of which must be pinned down by the missing file:

1. **The ½.** `L` is described as "max separation" (`simulateLNRM_ogival.R:26`), `find_salience`
   inverts `logit(targ / 10)` — both treat `L` as the *total* distance `z[2] − z[1]`. The plot
   code (`:206-209`) is consistent with that: each accumulator is offset by `±½·L·inv_logit`. But
   `getPr_ogival` (`:11`) uses `L·inv_logit` as if it were the per-accumulator offset, and a naive
   transcription of `lnrm2.stan:23-24` with `d = L·inv_logit` puts the full `L` on each side, so
   the total separation becomes `2L = 20`. At `d = 8` the correct accumulator's mean finishing
   time is `exp(mu − 8) ≈ 1.5 ms` and the incorrect one's `exp(mu + 8) ≈ 13 000 s`: every trial is
   correct and instantaneous, the likelihood is flat in `slope`/`midpoint`, and the sampler cannot
   move. That is the degenerate row above.

2. **The scale of `L` and the priors.** With the ½ in place, `h_targ = 8.0` means "separation 8
   out of a maximum 10" — a strong effect — and `l_targ = 1.3` a weak one. Whether those numbers
   are meaningful depends on `L` being *fixed* at 10 and on the priors for `slope` and `midpoint`,
   neither of which is recoverable from the R code (`pars=` on `:186` lists them as sampled
   parameters, but `L` is not in `pars`, so it is either `data` or a literal inside the Stan file).
   The two Python rebuilds diverge exactly here:

   | | `model_lnrm2a.py` (this repo) | `Visual_AudioWM/adaptivesft` |
   |---|---|---|
   | offset | `½·L·inv_logit` | `½·D·inv_logit` |
   | `L` / `D` | fixed 10 (`simulateLNRM_ogival.R:26`) | estimated, `HalfNormal(2)` |
   | `slope`, `midpoint` priors | `Normal(0, 2)` each (guess) | `HalfNormal(5)`, `Normal(0, 1)` on standardized intensity |
   | intensity | raw | standardized (z-scores) |
   | targets | `h_targ`/`l_targ` in separation units | accuracy (converted through `varZ`) |

### B.5 What the variable relates to

`L` is the asymptote of `z[2] − z[1]`; through the shared log-scale SD `varZ` it fixes the maximum
attainable accuracy of the simulated observer:

```
   P(correct | x) = Φ( (z[2] − z[1]) / (varZ · √2) )
```

so at `L = 10`, `varZ = 0.6`: `Φ(10 / 0.85) ≈ 1`; at `L = 2`: `Φ(2.36) ≈ 0.991`. `h_targ` and
`l_targ` are points on that same separation axis, so their meaning in accuracy terms changes
with `L` and `varZ`. That is why the Psi branch (accuracy targets) and the LNRM branch
(separation targets) cannot be reconciled without fixing these numbers (see Decision A.6, item 3).

### B.6 What has to be decided

1. **Confirm the `transformed parameters` block of `lnrm2a.stan`**: is it
   `z = mu ∓ 0.5 * L * inv_logit(slope * (intensity − midpoint))`? (The evidence above says
   yes; only the file can confirm.)
2. **Is `L` a literal 10 in the Stan file, passed as `data`, or a parameter?** If it is a
   parameter, what is its prior, and what were the fitted values in the original simulations
   (the `post95.Rdata` and the `*_SFTresults.csv` files that would show this are also missing)?
3. **What are the priors on `slope` and `midpoint`?**
4. **Which rebuild is canonical going forward.** The Python port in this repository follows this
   repository's conventions (`adaptivesft/models.py`: `½·L·inv_logit`, `L = 10` fixed,
   `Normal(0, 2)` priors, raw intensity, separation-unit targets with an accuracy converter);
   the alternative is documented above. This choice changes the numbers `find_salience` returns
   and must be made before any salience values are used in an experiment.
