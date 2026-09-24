# adaptiveSFT — inventory of R / Stan version breakages

Scope: `adaptiveSFT_functions.R`, `psi Simulation_25JUNE2018.R`, `psi Simulation_26MAR2019.R`,
`psiSimulation_functions.R`, `simulateLNRM_ogival.R`, and `lnrm2.stan` as called from R.
Repo untouched; all scratch work in `scratchpad/ragent/`.

Legend: **CONFIRMED** = reproduced here (R 4.3.3, node 22 + stanc.js) or read in the
package source; **unverified** = could not be executed/inspected here.

## 0. Environment and method

| item | result |
|---|---|
| R | 4.3.3 (`/usr/bin/Rscript`) |
| `rstan`, `StanHeaders`, `diffIRT`, `sft`, `Rcpp`, `cmdstanr` | **not installed here** |
| `install.packages(c("diffIRT","sft","rstan"), repos="https://cloud.r-project.org")` | **failed**: `Warning: unable to access index for repository https://cloud.r-project.org/src/contrib: cannot open URL 'https://cloud.r-project.org/src/contrib/PACKAGES'` then `packages 'diffIRT', 'sft', 'rstan' are not available for this version of R` (network to CRAN blocked; `curl` to cloud.r-project.org gives `CONNECT tunnel failed, response 403`) |
| raw.githubusercontent.com / github.com | reachable; used to read the CRAN mirrors `cran/rstan`, `cran/StanHeaders`, `cran/sft`, `cran/diffIRT` and `stan-dev/docs` |
| stanc3 | ran `inst/stanc.js` from StanHeaders 2.32.10 (stanc v2.32.2) and StanHeaders 2.39.1 (stanc v2.39.0) under node v22.22.2, calling `stanc(name, code, flags)` exactly as `rstan/R/stanc.R:230` does |

Package versions read from the CRAN GitHub mirrors (latest tag / master at time of check):

| package | version | published | notes |
|---|---|---|---|
| rstan | 2.32.7 | 2025-03-10 | `Depends: StanHeaders (>= 2.32.0)`; no newer tag on the mirror |
| StanHeaders | 2.39.1 | 2026-09-02 | ships `inst/stanc.js` = stanc3 **v2.39.0** |
| sft | 2.4 | 2025-04-20 | |
| diffIRT | 1.5 | 2015-08-14 | |

## 1. Running each file (`Rscript <file>`, 60 s timeout, cwd = repo)

| file | exit | first error (verbatim) | cause |
|---|---|---|---|
| `adaptiveSFT_functions.R` | 1 | `Error in rstan_options(auto_write = TRUE) : could not find function "rstan_options"` (after 3 warnings `there is no package called 'rstan'/'diffIRT'/'sft'`) | missing packages only; `require()` does not stop, line 4 does |
| `psi Simulation_25JUNE2018.R` | 1 | `Error in setwd("C:/Users/w018elf/Google Drive/Publications/Adaptive SFT/Model") : cannot change working directory` | hard-coded Windows path (line 2) |
| `psi Simulation_26MAR2019.R` | 1 | `Error in setwd("C:/Users/w018elf/Google Drive/Projects/Adaptive SFT/Model") : cannot change working directory` | hard-coded Windows path (line 2) |
| `psiSimulation_functions.R` | 0 | (none; only function definitions) | |
| `simulateLNRM_ogival.R` | 1 | `Error in rstan_options(auto_write = TRUE) : could not find function "rstan_options"` via `source("adaptiveSFT_functions.R")` | missing packages only |

All five files `parse()` cleanly under R 4.3.3 (no syntax-level breakage).

## 2. Findings caused by version changes

### F1. `if()` with a condition of length > 1 — R 4.2.0 — CONFIRMED

(a) `adaptiveSFT_functions.R:228`
(b) `  if (post.diff$alpha2 <0) {`
(c) `post.diff` is `extract(fitModel, c("mu","alpha","alpha2","psi","varZ"))` (line 219), so
`post.diff$alpha2` is the vector of all posterior draws (default 4 chains x 1000 = 4000).
R < 4.2.0: warning `the condition has length > 1 and only the first element will be used`.
R >= 4.2.0 (NEWS 4.2.0: "Calling if() or while() with a condition of length greater than one
gives an error rather than a warning"): hard error.
(d) Reproduced (R 4.3.3, data.frame of 4000 draws): `Error: the condition has length > 1`.
(e) Minimal fix that reproduces the old *success* path exactly: drop the guard.
```diff
-  if (post.diff$alpha2 <0) {
   l_targ.dist <- with(post.diff,
         (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * l_targ)) / 2)
   h_targ.dist <- with(post.diff,
         (-alpha/alpha2 - sqrt( (alpha/alpha2)^2 + 2 / alpha2 * h_targ)) / 2)
-  }
```
(f) Numerical behaviour. Old R decided on **draw 1 only**:

```
 alpha2[1] < 0  -->  both *.dist computed for ALL draws (alpha2<0 and alpha2>0 alike),
                     then mean(..., na.rm=TRUE)            (lines 229-232, 279-280)
 alpha2[1] >= 0 -->  *.dist never assigned  -->  line 279: object 'h_targ.dist' not found
```
With the guard removed, the first branch is taken unconditionally, so every result the old
code could return is reproduced bit-for-bit; the only change is that the "first-draw lottery"
failure path disappears. Alternatives change numbers or semantics and must be a deliberate
choice by the caller:
- `if (all(post.diff$alpha2 < 0))` — same numbers when it passes, `stop()`/skip otherwise;
- `if (mean(post.diff$alpha2) < 0)` — same numbers when it passes, otherwise as above;
- `ifelse(alpha2 < 0, <formula>, NaN)` — masks positive-`alpha2` draws. For `targ > 0` those
  draws have a positive discriminant, so the old code *included* them in the mean; masking
  them therefore changes `high`/`low` whenever the posterior straddles 0.
`find_salience_polynomial()` is not called by any script in the repo (only
`find_salience_ogival` is), so nothing downstream changes today.

Latent same-class cases (currently length 1, would break if a real prior array were passed):
- `psiSimulation_functions.R:45` `  if (is.na(prior)) {` and `:249` same line; `prior <- NA` on lines 7/216.
  Reproduced: with `prior <- array(0.5, c(1,2,2,1))` -> `Error: the condition has length > 1`.
  Fix: `if (all(is.na(prior)))` or `if (identical(prior, NA))` (no numerical change while `prior` is scalar NA).

No `while()` with vector conditions and no vectorised `&&`/`||` (R 4.3.0 error) were found (grep over all five files).

### F2. Stan old array syntax `real x[N]` — removed in Stan 2.33 — CONFIRMED (parser-level)

(a) `lnrm2.stan:3,4,6,9,20` (called from `adaptiveSFT_functions.R:216` `stan(file="lnrm2.stan", ...)`)
(b)
```
3:   real intensity[N];
4:   int<lower=0,upper=1> correct[N];
6:   real<lower=0> rt[N];
9:   real square_intensity[N];
20:   real z[2,N];
```
(c) stan-dev/docs `reference-manual/removals.qmd`: "Postfix brackets array syntax ... *Removed In*: Stan 2.33".
Deprecated (warning) from Stan 2.26 (first stanc3 release in rstan, rstan NEWS 2.26).

How the R side reaches the parser (read from `cran/rstan` 2.32.7 source):

```
 stan(file=...)                      adaptiveSFT_functions.R:216
   |
   v
 rstan:::stanc()                     rstan/R/stanc.R:230
   stanc_ctx$call("stanc", name, code, flags)
   |
   v
 stanc_ctx <- V8 or QuickJSR         rstan/R/zzz.R:23-26
   sourced from
   system.file("stanc.js", package="StanHeaders")   rstan/R/zzz.R:29-30
   (rstan's own copy is used ONLY when StanHeaders == "2.26.28")
   |
   v
 StanHeaders/inst/stanc.js  ==  stanc3 of whatever StanHeaders is installed
```
So the Stan *language version* is decided by the installed **StanHeaders**, not by rstan.
rstan 2.32.7 declares only `StanHeaders (>= 2.32.0)`, and the CRAN mirror's current
StanHeaders is **2.39.1 (2026-09-02) carrying stanc3 v2.39.0**. A fresh
`install.packages("rstan")` therefore gets a parser that rejects this file. The premise
"CRAN rstan bundles Stan 2.32, so it still parses" holds only with a *pinned* StanHeaders
2.32.x (last: 2.32.10, 2024-07-15, stanc v2.32.2).

(d) Reproduced by running both `stanc.js` files on `lnrm2.stan` with flags `["allow-undefined"]`:

StanHeaders 2.32.10 / stanc v2.32.2 -> compiles, 5 warnings (one per line 3,4,6,9,20):
```
Warning in 'string', line 3, column 3: Declaration of arrays by placing
    brackets after a variable name is deprecated and will be removed in Stan
    2.33.0. Instead use the array keyword before the type. ...
```
StanHeaders 2.39.1 / stanc v2.39.0 -> `errors` non-empty; rstan would `stop()` with:
```
Syntax error in 'string', line 3, column 18 to column 19, parsing error:
     3:     real intensity[N];
                           ^
Ill-formed declaration. ";" expected after variable declaration.
  It looks like you are trying to use the old array syntax.
  Please use the new syntax:
  array[N] real intensity;
```
cmdstanr / CmdStan >= 2.33 fail the same way (same stanc3; not run here — unverified).

(e) Minimal fix (verified: the patched file compiles under both stanc v2.32.2 and v2.39.0,
only remaining warning is the pedantic "parameter psi has no priors", which is intended per the
comment on line 35):
```diff
--- lnrm2.stan
-   real intensity[N];
-   int<lower=0,upper=1> correct[N];
+   array[N] real intensity;
+   array[N] int<lower=0,upper=1> correct;
-   real<lower=0> rt[N];
+   array[N] real<lower=0> rt;
-   real square_intensity[N];
+   array[N] real square_intensity;
-   real z[2,N];
+   array[2,N] real z;
```
(f) No numerical change: identical model, only declaration syntax. Note that
`lnrm2a.stan`, `lnrm1.stan`, `lnrm0.stan` (see F7) are missing, so they cannot be checked or fixed.

### F3. rstan API arguments — rstan 2.32.7 — CONFIRMED (source read), no breakage

Calls in the repo and their status against `cran/rstan/R/rstan.R:246-268` and `R/stanfit-class.R:427-429`:

| file:line | call | status |
|---|---|---|
| `adaptiveSFT_functions.R:185-187` | `stan(file="lnrm2a.stan", data=..., pars=..., open_progress=TRUE)` | `file`, `data`, `pars`, `open_progress` all still formals of `stan()` |
| `adaptiveSFT_functions.R:210,216`; `simulateLNRM_ogival.R:62,157` | `stan(file=..., data=..., pars=...)` | valid |
| `simulateLNRM_ogival.R:101` | `stan(fit=fitDiff, data=..., pars=...)` | `fit = NA` still a formal (rstan.R:248) |
| `adaptiveSFT_functions.R:189,213,219`; `simulateLNRM_ogival.R:64,103` | `extract(fit, c("..."))` | valid; default `permuted=TRUE` returns a named list of draws (what the code indexes with `$`) |
| `simulateLNRM_ogival.R:158` | `samps0 <- extract(fitDiff0, "mu", permute=TRUE)$mu` | formal is `permuted`; `permute=` partial-matches through S4 dispatch (reproduced with a mock generic/method: returns the passed value). Works, but is fragile; rename to `permuted=TRUE`. `samps0[,2]` then assumes `mu` is a 2-column matrix in the missing `lnrm0.stan` — unverified |
| `adaptiveSFT_functions.R:4-5` | `rstan_options(auto_write = TRUE)`; `options(mc.cores = parallel::detectCores())` | valid |

Dependency change worth stating: since rstan 2.26 the parser runs in JavaScript, so a working
install now also needs `V8` or `QuickJSR` (rstan/R/zzz.R:23-26) and a C++17 toolchain; none of
these are mentioned in the repo. (d) not reproducible here (rstan not installed).

### F4. `stringsAsFactors` default FALSE — R 4.0.0 — CONFIRMED no reliance

Checked every `data.frame()` / `read.csv()`:
- `simulateLNRM_ogival.R:540-541` `Condition=rep(paste(arch,srule,sep="."), n.trials*4)` is now
  `character` (reproduced: `class(...) == "character"`). Consumer `sft::sicGroup` builds its own
  factor: `sic.R:8-9` `conditions <- sort(unique(inData$Condition)); conditions <- factor(conditions)`
  and compares with `Condition==cond` (sic.R:49) -> works for character or factor.
- `psi Simulation_26MAR2019.R:51-52,267,316,369,420,507,510,657,660,791,794,926,929`
  `read.csv(...)` results are only used numerically (`[sn,i]`, `apply(..., mean)`, `$col`); no
  `levels()`/`as.integer(factor)` anywhere (grep: no `factor(`/`levels(` in the five files).
- `adaptiveSFT_functions.R:164`, `psi Simulation_26MAR2019.R:265,311,364,416`: numeric columns only.
Fix: none needed. (f) no numerical change.

### F5. `class(x) == "matrix"` — R 4.0.0 — none found

grep for `class(` over the five files: no matches.

### F6. `sample()` RNG change — R 3.6.0 — not applicable

No `sample()`, `set.seed()`, or `RNGkind()` in any file. All randomness is `rnorm`/`runif`
(`diffIRT::simdiffT`, `simulateLNRM_ogival.R:500-508`) which the 3.6.0 `sample.kind` change
does not affect. Because no seed is ever set, none of the scripts were reproducible on any R version.

### F7. Files referenced but missing from the repo — CONFIRMED

`git ls-files` contains a single `.stan` and no `.csv`/`.Rdata`; `git log --all` shows only
`lnrm2.stan` was ever added (commit `10a86ce Initial Code Upload`).

| referenced at | string | exists |
|---|---|---|
| `adaptiveSFT_functions.R:185`, `simulateLNRM_ogival.R:62` | `"lnrm2a.stan"` (ogival model; the one every script actually uses) | **missing** |
| `adaptiveSFT_functions.R:210` | `"lnrm1.stan"` | **missing** |
| `adaptiveSFT_functions.R:216` | `"lnrm2.stan"` | present |
| `simulateLNRM_ogival.R:157` | `"lnrm0.stan"` | **missing** |
| `simulateLNRM_ogival.R:111` | `load("post95.Rdata")` (taken because `runConvergence <- FALSE`, line 9) | **missing** |
| `psi Simulation_26MAR2019.R:51-52` | `read.csv("orientation_alpha.csv")`, `"orientation_beta.csv"` (written on 42-43 in the same run, so OK only if the 629x300 simulation above completes) | not in repo |
| `psi Simulation_26MAR2019.R:267,316,369,420` | `p.or.rts.accs.csv` etc. (written just before) | not in repo |
| `psi Simulation_26MAR2019.R:507,657,791,926` | `read.csv("Psi_Simulation_SFTresults.csv")` — needs columns `Threshold,v,ter,sdv` (line 508), which the file written by `psi Simulation_25JUNE2018.R:412` does **not** contain (its `SFTresults`, line 399, has no such columns) | **missing** and schema mismatch |
| `psi Simulation_26MAR2019.R:510,660,794,929` | `read.csv("PsiDDM_Simulation_Pars.csv")` — never written by any script (only commented-out `write.csv` at 597/731/866/1001) | **missing** |
| `psi Simulation_25JUNE2018.R:3-4`, `26MAR2019.R:5-6`, `simulateLNRM_ogival.R:1` | `source("adaptiveSFT_functions.R")`, `source("psiSimulation_functions.R")` | present (but only resolve after the failing `setwd`, see F9) |

Expected rstan error for a missing model file (not reproducible here, unverified wording):
`cannot open file 'lnrm2a.stan': No such file or directory`.

### F8. Objects / functions used but never defined — CONFIRMED by grep + reproduction

| file:line | verbatim | problem | error reproduced |
|---|---|---|---|
| `adaptiveSFT_functions.R:11` | `    x <- with(postSamps, L * inv_logit(slope * (intensity - midpoint)))` | `L` is not in `pars=c("slope","midpoint","mu","varZ","psi")` (line 186) nor in the extracted list (189). `with()` falls through to the global env; `L` is defined **only** in `simulateLNRM_ogival.R:26` `L <- 10 # max separation`. Works only when that script (not just the functions file) was sourced first. | `with(list(x=1:3), L*x)` with no global `L` -> `object 'L' not found`; with `L <- 10` -> works |
| `adaptiveSFT_functions.R:98` | `                      psi=psi, mu=mu, sigmasq=sigmasqx)$value` | typo `sigmasqx` for `sigmasq`; `sigmasqx` exists only as a global inside the `PLOT` block of `simulateLNRM_ogival.R:204,213,258,281`. Only hit in the second `tryCatch` fallback. | static |
| `simulateLNRM_ogival.R:171,175,207,211,243,256,279` | `rsamp` | never assigned anywhere in the repo | static |
| `simulateLNRM_ogival.R:258,259,261,281,282,284` | `postOpt.diff` | never assigned | static |
| `simulateLNRM_ogival.R:170,174` | `polynomial_order` | only a formal of `find_salience_polynomial`; not defined at top level | static |
| `simulateLNRM_ogival.R:571-573` | `    dp[sn] <- sic.out$sic[[sn]]$SICtest$positive$statistic` | `dp`, `dn`, `micp` never initialised | `dp[1] <- 3` -> `object 'dp' not found` |
| `simulateLNRM_ogival.R:633-634` | `    dat.o <- moc_ddm(N, a.p, v.p, ter.p, sdv.p, orientation.intensity)` | `N` not defined at top level (only `Ns`, `nPerLevel`) | static |
| `psi Simulation_25JUNE2018.R:62` | `mean.pm.fun <- matrix(0, length(trials), length(x.axis))` | `x.axis` vs defined `axis.x` (line 12) | `object 'x.axis' not found` |
| `psi Simulation_25JUNE2018.R:129`, `26MAR2019.R:161` | `  pm.fun = matrix(data=NA, nrow=nsamps, ncol=length(x))` | `x` only exists inside the Psi functions | `object 'x' not found` |
| `psi Simulation_26MAR2019.R:535,539,670,674,804,808,939,943` | `psi_color_ddm(result.color, nDFP, allpars[sn,])` | called with 3 args; definitions `psiSimulation_functions.R:173` `psi_color_ddm <- function(result.color, nDFP)` and `:378` take 2 -> `unused argument` | static |

These are not R-version regressions (they failed in 2018/2019 R too) but they block any run of
`simulateLNRM_ogival.R` beyond line 26 and of the 2019 script's subject loops.
Fix for `L`: pass it explicitly, e.g.
```diff
-getPr_ogival <- function(intensity, target, range, postSamps,
-                    plotDensity=FALSE, ...) {
-    x <- with(postSamps, L * inv_logit(slope * (intensity - midpoint)))
+getPr_ogival <- function(intensity, target, range, postSamps, L = 10,
+                    plotDensity=FALSE, ...) {
+    x <- L * with(postSamps, inv_logit(slope * (intensity - midpoint)))
```
(no numerical change when `L == 10`, the only value used in the repo) and for line 98
`sigmasq=sigmasqx` -> `sigmasq=sigmasq`.

### F9. Platform / device issues surfaced by the runs (not R-version, listed for completeness)

| file:line | verbatim | observed |
|---|---|---|
| `psi Simulation_25JUNE2018.R:2`, `26MAR2019.R:2` | `setwd("C:/Users/w018elf/Google Drive/.../Model")` | `cannot change working directory` (first error of both scripts) |
| `psi Simulation_25JUNE2018.R:71,102,110` | `windows()` | Windows-only device: `could not find function "windows"` on Linux/macOS |
| `psi Simulation_26MAR2019.R:125` | `dev.off()` (device already closed at line 99) | `cannot shut down device 1 (the null device)` |
| `psi Simulation_25JUNE2018.R:62-64` | `mean.pm.fun <- matrix(0, length(trials), ...)` then `+ pm.fun1[[ix]]` (1x1000 + 300x1000) | `non-conformable arrays` (the 2019 file wraps in `data.frame()` which recycles silently, dims 300x1000) |
| `simulateLNRM_ogival.R:562,676` | `Correct=c(HH.x$x, HL$.xx, LH$.xx, LL.x$x)` | `HL$.xx` is NULL -> no error, `Correct` silently recycled (reproduced: 20-row frame from 10 values) — wrong data, not a crash |
| `adaptiveSFT_functions.R:184,209,215` | `if (anyNA(fitModel))` with a `stanfit` (S4) | `anyNA()` on S4 warns `is.na() applied to non-(list or vector) of type 'S4'` and returns FALSE -> behaves as intended, with a warning |

## 3. Summary

```
 modern R 4.2+  ──> F1  if(vector)  adaptiveSFT_functions.R:228   ERROR (reproduced)
 Stan 2.33+     ──> F2  real x[N]   lnrm2.stan:3,4,6,9,20         ERROR under stanc 2.39.0
                       (StanHeaders 2.39.1 is what CRAN rstan 2.32.7 loads today)
 R 4.0 / 3.6    ──> F4-F6 stringsAsFactors / class(matrix) / sample(): nothing to fix
 rstan API      ──> F3  all arguments still valid; `permute=` should be `permuted=`
 repo contents  ──> F7  lnrm2a/lnrm1/lnrm0.stan, post95.Rdata, input CSVs missing
 code itself    ──> F8/F9 undefined L, sigmasqx, rsamp, postOpt.diff, dp/dn/micp, N, x.axis, x;
                       3-arg calls to 2-arg psi_*_ddm; Windows paths/devices
```
Only F1 and F2 are version regressions in the strict sense; F1 is fixed by deleting the two guard lines,
F2 by five `array[...]` declarations (fix verified against stanc 2.32.2 and 2.39.0).
Everything else prevents a run regardless of version.
