# adaptiveSFT environment check — 2026-09-24

Machine: Linux 6.18.44, Ubuntu 24.04 userland. Repo: /home/user/adaptiveSFT (untouched).
Logs: /tmp/claude-0/-home-user-adaptiveSFT/506a76af-a9ad-5127-9032-3d0f51e9a75b/scratchpad/envcheck/

Verdict in one line: **Python side runs (all 3 PyMC scripts complete); R side cannot run — the 3 needed CRAN packages are absent and CRAN is blocked by the egress proxy.**

```
                 R side                              Python side
  ┌──────────────────────────────┐        ┌────────────────────────────────┐
  │ R 4.3.3 + g++ 13.3  OK       │        │ venv py3.11 + PyMC 5.28.5  OK  │
  │ rstan / diffIRT / sft  MISSING│        │ model_lnrm2.py    runs (16 s)  │
  │ install.packages() ──> CRAN  │        │ model_lnrm2a.py   runs (8 s)   │
  │       │                      │        │ lnrm2_pymc.py     runs (21 s)  │
  │       X  proxy CONNECT 403   │        │ adaptive_sft2.py  needs pystan │
  │ (cloud.r-project.org denied) │        │   pystan 2.19 src build fails  │
  └──────────────────────────────┘        └────────────────────────────────┘
```

## R

### 1. Interpreter / toolchain
- `which R Rscript` -> `/usr/bin/R`, `/usr/bin/Rscript`
- `R --version | head -1` -> `R version 4.3.3 (2024-02-29) -- "Angel Food Cake"`
- `which g++ make gcc gfortran` -> all present; `g++ --version | head -1` -> `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
  (C++ toolchain for rstan exists.)

### 2. Packages required by the *.R files
`grep -hoE '(library|require)\(...' *.R` — only three non-base packages are used:

| package | used in | `requireNamespace()` |
|---|---|---|
| rstan | adaptiveSFT_functions.R:1 | FALSE (not installed) |
| diffIRT | adaptiveSFT_functions.R:2, psi Simulation_*.R | FALSE (not installed) |
| sft | adaptiveSFT_functions.R:3, psi Simulation_*.R, simulateLNRM_ogival.R:311 | FALSE (not installed) |

Also checked: StanHeaders FALSE, Rcpp FALSE, V8 FALSE. `installed.packages()` lists 29 packages, all base + recommended
(KernSmooth MASS Matrix boot class cluster codetools foreign lattice mgcv nlme nnet rpart spatial survival + base). No CRAN add-ons at all.

### 3. Install attempts (one each, 300 s timeout) — all fail in ~1 s
Command: `Rscript -e 'install.packages("PKG", repos="https://cloud.r-project.org")'`
Identical output for rstan, diffIRT, sft (logs: install_rstan.log, install_diffIRT.log, install_sft.log):

```
Installing package into '/usr/local/lib/R/site-library'
(as 'lib' is unspecified)
Warning: unable to access index for repository https://cloud.r-project.org/src/contrib:
  cannot open URL 'https://cloud.r-project.org/src/contrib/PACKAGES'
Warning message:
package 'rstan' is not available for this version of R
```

Root cause (see Network): the egress proxy refuses the CONNECT to cloud.r-project.org:443 with 403.
Not an R problem, not a version problem.

### 4. source / parse
- `Rscript -e 'source("adaptiveSFT_functions.R")'` -> exit 1. First error verbatim:
  ```
  Error in rstan_options(auto_write = TRUE) :
    could not find function "rstan_options"
  Calls: source -> withVisible -> eval -> eval
  In addition: Warning messages:
  1: ... there is no package called 'rstan'
  2: ... there is no package called 'diffIRT'
  3: ... there is no package called 'sft'
  ```
  (adaptiveSFT_functions.R:4 calls `rstan_options()` immediately after the `require()`s.)
- `Rscript -e 'parse("adaptiveSFT_functions.R"); parse("psiSimulation_functions.R"); parse("simulateLNRM_ogival.R"); cat("parse ok\n")'` -> exit 0, `parse ok`.
- Extra: both `psi Simulation_25JUNE2018.R` and `psi Simulation_26MAR2019.R` also parse OK. So the R code is syntactically valid under R 4.3.3; only the package dependencies are missing.

### 5. Could a Stan compiler work at all on the R side?
- `which node; node --version` -> `/opt/node22/bin/node`, `v22.22.2` (node exists; rstan >= 2.26 uses it to run stanc.js)
- `find / -name stanc.js` -> nothing found (no StanHeaders anywhere)
- `ls ~/.cmdstan` -> `No such file or directory`; `which stanc` -> not found
- V8 R package -> FALSE
Conclusion: the toolchain prerequisites (g++, make, node) are all present, so rstan *would* compile lnrm2.stan if the package
could be installed. The only blocker is getting rstan/StanHeaders onto the machine.

## Python

### 6. System python
- `which python3` -> `/usr/local/bin/python3`; `python3 --version` -> `Python 3.11.15`
- `python3 -c "import numpy"` -> `ModuleNotFoundError: No module named 'numpy'` (system python has nothing installed)
- Other interpreters present: /usr/bin/python3.10, 3.11, 3.12 (`python3.12 --version` -> `Python 3.12.3`), 3.13.

### 7. Scratch venv
`.../scratchpad/venv/bin/python` -> Python 3.11.15
`import pymc, pytensor, numba, arviz, scipy, numpy` -> versions:
`pymc 5.28.5  pytensor 2.38.3  numba 0.65.1  arviz 0.23.4  scipy 1.17.1  numpy 2.4.6`
Note: pytensor warns `could not link to a BLAS installation` (pip install) — slower, not a failure.

### 8. Runs from the repo dir with the venv python

| command | exit | wall | result |
|---|---|---|---|
| `python -c "import model_lnrm2"` | 0 | — | OK |
| `python -c "import model_lnrm2a"` | 0 | — | OK |
| `python -c "import lnrm2_pymc"` | 0 | — | OK |
| `python model_lnrm2.py` (no argparse; `--help`/`--step` ignored, hard-coded N=1000, tune=3000, draws=3000, chains=8 at model_lnrm2.py:228) | 0 | 16.4 s | completes, DEMetropolisZ, r_hat 1.01 |
| `python model_lnrm2a.py --n 300 --tune 300 --draws 300 --chains 4` | 0 | 7.8 s | completes; r_hat 2–4 at this tiny size (expected); one `RuntimeWarning: overflow encountered in exp` from pymc metropolis.py:1198 — warning only |
| `python lnrm2_pymc.py` (no argparse; `__main__` at line 203 runs N=500, NUTS 4 chains × 1000/1000) | 0 | 21.5 s | completes, r_hat 1.00, divergences 0 |
| `python -c "import adaptive_sft2"` | 1 | — | FAILS (below) |

adaptive_sft2 verbatim:
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/user/adaptiveSFT/adaptive_sft2.py", line 2, in <module>
    import pystan
ModuleNotFoundError: No module named 'pystan'
```
Only `model_lnrm2a.py` has argparse (`--n --tune --draws --chains`). `model_lnrm2.py --help` and `lnrm2_pymc.py --help` just ran the full default script.

Last 15 lines of each run are in run_model_lnrm2.log, run_model_lnrm2a.log, run_lnrm2_pymc.log.

### 9. pip installability (PyPI reachability)
- `pip download --no-deps -d pipdl pystan==2.19.1.1` (120 s) -> exit 1. It **did download** `pystan-2.19.1.1.tar.gz (16.2 MB)` (so PyPI works), then failed building metadata:
  ```
  Preparing metadata (pyproject.toml): finished with status 'error'
  error: subprocess-exited-with-error
      Cython>=0.22 and NumPy are required.
  error: metadata-generation-failed
  ```
  i.e. pystan 2.19 has no wheel for Python 3.11; a source build needs Cython + NumPy in the build env (and 2.19 is known not to build against modern NumPy/Cython 3). Not a network problem.
- `pip download --no-deps cmdstanpy` -> exit 0, `Saved .../pipdl/cmdstanpy-1.3.0-py3-none-any.whl`. PyPI + files.pythonhosted.org are reachable.
  (cmdstanpy still needs a CmdStan build; CmdStan is fetched from github.com releases — github.com answered HTTP 400 to a bare curl, i.e. reachable through the proxy, but installing CmdStan was not attempted: not run.)

### 10. Python versions / PyMC 6
python3.12 (3.12.3) and python3.13 exist at /usr/bin, so a PyMC 6 venv (needs >= 3.12) is possible in principle. Not attempted: not run.

## Network

```
  R / curl ──HTTPS_PROXY=http://127.0.0.1:46653──> agent proxy ──CONNECT──> egress gateway
                                                                              │
                          cloud.r-project.org:443  ─────────────────────── 403 (policy denial)
                          cran.r-project.org / cran.rstudio.com /
                          packagemanager.posit.co  ───────────────────────  000 (blocked)
                          pypi.org / files.pythonhosted.org  ── in noProxy list ── direct, 200
                          github.com  ─────────────────────────────────────  400 (reachable)
```

- `curl -sS -o /dev/null -w '%{http_code}\n' https://cloud.r-project.org/src/contrib/PACKAGES` ->
  `curl: (56) CONNECT tunnel failed, response 403` / `000`
- Same for cran.r-project.org, cran.rstudio.com, packagemanager.posit.co -> `000`.
- `curl -sS "$HTTPS_PROXY/__agentproxy/status"` -> `recentRelayFailures` entries:
  `"kind": "connect_rejected", "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)", "host": "cloud.r-project.org:443"` (repeated).
  `noProxy` explicitly lists `pypi.org, files.pythonhosted.org, registry.npmjs.org, index.crates.io, proxy.golang.org` — PyPI bypasses the proxy entirely; CRAN is not on the allow-list.
- /root/.ccr/README.md, section "403 / 407 from the proxy": "The destination host is not allowed by your organization's egress policy for this session. Do not retry or route around it — report the blocked host." -> Blocked host to report: **cloud.r-project.org (and every other CRAN mirror tried)**.
- PyPI: `curl https://pypi.org/simple/pystan/` -> 200.

## Resources
- `df -h /home/user /tmp` -> `/dev/vda 252G 8.2G 29G 22% /` (29 GB free; both paths on the same root fs)
- `nproc` -> 4
- `free -h` -> `Mem: 15Gi total, 1.2Gi used, 14Gi available`
(Note: PyMC reports "8 chains in 2 jobs" / "4 chains in 2 jobs" — the scripts cap cores at 2.)

## Summary table

| component | works? | blocker (verbatim first error line) |
|---|---|---|
| R 4.3.3 interpreter | yes | — |
| C++ toolchain (g++ 13.3, make, gfortran) | yes | — |
| R: parse of all 5 .R files | yes | — |
| R: `source("adaptiveSFT_functions.R")` | NO | `Error in rstan_options(auto_write = TRUE) : could not find function "rstan_options"` |
| R pkg rstan | NO (not installed) | `Warning: unable to access index for repository https://cloud.r-project.org/src/contrib:` |
| R pkg diffIRT | NO (not installed) | same |
| R pkg sft | NO (not installed) | same |
| CRAN reachability | NO | `curl: (56) CONNECT tunnel failed, response 403` — proxy status: `gateway answered 403 to CONNECT (policy denial or upstream failure)`, host cloud.r-project.org:443 |
| Stan compiler on R side (stanc.js / cmdstan / stanc) | NO (nothing installed) | `find / -name stanc.js` -> empty; `ls ~/.cmdstan` -> `No such file or directory` (node v22.22.2 is available, so rstan would work once installed) |
| System python3 (3.11.15) + numpy | NO | `ModuleNotFoundError: No module named 'numpy'` |
| venv python 3.11.15 + pymc 5.28.5 / pytensor 2.38.3 / numba 0.65.1 / arviz 0.23.4 / scipy 1.17.1 / numpy 2.4.6 | yes | — (BLAS warning only) |
| `import model_lnrm2` / `model_lnrm2a` / `lnrm2_pymc` | yes | — |
| `python model_lnrm2.py` (defaults) | yes, 16 s | — |
| `python model_lnrm2a.py --n 300 --tune 300 --draws 300 --chains 4` | yes, 8 s | — (r_hat bad at this size, expected) |
| `python lnrm2_pymc.py` (defaults) | yes, 21 s | — |
| `import adaptive_sft2` | NO | `ModuleNotFoundError: No module named 'pystan'` |
| `pip download pystan==2.19.1.1` | NO (download OK, build fails) | `Cython>=0.22 and NumPy are required.` -> `error: metadata-generation-failed` |
| `pip download cmdstanpy` | yes | — (`cmdstanpy-1.3.0-py3-none-any.whl` saved; CmdStan itself not installed: not run) |
| PyPI reachability | yes | — (pypi.org / files.pythonhosted.org are in the proxy noProxy list) |
| python3.12 for PyMC 6 | present (3.12.3) | PyMC 6 install not run |
| Disk / CPU / RAM | 29 GB free, 4 cores, 15 GB RAM | — |
