# Stan oracle for adaptivesft/models.py (link = "quadratic" = lnrm2.stan).
# Fits the ORIGINAL model with Stan on the fixed data set in tests/data/lnrm_oracle_input.csv
# (made by make_lnrm_oracle_input.py) and stores posterior summaries. The samplers differ
# (Stan NUTS vs PyMC NUTS), so tests/test_lnrm_vs_stan.py compares means and quantiles with a
# Monte-Carlo tolerance, not bit for bit.
#
# Two back ends, picked automatically (or force with BACKEND=rstan / BACKEND=cmdstanr):
#   cmdstanr  needs an installed CmdStan + the cmdstanr/posterior/jsonlite packages (no C++ build of rstan)
#   rstan     needs the rstan + jsonlite packages
#
#   Rscript tests/data/make_lnrm_oracle.R                              # stan/lnrm2_array.stan (Stan >= 2.26 syntax)
#   STAN_FILE=lnrm2.stan Rscript tests/data/make_lnrm_oracle.R         # the untouched original; needs stanc <= 2.32
#   CMDSTAN=/path/to/cmdstan-2.36.0 Rscript tests/data/make_lnrm_oracle.R   # if cmdstanr cannot find CmdStan
#
# Set-up: docs/stan_comparison_redhat.md (single machine), docs/hpc_arc_tutorial.md (Arc / Slurm).
suppressMessages(library(jsonlite))
here <- "tests/data"
stan_file <- Sys.getenv("STAN_FILE", "stan/lnrm2_array.stan")
backend <- Sys.getenv("BACKEND", "")
if (backend == "") backend <- if (requireNamespace("cmdstanr", quietly = TRUE)) "cmdstanr" else "rstan"
d <- read.csv(file.path(here, "lnrm_oracle_input.csv"))
standat <- list(N = nrow(d), intensity = d$intensity, correct = as.integer(d$correct),
                minRT = min(d$rt), rt = d$rt)                           # adaptiveSFT_functions.R:168-173
pars <- c("mu", "alpha", "alpha2", "varZ", "psi")
seed <- 20260930
t0 <- Sys.time()

if (backend == "cmdstanr") {
  suppressMessages(library(cmdstanr))
  if (nzchar(Sys.getenv("CMDSTAN"))) set_cmdstan_path(Sys.getenv("CMDSTAN"))
  cat("cmdstanr", as.character(packageVersion("cmdstanr")), "CmdStan", cmdstan_version(), "at", cmdstan_path(), "\n")
  mod <- cmdstan_model(stan_file)
  fit <- mod$sample(data = standat, chains = 4, parallel_chains = min(4L, parallel::detectCores()),
                    iter_warmup = 2000, iter_sampling = 2000, seed = seed, refresh = 0)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  dm <- posterior::as_draws_matrix(fit$draws(pars))
  post <- lapply(pars, function(p) as.numeric(dm[, p])); names(post) <- pars
  sm <- fit$summary(pars)
  summ <- data.frame(mean = sm$mean, sd = sm$sd, n_eff = sm$ess_bulk, Rhat = sm$rhat, row.names = sm$variable)
  stan_ver <- cmdstan_version(); pkg_ver <- paste("cmdstanr", as.character(packageVersion("cmdstanr")))
} else {
  suppressMessages(library(rstan))
  options(mc.cores = min(4L, parallel::detectCores()))
  rstan_options(auto_write = TRUE)
  cat("rstan", as.character(packageVersion("rstan")), "Stan", stan_version(), "\n")
  fit <- stan(file = stan_file, data = standat, pars = pars, chains = 4, iter = 4000, warmup = 2000,
              seed = seed, refresh = 0)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  post <- extract(fit, pars)
  s4 <- summary(fit, pars = pars)$summary
  summ <- data.frame(mean = s4[, "mean"], sd = s4[, "sd"], n_eff = s4[, "n_eff"], Rhat = s4[, "Rhat"], row.names = rownames(s4))
  stan_ver <- stan_version(); pkg_ver <- paste("rstan", as.character(packageVersion("rstan")))
}

inv <- function(targ) (-post$alpha / post$alpha2 - sqrt((post$alpha / post$alpha2)^2 + 2 / post$alpha2 * targ)) / 2
out <- list(
  backend = backend, stan_file = stan_file, stan_version = as.character(stan_ver), package = pkg_ver,
  n_trials = nrow(d), chains = 4, iter = 4000, warmup = 2000, seconds = elapsed,
  truth = list(mu = 1.5, alpha = 0.8, alpha2 = -0.15, varZ = 0.6, psi = 0.12),
  summary = lapply(pars, function(p) list(
    name = p, mean = summ[p, "mean"], sd = summ[p, "sd"],
    q05 = unname(quantile(post[[p]], .05)), q50 = unname(quantile(post[[p]], .5)), q95 = unname(quantile(post[[p]], .95)),
    n_eff = summ[p, "n_eff"], rhat = summ[p, "Rhat"])),
  # the salience inversion the R code would do (adaptiveSFT_functions.R:229-232) on these draws
  salience = list(h_targ = 1.6, l_targ = 0.6,
                  high = mean(inv(1.6), na.rm = TRUE), low = mean(inv(0.6), na.rm = TRUE),
                  n_na_high = sum(is.na(inv(1.6))), n_draws = length(post$alpha))
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), file.path(here, "lnrm_stan_oracle.json"))
print(round(summ, 4))
cat(sprintf("wrote %s  (%.0f s, %s, %s)\n", file.path(here, "lnrm_stan_oracle.json"), elapsed, backend, stan_file))
