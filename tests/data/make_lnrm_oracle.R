# Stan oracle for adaptivesft/models.py (link = "quadratic" = lnrm2.stan).
# Fits the ORIGINAL model with rstan on the fixed data set in tests/data/lnrm_oracle_input.csv
# (made by make_lnrm_oracle_input.py) and stores posterior summaries. The samplers differ
# (Stan NUTS vs PyMC NUTS), so tests/test_lnrm_vs_stan.py compares means and quantiles with a
# Monte-Carlo tolerance, not bit for bit.
#
#   Rscript tests/data/make_lnrm_oracle.R            # uses stan/lnrm2_array.stan (Stan >= 2.26 syntax)
#   STAN_FILE=lnrm2.stan Rscript tests/data/make_lnrm_oracle.R   # the untouched original; needs stanc <= 2.32
#
# Set-up on a Red Hat / Rocky / Alma machine: see docs/stan_comparison_redhat.md.
suppressMessages(library(rstan))
suppressMessages(library(jsonlite))
options(mc.cores = min(4L, parallel::detectCores()))
rstan_options(auto_write = TRUE)

here <- "tests/data"
stan_file <- Sys.getenv("STAN_FILE", "stan/lnrm2_array.stan")
d <- read.csv(file.path(here, "lnrm_oracle_input.csv"))
standat <- list(N = nrow(d), intensity = d$intensity, correct = as.integer(d$correct),
                minRT = min(d$rt), rt = d$rt)                           # adaptiveSFT_functions.R:168-173

t0 <- Sys.time()
fit <- stan(file = stan_file, data = standat, pars = c("mu", "alpha", "alpha2", "varZ", "psi"),
            chains = 4, iter = 4000, warmup = 2000, seed = 20260930, refresh = 0)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
post <- extract(fit, c("mu", "alpha", "alpha2", "varZ", "psi"))
summ <- summary(fit, pars = c("mu", "alpha", "alpha2", "varZ", "psi"))$summary

out <- list(
  stan_file = stan_file, stan_version = as.character(stan_version()), rstan_version = as.character(packageVersion("rstan")),
  n_trials = nrow(d), chains = 4, iter = 4000, warmup = 2000, seconds = elapsed,
  truth = list(mu = 1.5, alpha = 0.8, alpha2 = -0.15, varZ = 0.6, psi = 0.12),
  summary = lapply(rownames(summ), function(p) list(
    name = p, mean = summ[p, "mean"], sd = summ[p, "sd"],
    q05 = unname(quantile(post[[p]], .05)), q50 = unname(quantile(post[[p]], .5)), q95 = unname(quantile(post[[p]], .95)),
    n_eff = summ[p, "n_eff"], rhat = summ[p, "Rhat"])),
  # the salience inversion the R code would do (adaptiveSFT_functions.R:229-232) on these draws
  salience = list(
    h_targ = 1.6, l_targ = 0.6,
    high = mean((-post$alpha / post$alpha2 - sqrt((post$alpha / post$alpha2)^2 + 2 / post$alpha2 * 1.6)) / 2, na.rm = TRUE),
    low  = mean((-post$alpha / post$alpha2 - sqrt((post$alpha / post$alpha2)^2 + 2 / post$alpha2 * 0.6)) / 2, na.rm = TRUE),
    n_na_high = sum(is.na((-post$alpha / post$alpha2 - sqrt((post$alpha / post$alpha2)^2 + 2 / post$alpha2 * 1.6)) / 2)),
    n_draws = length(post$alpha))
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), file.path(here, "lnrm_stan_oracle.json"))
print(round(summ[, c("mean", "sd", "2.5%", "97.5%", "n_eff", "Rhat")], 4))
cat(sprintf("wrote %s  (%.0f s, %s)\n", file.path(here, "lnrm_stan_oracle.json"), elapsed, stan_file))
